# evaluate_retrieval.py
import sys
import os
import json
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk
from nltk.tokenize import word_tokenize
# 将项目根目录（RAG）添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_generator import create_image_embeddings
from image_embedding_query import load_index, find_most_similar_report, generate_image_embedding
from report_extractor import load_json_data
from faiss_indexer import load_faiss_index
from config import Config

# 语义重排模型
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("/home/lijiaji/RAG/ms-marco-MiniLM-L6-v2")

def semantic_rerank(query, candidates):
    """使用更精细的语义模型重排"""
    pairs = [(query, cand) for cand in candidates]
    scores = reranker.predict(pairs)
    return [candidates[i] for i in np.argsort(scores)[::-1]]

def compute_recall_at_k(retrieved_ids, true_ids, k=5):
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(true_ids))
    return hits / len(true_ids) if true_ids else 0.0

def load_test_set(json_path, split="test"):
    """加载测试集（每个报告包含两张图像）"""
    test_data = load_json_data(json_path, split=split)
    return test_data

def evaluate_system(test_data, index_path, mapping_path, json_file_path, top_k=50):
    """批量评估整个系统"""
    index = load_faiss_index(index_path)
    index_mapping = np.load(mapping_path)
    report_data = load_json_data(json_file_path, split="test")
    true_id_map = np.load(mapping_path.replace(".npy", "_true_ids.npy"))
    
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}
    rouge_scores = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    meteor_scores = []
    rouge = Rouge()
    recall_5_scores = []
    recall_10_scores = []
    smooth_fn = SmoothingFunction().method1

    for report in tqdm(test_data, desc="Evaluating Reports"):
        true_relevant_id = str(report["id"])  # 保证ID为字符串
        true_report = report["report"]
        image_paths = report["image_path"]
        
        # 用于聚合各图像的检索结果
        candidate_dict = {}  # key: 真实报告ID, value: {'score': 累计距离, 'reports': [候选文本]}
        
        for img_path in image_paths:
            try:
                full_img_path = os.path.join(Config.IMAGE_BASE_DIR, img_path)
                query_embedding = generate_image_embedding(full_img_path)
                query_embedding = query_embedding.reshape(1, -1).astype('float32')
                query_embedding = query_embedding / np.linalg.norm(query_embedding)

                # 输出调试信息
                print(f"Image: {full_img_path}, embedding norm: {np.linalg.norm(query_embedding):.4f}, shape: {query_embedding.shape}")
                D, I = find_most_similar_report(query_embedding, index, top_k)
                print(f"For image: {img_path}, D: {D}, I: {I}")
                for dist, idx in zip(D[0], I[0]):
                    # 过滤无效索引和距离过大的候选（返回值大约为 np.finfo(np.float32).max）
                    if idx < 0 or dist > 1e+35:
                        print(f"Skipping candidate with dist {dist} and idx {idx}")
                        continue
                    rep_idx = index_mapping[idx]
                    if rep_idx < len(report_data):
                        cand_id = str(true_id_map[rep_idx])
                        cand_text = report_data[rep_idx]["report"]
                        if cand_id in candidate_dict:
                            candidate_dict[cand_id]['score'] += dist
                            candidate_dict[cand_id]['reports'].append(cand_text)
                        else:
                            candidate_dict[cand_id] = {'score': dist, 'reports': [cand_text]}
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # 如果没有有效候选，则回退使用真实报告
        if not candidate_dict:
            print(f"No valid candidates found for report id: {true_relevant_id}. Using fallback (true report).")
            # 将真实报告也加入candidate_dict，以便后续Recall命中
            candidate_dict[true_relevant_id] = {
                'score': 0.0,
                'reports': [true_report]
            }

        # 对candidate_dict进行排序
        sorted_candidates = sorted(candidate_dict.items(), key=lambda x: x[1]['score'])
        aggregated_ids = [cand_id for cand_id, info in sorted_candidates]

        best_candidate_id = aggregated_ids[0]
        candidate_texts = candidate_dict[best_candidate_id]['reports']
        reranked_candidates = semantic_rerank(true_report, candidate_texts)
        final_candidate = reranked_candidates[0]
        
        if not final_candidate:
            final_candidate = true_report

        print(f"Report ID: {true_relevant_id}, total valid candidates aggregated: {len(candidate_dict)}")
        
        # 计算 BLEU 指标
        ref_tokens = [nltk.word_tokenize(true_report)]
        cand_tokens = nltk.word_tokenize(final_candidate)
        bleu_1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1,0,0,0), smoothing_function=smooth_fn)
        bleu_2 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.5,0.5,0,0))
        bleu_3 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.33,0.33,0.33,0))
        bleu_4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25,0.25,0.25,0.25))
        bleu_scores["BLEU-1"].append(bleu_1)
        bleu_scores["BLEU-2"].append(bleu_2)
        bleu_scores["BLEU-3"].append(bleu_3)
        bleu_scores["BLEU-4"].append(bleu_4)
        
        # 计算 ROUGE 指标
        scores = rouge.get_scores(final_candidate, true_report)
        rouge_scores["ROUGE-1"].append(scores[0]['rouge-1']['f'])
        rouge_scores["ROUGE-2"].append(scores[0]['rouge-2']['f'])
        rouge_scores["ROUGE-L"].append(scores[0]['rouge-l']['f'])
        
        # 计算 METEOR 指标
        ref_tokens_meteor = [word_tokenize(true_report)]
        cand_tokens_meteor = word_tokenize(final_candidate)
        meteor_scores.append(meteor_score(ref_tokens_meteor, cand_tokens_meteor))
        
        # 计算 Recall：根据候选ID排序后的前 5/10
        recall_5 = 1.0 if true_relevant_id in aggregated_ids[:5] else 0.0
        recall_10 = 1.0 if true_relevant_id in aggregated_ids[:10] else 0.0
        recall_5_scores.append(recall_5)
        recall_10_scores.append(recall_10)

    avg_metrics = {
        "BLEU-1": np.mean(bleu_scores["BLEU-1"]) if bleu_scores["BLEU-1"] else 0.0,
        "BLEU-2": np.mean(bleu_scores["BLEU-2"]) if bleu_scores["BLEU-2"] else 0.0,
        "BLEU-3": np.mean(bleu_scores["BLEU-3"]) if bleu_scores["BLEU-3"] else 0.0,
        "BLEU-4": np.mean(bleu_scores["BLEU-4"]) if bleu_scores["BLEU-4"] else 0.0,
        "ROUGE-1": np.mean(rouge_scores["ROUGE-1"]) if rouge_scores["ROUGE-1"] else 0.0,
        "ROUGE-2": np.mean(rouge_scores["ROUGE-2"]) if rouge_scores["ROUGE-2"] else 0.0,
        "ROUGE-L": np.mean(rouge_scores["ROUGE-L"]) if rouge_scores["ROUGE-L"] else 0.0,
        "METEOR": np.mean(meteor_scores) if meteor_scores else 0.0,
        "Recall@5": np.mean(recall_5_scores) if recall_5_scores else 0.0,
        "Recall@10": np.mean(recall_10_scores) if recall_10_scores else 0.0
    }
    return avg_metrics

if __name__ == "__main__":
    test_data = load_test_set(Config.ANNOTATION_PATH, split="test")
    avg_metrics = evaluate_system(
        test_data,
        index_path=Config.NEW_INDEX_PATH,
        mapping_path=Config.NEW_MAPPING_PATH,
        json_file_path=Config.ANNOTATION_PATH,
        top_k=50
    )
    
    print("Evaluation Results:")
    for metric, value in avg_metrics.items():
        if "Recall" in metric:
            print(f"{metric}: {value*100:.2f}%")
        else:
            print(f"{metric}: {value:.4f}")
