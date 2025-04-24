# image_embedding_query.py 用于查询单张图像的最相似报告。
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from report_extractor import load_json_data
import torch.nn.functional as F
from model_loader import ImageModelLoader
from tqdm import tqdm  # 进度条工具
from config import Config

def batch_query(test_images, ground_truth_reports, index_path, json_file_path, k=5):
    """
    批量查询测试集中的所有图像，并收集结果。
    :param test_images: 测试图像的路径列表。
    :param ground_truth_reports: 每个图像的真实相关报告列表。
    :param index_path: FAISS 索引路径。
    :param json_file_path: 报告数据的 JSON 文件路径。
    :param k: 检索返回的 Top-K 结果数量。
    :return: 包含所有查询结果的列表。
    """
    # 加载 FAISS 索引和映射
    index = load_index(index_path)
    index_mapping = np.load(Config.NEW_MAPPING_PATH)
    data = load_json_data(json_file_path, split="train")
    
    all_results = []
    for image_path, true_reports in tqdm(zip(test_images, ground_truth_reports), desc="Processing images"):
        # 生成查询图像的嵌入
        image_embedding = generate_image_embedding(image_path)
        
        # 执行 FAISS 查询
        D, I = find_most_similar_report(image_embedding, index, k)
        
        # 提取检索到的报告内容
        retrieved_reports = []
        for idx in I[0]:
            report_idx = index_mapping[idx]
            if report_idx < len(data):
                retrieved_reports.append(data[report_idx]["report"])
            else:
                retrieved_reports.append("Invalid index")
        
        # 保存结果
        all_results.append({
            "image_path": image_path,
            "true_reports": true_reports,
            "retrieved_reports": retrieved_reports,
            "distances": D[0].tolist()
        })
    
    return all_results

# 加载 FAISS 索引
# 修改 image_embedding_query.py 的 load_index
def load_index(index_path):
    cpu_index = faiss.read_index(index_path)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 加载到 GPU 0
    return gpu_index

# 加载PCA和FAISS索引
'''
def load_pca_and_index(pca_path, index_path):
    pca = faiss.read_VectorTransform(pca_path)  # 加载PCA变换矩阵
    index = faiss.read_index(index_path)  # 加载FAISS索引
    return pca, index
'''

# 使用 CLIP 模型生成图像嵌入
def generate_image_embedding(image_path, model_type="image"):
    """
    生成图像的嵌入向量。
    :param image_path: 图像文件的路径。
    :param model_type: 模型类型（默认为 "image"）。
    :return: 图像的嵌入向量（numpy 数组）。
    """
    # 初始化 ImageModelLoader
    model_path = Config.MODEL_SAVE_DIR 
    image_model_loader = ImageModelLoader(model_path)
    
    # 生成图像嵌入
    image_embedding = image_model_loader.generate_image_embedding(image_path)
    # 确保嵌入是 PyTorch Tensor（如果模型返回numpy则需转换）
    if isinstance(image_embedding, np.ndarray):
        image_embedding = torch.from_numpy(image_embedding)
    
    # 使用 PyTorch 进行归一化
    #eps = 1e-8
    image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
    
    # 转换为 numpy 并压缩维度
    return image_embedding.cpu().numpy().squeeze()


# 使用FAISS进行查询
def find_most_similar_report(image_embedding, index, k):
    # 确保查询向量为二维
    if len(image_embedding.shape) == 1:
        image_embedding = image_embedding.reshape(1, -1)
    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Index dimension: {index.d}")

    # 确保查询向量维度匹配索引维度
    assert image_embedding.shape[1] == index.d, "Query vector dimension does not match index dimension."
    
    # 执行最近邻搜索
    D, I = index.search(image_embedding, k)
    return D, I

# 修改 image_embedding_query.py 的查询部分
def parallel_query(image_paths, k=5):
    """多GPU并行查询"""
    from functools import partial
    import concurrent.futures
    
    # MODIFIED: 任务拆分
    def _query_single(img_path, gpu_id, index):
        with torch.cuda.device(gpu_id):
            embedding = generate_image_embedding(img_path)
            D, I = index.search(embedding, k)
            return D, I
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, img_path in enumerate(image_paths):
            gpu_id = Config.GPU_IDS[i % len(Config.GPU_IDS)]  # 轮询分配GPU
            futures.append(executor.submit(partial(_query_single, img_path, gpu_id, index)))
            
        return [f.result() for f in futures]


# 主函数：输入查询图像并返回最相似的报告
def query_image(image_path, index_path=Config.NEW_INDEX_PATH, mapping_path=Config.NEW_MAPPING_PATH, json_file_path=Config.ANNOTATION_PATH):

    # 加载 FAISS 索引和映射
    index = load_index(index_path)
    index_mapping = np.load(mapping_path)
    print(f"Index mapping loaded with {len(index_mapping)} entries.")

    # 生成查询图像的嵌入
    image_embedding = generate_image_embedding(image_path)

    # 查找最相似的报告
    k = 5  # 返回前5个最相似的报告
    D, most_similar_embeddings = find_most_similar_report(image_embedding, index, k)

    # 加载报告数据
    data = load_json_data(json_file_path, split="train")

    # 输出最相似报告的内容
    similar_report_texts = []
    for embedding_idx, dist in zip(most_similar_embeddings[0], D[0]):
        report_idx = index_mapping[embedding_idx]  # 使用映射找到对应的报告索引
        if report_idx < len(data):  # 确保索引在范围内
            similar_report_texts.append(f"index: {report_idx}    dist: {dist:.4f}    report: {data[report_idx]['report']}")
        else:
            print(f"Index {report_idx} is out of range. Skipping.")

    return similar_report_texts

# image_embedding_query.py 添加多模态融合##################################
def hybrid_retrieval(image_embed, text_query=None, alpha=0.7):
    """
    混合图像和文本查询
    :param alpha: 图像权重 (0.7) vs 文本权重 (0.3)
    """
    image_sim = index.search(image_embed, k=100)  # 初筛
    
    if text_query:
        text_embed = text_model.encode([text_query])
        text_sim = index.search(text_embed, k=100)
        
        # 加权融合
        combined_scores = alpha * image_sim + (1-alpha) * text_sim
        return rerank_topk(combined_scores, k=5)
    
    return image_sim


# 示例：查询一个图像并输出最相似的报告
if __name__ == "__main__":
    image_path = "/home/lijiaji/mimic_cxr/images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"  # 输入图片路径
    index_path = "/home/lijiaji/RAG/faiss_ivf_index.bin"
    json_file_path = "/home/lijiaji/RAG/data/iu_xray/annotation.json"

    similar_reports = query_image(image_path, index_path=index_path, json_file_path=json_file_path)

    print("Most similar reports:")
    for report in similar_reports:
        print(report)  # 输出最相似的报告内容
