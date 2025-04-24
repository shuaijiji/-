#index_builder.py 构建FAISS索引。
import faiss
import numpy as np
from embedding_generator import generate_embeddings
from report_extractor import load_json_data
import pickle
from config import Config
from config_loader import hw_config
import torch

# IndexFlatL2 是一种强力索引，它将所有向量存储在内存中，并且可以在它们之间计算 L2 距离


def build_faiss_ivf_index(embeddings, nlist=100, index_path=Config.NEW_INDEX_PATH):
    """适配对比学习的索引构建"""
    d = embeddings.shape[1]

    #res = faiss.StandardGpuResources()

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # 使用余弦相似度（需要内积索引）
    #index = faiss.IndexFlatIP(d)  # Inner Product
    
    
    # 保存为兼容格式
    faiss.write_index(index, index_path)
    print(f"CPU 索引已保存至 {index_path}")


# 为后续处理准备索引（加载数据并生成嵌入）
# index_builder.py
def prepare_index(json_file_path, index_path, mapping_path="Config.NEW_MAPPING_PATH"):
    report_data = load_json_data(json_file_path, split="train")
    embeddings = generate_embeddings(report_data, model_type="image")
    
def prepare_index(json_file_path, index_path, mapping_path="Config.NEW_MAPPING_PATH"):
    report_data = load_json_data(json_file_path, split="train")
    embeddings = generate_embeddings(report_data, model_type="image")
    
    # 添加归一化：使每个嵌入归一化到单位长度
    faiss.normalize_L2(embeddings)

    # 创建映射：每个嵌入对应原始报告的索引
    index_mapping = []
    true_ids = []  # 新增：保存真实报告 ID 映射（均转换为字符串）
    for report_idx, report in enumerate(report_data):
        num_images = len(report["image_path"])
        index_mapping.extend([report_idx] * num_images)
        true_ids.extend([str(report["id"])] * num_images)
    
    # 保存映射和构建索引
    np.save(mapping_path, np.array(index_mapping))
    # 保存真实ID映射
    np.save(mapping_path.replace(".npy", "_true_ids.npy"), np.array(true_ids))

    build_faiss_ivf_index(embeddings, nlist=100, index_path=index_path)
    print(f"Index built with {len(embeddings)} embeddings (2 per report).")


if __name__ == "__main__":
    json_file_path = Config.ANNOTATION_PATH
    index_path = Config.NEW_INDEX_PATH
    mapping_path = Config.NEW_MAPPING_PATH

    # 加载 JSON 数据
    report_data = load_json_data(json_file_path, split="train")
    #print(f"Loaded {len(report_data)} records. Example records: {report_data[:2]}")

    # 准备索引
    prepare_index(json_file_path, index_path, mapping_path)
    
    # 文本处理
    print("Processing text embeddings...")
    text_embeddings = generate_embeddings(report_data).astype('float32')
    
    # 使用原始嵌入构建 FAISS 索引
    build_faiss_ivf_index(text_embeddings, nlist=100, index_path=index_path)
    print("Text FAISS index built.")

    # 图像处理
    print("Processing image embeddings...")
    image_embeddings = generate_embeddings(report_data, model_type="image").astype('float32')  # 加载图像嵌入

    # 使用原始图像嵌入构建 FAISS 索引
    build_faiss_ivf_index(image_embeddings, nlist=100, index_path=index_path)
    print("Image FAISS index built.")
    
