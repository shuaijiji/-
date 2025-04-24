import faiss
import numpy as np
from embedding_generator import generate_embeddings,text_model  # 导入嵌入生成模块
from report_extractor import load_json_data

# 创建 FAISS 索引
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # 使用 L2 距离进行相似度计算
    index.add(embeddings)
    return index

# 查询 FAISS 索引
def query_faiss(query_text, top_k=1, json_file_path="RAG/dataset/MIMIC-annotation.json"):
    embeddings = generate_embeddings(json_file_path)  # 获取报告的嵌入向量
    query_embedding = text_model.encode([query_text], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')

    # 创建 FAISS 索引
    faiss_index = build_faiss_index(embeddings)
    # 保存索引到磁盘
    save_faiss_index(faiss_index, "RAG/faiss_index.bin")  

    # 在 FAISS 索引中进行检索
    distances, indices = faiss_index.search(query_embedding, top_k)

    # 返回最相似的报告和对应的图像路径
    results = []
    report_data = load_json_data(json_file_path)  # 重新加载报告数据
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(report_data):
            results.append({
                "ImagePath": report_data[idx]['ImagePath'],
                "Report": report_data[idx]['Report'],
                "Distance": dist
            })

    return results

# 保存 FAISS 索引
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# 加载 FAISS 索引
def load_faiss_index(file_path):
    return faiss.read_index(file_path)


# 示例：
# query_text = "Bilateral nodular opacities suggest nipple shadows."
# results = query_faiss(query_text, top_k=2, json_file_path="path/to/your/report_data.json")
