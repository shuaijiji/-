# query_faiss.py 处理文本查询
import faiss
import numpy as np
from embedding_generator import text_model  # Ensure text_model has an 'encode' method
import json
from config import Config

def load_report_data(report_data_path="/home/lijiaji/RAG/report_data.npy"):
    try:
        report_data = np.load(report_data_path, allow_pickle=True)
        if isinstance(report_data, np.ndarray):
            report_data = report_data.tolist()  # 转换为列表
        return report_data
    except Exception as e:
        print(f"Error loading report data: {e}")
        return []


def query_faiss(query_text, top_k=1, index_path=Config.NEW_INDEX_PATH, report_data_path="RAG/report_data.npy"):
    try:
        # Load the FAISS index
        index = faiss.read_index(index_path)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return []
    
    try:
        # Load the report data
        report_data = load_report_data(report_data_path)
    except Exception as e:
        print(f"Error loading report data: {e}")
        return []
    
    try:
        # Generate query embedding
        query_embedding = text_model.encode([query_text], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []
    
    try:
        # Perform the search
        distances, indices = index.search(query_embedding, top_k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []
    
    # Prepare the results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(report_data):
            results.append({
                "ImagePath": report_data[idx]['ImagePath'],
                "Report": report_data[idx]['Report'],
                "Distance": dist
            })
    
    return results

if __name__ == "__main__":
    report_data_path = "/home/lijiaji/RAG/report_data.npy"
    report_data = load_report_data(report_data_path)
    print(f"Loaded {len(report_data)} reports.")


# Example usage:
# results = query_faiss("Bilateral nodular opacities suggest nipple shadows.", top_k=2)
# print(results)
