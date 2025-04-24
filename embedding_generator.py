#`embedding_generator.py` 负责生成文本和图像嵌入
from sentence_transformers import models,SentenceTransformer
import numpy as np
from report_extractor import load_json_data
import pickle
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from config import Config


# 加载 SentenceTransformer 模型
model_path = "/home/lijiaji/RAG/all-MiniLM-L12-v2"
#text_model = SentenceTransformer(model_path)

# 手动加载 Transformer 模型
word_embedding_model = models.Transformer(model_path)

# 在 embedding_generator.py 中添加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义池化层
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# 构建 Sentence-Transformers 模型
text_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


class MedicalImageDataset(Dataset):
    """医疗图像异步加载数据集"""
    def __init__(self, data, base_dir="/home/lijiaji/RAG/data/iu_xray/images"):
        # 禁用文件锁避免冲突
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['GOTO_NUM_THREADS'] = '1'

        self.data = data
        self.base_dir = base_dir
        
    def __len__(self):
        return sum(len(report["image_path"]) for report in self.data)
    
    def __getitem__(self, idx):
        
        # 将线性索引转换为报告索引和图像索引
        report_idx = 0
        while idx >= len(self.data[report_idx]["image_path"]):
            idx -= len(self.data[report_idx]["image_path"])
            report_idx += 1
        img_path = self.data[report_idx]["image_path"][idx]
        
        try:
            full_path = os.path.join(self.base_dir, img_path)
            image = Image.open(full_path).convert("RGB")
            return image  # 返回原始 PIL 图像，后续统一预处理
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None  # 返回 None 后续过滤

class ImageModelLoader:
    def __init__(self):
        self.device = torch.device(f"cuda:{Config.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_SAVE_DIR)  # 修改为微调后的模型路径
        self.model = CLIPModel.from_pretrained(Config.MODEL_SAVE_DIR, projection_head=True).to(self.device)
        # 加载对比学习权重
        if Config.CONTRASTIVE_WEIGHTS: 
            self.model.load_state_dict(torch.load(Config.CONTRASTIVE_WEIGHTS))

        # 冻结非必要层
        for name, param in self.model.named_parameters():
            if "visual_projection" not in name:  # 只训练投影头
                param.requires_grad_(False)

# 修改 embedding_generator.py
def create_paragraph_embeddings(report_data):
    """生成段落级嵌入"""
    all_embeddings = []
    index_mapping = []
    
    for report_idx, report in enumerate(report_data):
        parsed = load_json_data(report['report'])
        
        # 为每个段落生成嵌入
        for section in ['findings', 'impression']:
            if parsed[section]:
                embedding = text_model.encode(parsed[section])
                all_embeddings.append(embedding)
                index_mapping.append({
                    'report_idx': report_idx,
                    'section': section
                })
                
    return np.array(all_embeddings), index_mapping


# embedding_generator.py 添加有效 collate_fn
def collate_fn(batch):
    images = [img for img in batch if img is not None]
    return images # 简单返回有效图像列表


# 创建嵌入向量
def create_embeddings(data, model, batch_size=128):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained(Config.MODEL_SAVE_DIR)
    model = CLIPModel.from_pretrained(Config.MODEL_SAVE_DIR)
    model = torch.nn.DataParallel(model)  # 添加并行
    model = model.to(device)
    model.eval()

    embeddings = []
    
    for i in range(0, len(data), batch_size):
        batch_reports = [entry.get("report", "") for entry in data[i:i + batch_size]]  ## 使用 .get 防止 KeyError
        texts = batch_reports  # 定义 texts 变量
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        
        # 生成批次嵌入
        batch_embeddings = model.module.get_image_features(**inputs)
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)

        if (i // batch_size) % 100 == 0 or i == len(data) - batch_size:  # 每 100 个批次打印一次
            print(f"Processing batch {i // batch_size + 1}, size: {len(batch_reports)}")
            print(f"Batch embeddings shape: {batch_embeddings.shape}")
            
        embeddings.extend(batch_embeddings)
    return np.array(embeddings).astype('float32')


# 创建图片嵌入向量
# embedding_generator.py
def create_image_embeddings(data, batch_size=512):
    """为每个报告的两张图像生成嵌入，返回所有图像的嵌入列表"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained(Config.MODEL_DIR)
    model = CLIPModel.from_pretrained(Config.MODEL_DIR)
    model = torch.nn.DataParallel(model)  # 添加并行
    model = model.to(device)
    model.eval()

    embeddings = []

    for i in range(0, len(data), batch_size):
        batch_reports = data[i:i + batch_size]
        batch_images = []
        for report in batch_reports:
            # 处理每个报告的两张图像
            for path in report.get("image_path", []):
                try:
                    image_path = os.path.join(Config.IMAGE_BASE_DIR, path)
                    img = Image.open(image_path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error opening image {path}: {e}")
                    continue

        if not batch_images:
            print(f"跳过空批次: {i}-{i+batch_size}")
            continue  # 跳过没有图像的批次


                    # 处理输入
        inputs = processor(
            images=batch_images,
            return_tensors="pt",
            do_normalize=True
        ).to(device)


        # 生成嵌入
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            batch_embeddings = model.module.get_image_features(**inputs).cpu().numpy()
        
        embeddings.extend(batch_embeddings)
        #embeddings = F.normalize(embeddings, p=2, dim=-1)  # 添加L2归一化

    return np.array(embeddings).astype('float32')

# 从报告提取数据并生成嵌入
def generate_embeddings(report_data, model_type="image", max_records=None):
    """
    从报告提取数据并生成图像或文本嵌入
    """
    # 加载 JSON 数据
    if max_records:
        report_data = report_data[:max_records]  # 只加载前 max_records 条数据
    print(f"Total records loaded: {len(report_data)}")

    # 如果 model_type 为 'image'，生成图像嵌入
    if model_type == "image":
        embeddings = create_image_embeddings(report_data)
    else:
        embeddings = create_embeddings(report_data, text_model)

    print(f"Generated embeddings: {embeddings.shape}")  # 打印嵌入向量形状

    # 保存嵌入到磁盘
    save_embeddings(embeddings, "/home/lijiaji/RAG/embeddings.pkl")
    return embeddings


def save_embeddings(embeddings, file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        print(f"Loading embeddings from: {file_path}")
        return pickle.load(f)
