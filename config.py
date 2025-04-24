# ADDED: config.py - 集中管理配置参数
import os

class Config:
    BASE_DIR = "/home/lijiaji/RAG"
    IMAGE_DIR = os.path.join(BASE_DIR, "data/iu_xray/images")
    MODEL_DIR = os.path.join(BASE_DIR, "clip-vit-base-patch32")
    
    TEXT_MAX_LENGTH = 77       # CLIP文本编码器最大长度
    IMAGE_SIZE = (224, 224)    # 统一图像尺寸

    # GPU 配置
    GPU_IDS = [0, 1]
    BATCH_SIZE = 32 # 根据4090显存调整 #每个样本有两张图，有效batch_size为64
    EPOCHS = 10
    NUM_WORKERS = 8
    USE_MMAP = True # 是否使用内存映射(使用：减少内存压力，加速索引；不使用：减少磁盘IO)

    # 索引路径
    INDEX_PATH = os.path.join(BASE_DIR, "faiss_ivf_index.bin")
    MAPPING_PATH = os.path.join(BASE_DIR, "index_mapping.npy")
    
    # 数据路径
    ANNOTATION_PATH = os.path.join(BASE_DIR, "data/iu_xray/annotation.json")
    IMAGE_BASE_DIR = os.path.join(BASE_DIR,"data/iu_xray/images")  # 图像基础路径

    FP16 = True  # 是否使用混合精度训练

    VALIDATE_BEFORE_TRAIN = True

    TEXT_MODEL_PATH = "/home/lijiaji/RAG/all-MiniLM-L6-v2"  # 文本模型路径
    IMAGE_CACHE_DIR = "./image_cache"  # 图像缓存目录
    EMBEDDING_CHUNK = 512  # 嵌入分块大小

    GRADIENT_ACCUMULATION_STEPS = 1     # 梯度累积步数，单卡关闭梯度积累
    MAX_GRAD_NORM = 1.0                 # 梯度裁剪
    WARMUP_STEPS = 1000                 # 学习率预热
    LOG_INTERVAL = 10                   # 日志间隔

    TEMPERATURE = 0.1  # 对比损失温度系数
    MAX_VIEWS = 2       # 每个病例最大视图数
    MIN_VIEWS = 1       # 最少需要视图数
    
    # 学习率调度
    MAX_LR = 5e-4       # OneCycle最大学习率
    WARMUP_PCT = 0.1    # 预热比例

    MODEL_SAVE_DIR = os.path.join(BASE_DIR,"contrastiveGen") # 对比模型保存路径
    NEW_INDEX_PATH = os.path.join(BASE_DIR,"contrastiveGen/new_faiss_index.bin")
    NEW_MAPPING_PATH = os.path.join(BASE_DIR,"contrastiveGen/new_index_mapping.npy")

    NEW_MAPPING_PATH_true_ids = os.path.join(BASE_DIR,"contrastiveGen/new_index_mapping_true_ids.npy")