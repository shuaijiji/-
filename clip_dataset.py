# ADDED: clip_dataset.py
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from PIL import Image
from config import Config
import os
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
from torchvision.transforms import functional as F, transforms
from transformers import CLIPTokenizer
from torchvision import transforms  
from collections import Counter
from PIL import Image
import re
import json

class RandomROI:
    def __call__(self, img):
        w, h = img.size
        # 随机裁剪中心区域
        crop_size = min(w,h) // 2
        i = (h - crop_size) // 2
        j = (w - crop_size) // 2
        return F.crop(img, i, j, crop_size, crop_size)
    
class CLIPDataset(Dataset):
# 在CLIPDataset的__init__方法中调整
    def __init__(self, split='train'):
        self.mode = split  # 添加mode属性
        
        # 加载完整数据
        with open(Config.ANNOTATION_PATH) as f:
            full_data = json.load(f)  # 此时full_data是字典类型
        
        # 获取指定split的数据
        raw_samples = full_data.get(str(split), [])

        # 数据校验和处理
        self.samples = []
        for sample in raw_samples:
            self.samples.append({
                'id': sample['id'],
                'report': sample['report'],
                'image_paths': [os.path.join(Config.IMAGE_BASE_DIR, p) for p in sample['image_path']]
            })

        # 定义数据增强
        # base_transform基础变换，包括调整大小，归一化等
        self.base_transform = transforms.Compose([
            transforms.Resize(256),  # 先放大后随机裁剪
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.498]*3, std=[0.249]*3)  # 医疗灰度图参数
        ])
        # 训练时的数据增强, 包括随机变换、高斯模糊、随机擦除等
        self.train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1,0.1)), # 增加空间变换
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 关键：将PIL图像转为[C, H, W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        '''
            RandomROI(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            self.base_transform
            '''

        # 单处理器实例
        self.processor = CLIPProcessor.from_pretrained(
            Config.MODEL_DIR,
            max_length=77,   #文本序列的最大长度。如果文本长度超过这个值，会被截断；如果不足，则会被填充。
            padding="max_length",
            truncation=True
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(Config.MODEL_DIR)

        # 文本缓存优化
        self.text_cache = {}
        self._warmup_cache()

        print(f"成功加载 {len(self.samples)} 个样本")
        '''
        for sample in self.samples[:3]:  # 打印前3个样本的路径
            print(f"样本示例路径: {sample['image_paths'][0]}")'''

    def __len__(self):
        return len(self.samples)  # 返回数据集样本数量
    
    def _warmup_cache(self):      # 预处理高频出现的报告文本，缓存其处理结果以加速后续操作。
        """预热高频文本的缓存"""
        report_counter = Counter(s['report'] for s in self.samples)
        top_reports = [r for r, _ in report_counter.most_common(100)]
        for report in tqdm(top_reports, desc="预热文本缓存"):
            self._process_text(report)

    def _process_text(self, text):  #对文本进行分词和编码，利用缓存避免重复计算。
        """带缓存的文本处理"""
        text_hash = hash(text)
        if text_hash not in self.text_cache:
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",   # 返回PyTorch张量
                max_length=77,
                padding="max_length",
                truncation=True
            )
            self.text_cache[text_hash] = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            }
        return self.text_cache[text_hash]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = []
        for img_path in sample['image_paths']:  # 使用正确的键名
            try:
                with Image.open(img_path) as img:
                    #img = Image.open(img_path).convert("L")  # 灰度模式
                    img = img.convert("RGB")
                    if self.mode == 'train':
                        img = self.train_transform(img)
                    else:
                        img = self.base_transform(img)
                    images.append(img)
            except:
                images.append(torch.zeros(3, 224, 224))


        # 堆叠所有图像 [num_images, C, H, W]
        pixel_values = torch.stack(images, dim=0)# if images else torch.zeros(2, 3, 224, 224)
        #pixel_values = torch.stack(images, dim=0) #images[0] if images else torch.zeros(3, 224, 224)
        
        clean_report = re.sub(r'[\n\t]+', ' ', sample['report']).strip()

        text_inputs = self.tokenizer(
            clean_report,
            return_tensors="pt",
            max_length=77,
            padding="max_length",
            truncation=True
        )

        input_ids = text_inputs['input_ids'].squeeze(0)
        # 调试输出
        if idx == 0:
            print(f"Debug样本0的input_ids形状: {input_ids.shape}")
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'study_id': sample['id']
        }
    

class ResizeDynamic:
    """动态调整分辨率保持宽高比"""
    def __init__(self, max_size):
        self.max_size = max_size
        
    def __call__(self, img):
        w, h = img.size
        ratio = min(self.max_size/w, self.max_size/h)
        return img.resize((int(w*ratio), int(h*ratio)), Image.BICUBIC)

