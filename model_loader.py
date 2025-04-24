# model_loader.py
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

class TextModelLoader:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

class ImageModelLoader:
    def __init__(self, model_path):
        """
        初始化 CLIP 模型和处理器。
        :param model_path: CLIP 模型的路径。
        """
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path)

    def generate_image_embedding(self, image_path):
        """
        生成图像的嵌入向量。
        :param image_path: 图像文件的路径。
        :return: 图像的嵌入向量(numpy 数组）。
        """
        # 打开图像并转换为 RGB 格式
        img = Image.open(image_path).convert("RGB")
        
        # 使用 CLIP 处理器预处理图像
        inputs = self.processor(images=img, return_tensors="pt", do_normalize=True)

        # 使用 CLIP 模型生成图像特征
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # 将特征转换为 numpy 数组并返回
        return image_features.cpu().numpy()