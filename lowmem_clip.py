# lowmem_clip.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPConfig
from config import Config

#########################
# 自定义视觉嵌入层定义
#########################
class CustomVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size  # 例如 16
        self.image_size = config.image_size  # 例如 224
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # 加上CLS token

        # 添加 class_embedding 参数，形状与 hidden_size 相同
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        # 定义patch嵌入层
        self.patch_embedding = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )
        # 位置编码层
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
        self._init_weights()

    def _init_weights(self):
        # 对新增位置编码部分进行初始化
        if self.position_embedding.weight.shape[0] > 50:
            nn.init.normal_(self.position_embedding.weight[50:], mean=0.0, std=0.02)
        # 初始化卷积核
        nn.init.kaiming_normal_(self.patch_embedding.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, pixel_values, interpolate_pos_encoding=False, **kwargs):
        batch_size = pixel_values.shape[0]
        # 生成patch嵌入
        patches = self.patch_embedding(pixel_values)  # [B, hidden, H', W']
        patches = patches.flatten(2).transpose(1, 2)    # [B, num_patches, hidden]
        # 获取对应位置编码（不包含CLS token）
        patch_pos_embed = self.position_embedding(self.position_ids[:, 1:1+patches.size(1)])
        embeddings = patches + patch_pos_embed
        # 获取CLS token的编码，并在最前面拼接
        cls_pos_embed = self.position_embedding(self.position_ids[:, :1])
        embeddings = torch.cat([cls_pos_embed.expand(batch_size, -1, -1), embeddings], dim=1)
        # 可选：对CLS token再加一次位置编码
        embeddings[:, 0:1] += cls_pos_embed
        return embeddings

#########################
# 配置适配（医疗专用配置）
#########################
class MedicalCLIPConfig:
    @staticmethod
    def get_config(pretrained_path):
        base_config = CLIPConfig.from_pretrained(pretrained_path)
        # 根据需要修改视觉部分参数
        vision_config = base_config.vision_config.to_dict()
        vision_config.update({
            "patch_size": 16,
            "image_size": 224,
            "num_hidden_layers": 6
        })
        return CLIPConfig.from_dict({
            "vision_config": vision_config,
            "text_config": base_config.text_config.to_dict()
        })

#########################
# 模型定义
#########################
class LowMemCLIP(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        self.register_for_auto_class()
        # 从配置中获取关键信息
        self.patch_size = config.vision_config.patch_size
        self.image_size = config.vision_config.image_size
        self.hidden_size = config.vision_config.hidden_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # 定义基础patch嵌入和位置编码（用于适配预训练权重时使用）
        self.patch_embedding = nn.Conv2d(
            3, config.vision_config.hidden_size,
            kernel_size=16, stride=16, bias=False
        )
        self.position_embedding = nn.Embedding(
            self.num_positions, config.vision_config.hidden_size
        )
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
        # 保留父类投影层属性（后面会替换）
        self.vision_projection = self.visual_projection
        self.text_projection = self.text_projection

        # 使用自定义视觉嵌入层
        self.vision_model.embeddings = CustomVisionEmbeddings(config.vision_config)

        # 替换投影层（将视觉和文本特征映射到相同维度）
        self.vision_projection = nn.Linear(config.vision_config.hidden_size, 512)
        self.text_projection = nn.Linear(config.text_config.hidden_size, 512)

        # 初始化新增层
        self._init_custom_layers()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))  # 初始化到合理范围

        # 断言验证关键参数
        assert config.vision_config.patch_size == 16, "视觉配置patch_size未正确设置!"
        assert config.vision_config.image_size == 224, "视觉配置image_size错误!"
        assert self.vision_projection.out_features == 512, "视觉投影层维度错误"
        assert self.text_projection.out_features == 512, "文本投影层维度错误"

    def _init_custom_layers(self):
        # 初始化视觉投影层
        vision_hidden_size = self.config.vision_config.hidden_size
        self.vision_projection = nn.Linear(vision_hidden_size, 512)
        # 定义多视图融合注意力层
        self.view_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Flatten()
        )
        # 定义文本适配器
        self.text_adapter = nn.Linear(self.config.text_config.hidden_size, 512)
        # 如果父类未定义文本投影层则初始化之
        if self.text_projection is None:
            text_hidden_size = self.config.text_config.hidden_size
            self.text_projection = nn.Linear(text_hidden_size, 512)

        # 初始化各层权重
        nn.init.kaiming_normal_(self.vision_model.embeddings.patch_embedding.weight,
                                mode='fan_out', nonlinearity='relu')
        
        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.xavier_uniform_(self.text_projection.weight)
        if self.vision_projection.bias is not None:
            nn.init.constant_(self.vision_projection.bias, 0.0)
        if self.text_projection.bias is not None:
            nn.init.constant_(self.text_projection.bias, 0.0)

        self.view_attention = nn.Sequential(
        nn.Linear(512, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
        nn.Flatten()
        )
        self.text_adapter = nn.Linear(self.config.text_config.hidden_size, 512)

        # 冻结文本模型参数（只微调文本投影层）
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_projection.requires_grad = True

        print("视觉投影层权重均值：", self.vision_projection.weight.data.mean().item())
        print("文本投影层权重均值：", self.text_projection.weight.data.mean().item())

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        # 判断输入的pixel_values维度，处理多视图情况
        if pixel_values.dim() == 5:
            batch_size, num_views = pixel_values.shape[:2]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        else:
            batch_size = pixel_values.shape[0]
            num_views = 1

        # 生成patch嵌入和位置编码（这里仅用于辅助参考，可根据需要定制）
        patches = self.patch_embedding(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)
        position_embeddings = self.position_embedding(self.position_ids[:, :patches.size(1)+1])
        embeddings = patches + position_embeddings[:, 1:, :]

        # 视觉分支前向传播（这里调用了父类的视觉模型）
        vision_outputs = self.vision_model(pixel_values)
        image_features = vision_outputs.last_hidden_state[:, 0, :]
        image_features = self.vision_projection(image_features)
        image_features = image_features.view(batch_size, num_views, -1)

        # 计算多视图融合：计算注意力权重后融合所有视图的特征
        attn_scores = self.view_attention(image_features)
        attn_weights = F.softmax(attn_scores, dim=1)
        fused_features = (image_features * attn_weights.unsqueeze(-1)).sum(dim=1)

        # 文本分支前向传播
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # 先经过文本适配器，再投影到同一维度
        text_features = self.text_adapter(text_features)
        text_features = self.text_projection(text_features)

        # 归一化特征
        image_features = fused_features / (fused_features.norm(dim=1, keepdim=True) + 1e-6)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)

        # 限制 logit_scale 的最大值，防止标度过大
        clamped_scale = self.logit_scale.clamp(max=4.6052)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)  # 原无约束→max=100
        logit_scale = clamped_scale.exp()

        # 计算相似度对数
        logit_scale = self.logit_scale.exp()
        logit_scale = min(logit_scale.item(), 100.0)
        logits_per_image = logit_scale * image_features @ text_features.t()
        #logits_per_image = (image_features @ text_features.t()) / Config.TEMPERATURE
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 强制忽略尺寸不匹配
        kwargs['ignore_mismatched_sizes'] = True
        # 获取定制的医疗配置
        config = MedicalCLIPConfig.get_config(pretrained_model_name_or_path)
        # 初始化模型
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            config=config,
            **kwargs
        )
        # 手动加载权重
        state_dict = torch.load(
            os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        if 'vision_model.embeddings.class_embedding' in state_dict:
            model.vision_model.embeddings.class_embedding.data = state_dict['vision_model.embeddings.class_embedding'].clone()
            print("成功加载class_embedding权重")
        else:
            print("警告：未找到class_embedding权重，使用随机初始化")
        # 裁剪卷积核：从预训练权重中提取中心16x16区域
        original_conv = state_dict['vision_model.embeddings.patch_embedding.weight']
        center = (32 - 16) // 2
        cropped_conv = original_conv[:, :, center:center+16, center:center+16]
        model.vision_model.embeddings.patch_embedding.weight.data = cropped_conv
        # 处理位置编码：继承前50个位置，其余部分随机初始化
        original_pos = state_dict['vision_model.embeddings.position_embedding.weight']
        model_pos = model.vision_model.embeddings.position_embedding.weight
        model_pos.data[:50] = original_pos
        nn.init.normal_(model_pos[50:], std=0.01)
        # 加载其他兼容权重
        compatible_keys = [
            'text_model.encoder',
            'text_projection.weight',
            'visual_projection.weight',
            'logit_scale'
        ]
        for key in compatible_keys:
            if key in state_dict:
                parts = key.split('.')
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                target_param = getattr(obj, parts[-1])
                if target_param.shape != state_dict[key].shape:
                    print(f"跳过不兼容参数: {key} (预期 {target_param.shape}, 实际 {state_dict[key].shape})")
                    continue
                target_param.data.copy_(state_dict[key])
        nn.init.normal_(model.vision_projection.weight, std=0.01)
        nn.init.normal_(model.text_projection.weight, std=0.01)
        nn.init.constant_(model.vision_projection.bias, 0.0)
        nn.init.constant_(model.text_projection.bias, 0.0)
        return model
