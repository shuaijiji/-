import torch
import torch.nn as nn
from train import LowMemCLIP, MedicalCLIPConfig
from transformers import CLIPModel

# 初始化自定义模型
config = MedicalCLIPConfig.get_config("/home/lijiaji/RAG/clip-vit-base-patch32")
model = LowMemCLIP(config)

# 加载官方模型
official_model = CLIPModel.from_pretrained("/home/lijiaji/RAG/clip-vit-base-patch32")
official_state_dict = official_model.state_dict()

# ---- 手动适配权重 ----
# 1. class_embedding
model.vision_model.embeddings.class_embedding.data.copy_(
    official_state_dict['vision_model.embeddings.class_embedding']
)

# 2. patch_embedding (中心裁剪)
original_patch = official_state_dict['vision_model.embeddings.patch_embedding.weight']
center = (32 - 16) // 2
cropped_patch = original_patch[:, :, center:center+16, center:center+16]
model.vision_model.embeddings.patch_embedding.weight.data.copy_(cropped_patch)

# 3. position_embedding (部分继承+初始化)
model_pos = model.vision_model.embeddings.position_embedding.weight.data
model_pos[:50] = official_state_dict['vision_model.embeddings.position_embedding.weight']
nn.init.trunc_normal_(model_pos[50:], std=0.02)

# 4. 加载其他兼容权重
compatible_keys = [
    'text_model',
    'visual_projection.weight',
    'text_projection.weight',
    'logit_scale'
]
for key in compatible_keys:
    if key in official_state_dict:
        model.load_state_dict({key: official_state_dict[key]}, strict=False)

# 保存适配后的权重
# 在权重生成时显式保存文本投影层
torch.save({
    'vision_model.embeddings.patch_embedding.weight': model.vision_model.embeddings.patch_embedding.weight,
    'text_projection.weight': model.text_projection.weight,  # 新增此行
    # 其他键...
}, "/home/lijiaji/RAG/clip-vit-base-patch32/original_weights.pth")

# 在 generate_weights.py 末尾添加
print("=== 权重验证 ===")
print(f"patch_embedding.shape: {cropped_patch.shape} ([768, 3, 16, 16])")
assert cropped_patch.shape == (768, 3, 16, 16), "卷积核裁剪尺寸错误！"

print("权重适配完成！")