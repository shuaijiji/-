# train.py - 修正后完整代码
# 在代码最开头添加
import resource
# 将文件描述符限制提升到65535
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, rlimit[1]))
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行
os.environ["OMP_NUM_THREADS"] = "1"             # 限制OpenMP线程
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPConfig, CLIPVisionConfig, CLIPTextConfig, CLIPProcessor
from config import Config
from report_extractor import load_json_data
from clip_dataset import CLIPDataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.distributed import DistributedSampler
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
# ==== 新增：命令行参数解析 ====
import argparse
# 在训练脚本开头添加
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import fnmatch
# 在代码开头添加
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # 替代旧的NCCL设置
os.environ["NCCL_IB_DISABLE"] = "1"  # 如果使用非InfiniBand网络

from lowmem_clip import LowMemCLIP, MedicalCLIPConfig


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 分布式必需参数（必须保留）
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Total batch size across all GPUs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", help="启用混合精度训练")
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # 数据加载参数
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_mmap", action="store_true")
    
    # 医疗影像特化参数
    parser.add_argument("--dicom", action="store_true",
                       help="输入为DICOM格式")
    

    # 解析已知参数，忽略其他（兼容torchrun参数）
    args, _ = parser.parse_known_args()

        # ==== 关键修改：动态更新Config ====
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.NUM_WORKERS = args.num_workers
    Config.USE_MMAP = args.use_mmap
    Config.FP16 = args.fp16

    print(f"=== 参数验证 ===")
    print(f"命令行batch_size: {args.batch_size} | Config.BATCH_SIZE: {Config.BATCH_SIZE}")

    return args


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])  # [batch, 77]
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values
    }

def validate_config_system():

    # 通过模型类加载配置（关键修改）
    model = LowMemCLIP.from_pretrained(Config.MODEL_DIR)
    print("=== 关键参数验证 ===")
    print(f"视觉嵌入层形状: {model.vision_model.embeddings.patch_embedding.weight.shape}")
    print(f"位置编码长度: {model.vision_model.embeddings.position_embedding.num_embeddings}")
    print(f"文本层冻结状态: {sum(not p.requires_grad for p in model.text_model.parameters())}/{len(list(model.text_model.parameters()))}")
    test_config = model.config
    
    # 关键参数断言
    assert test_config.vision_config.patch_size == 16, f"当前patch_size: {test_config.vision_config.patch_size}"
    assert test_config.vision_config.image_size == 224, f"当前image_size: {test_config.vision_config.image_size}"

    
    # 模型初始化测试
    test_model = LowMemCLIP(test_config)
    
    # 维度验证
    test_input = torch.randn(1, 3, 224, 224)
    embeddings = test_model.vision_model.embeddings(test_input)
    assert embeddings.shape[1] == (224//16)**2 + 1, "视觉嵌入维度错误"
    
    # 权重加载测试
    loaded = test_model.load_state_dict(
        torch.load(Config.PRETRAINED_WEIGHTS), 
        strict=False
    )
    assert len(loaded.missing_keys) == 0, f"权重缺失: {loaded.missing_keys}"

    # ==== 新增class_embedding验证 ====
    original_class_embed = torch.load(
        os.path.join(Config.MODEL_DIR, "pytorch_model.bin"),
        map_location="cpu"
    )['vision_model.embeddings.class_embedding']

    current_class_embed = model.vision_model.embeddings.class_embedding.data.cpu()

    print("\n=== Class Embedding验证 ===")
    print(f"原始形状: {original_class_embed.shape}")
    print(f"当前形状: {current_class_embed.shape}")
    print(f"最大差异: {(original_class_embed - current_class_embed).abs().max().item():.4f}")
    # =================================

    current_class_embed = model.vision_model.embeddings.class_embedding.data.cpu()

    print(f"形状匹配: {original_class_embed.shape == current_class_embed.shape}")
    print(f"数值相似度: {F.cosine_similarity(original_class_embed.flatten(), current_class_embed.flatten(), dim=0):.4f}")

def validate_distributed_shapes(model, dataset, image_size=224):
    device = next(model.parameters()).device

        # 测试前添加验证
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        embeddings = model.vision_model.embeddings(test_input)
        print(f"嵌入形状验证: {embeddings.shape} (应如 [1, 197, 768])")

    # 确保传入的是原始模型
    if isinstance(model, DDP):
        model = model.module
    assert not isinstance(model, DDP), "必须传入原始模型，而非DDP包装对象"



    # ==== 动态获取参数 ====
    patch_size = model.patch_size
    image_size = model.image_size

        # 测试自定义视觉层
    test_image = torch.randn(1, 3, image_size, image_size).to(device)

        # ==== 前向传播验证 ====
    with torch.no_grad():
        embeddings = model.vision_model.embeddings(test_image)
        #print(f"Embeddings shape: {embeddings.shape} (应如 [1, {(image_size//patch_size)**2}, 768])")


        # 测试完整前向传播
    with torch.no_grad():
        outputs = model.vision_model(test_image)
        #print(f"视觉模型输出形状: {outputs.last_hidden_state.shape} (应如 [1, 50, 768])")

        # ==== 新增参数处理 ====
    if image_size is None:
        image_size = model.image_size  # 从模型属性获取

    
        # 计算预期维度
    expected_num_patches = (model.image_size // model.patch_size) ** 2 + 1

    """简化验证流程，使用固定小批量"""
    # 仅验证前100个样本
    subset = torch.utils.data.Subset(dataset, indices=range(100))
    temp_loader = DataLoader(
        subset,
        batch_size=8,  # 固定小批量
        collate_fn=collate_fn,
        num_workers=0
    )

        # 生成测试图像
    # 生成符合实际分布的测试图像（0-255范围）
    test_image = torch.randint(0, 255, (1, 3, image_size, image_size), dtype=torch.float32).to(device)
    test_image = test_image / 255.0  # 归一化到[0,1]
    
    # 单次前向验证
    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type='cuda',  # 新增参数
        dtype=torch.float16,  # 明确指定精度类型
        enabled=True          # 确保启用
    ):
        batch = next(iter(temp_loader))
        outputs = model.vision_model(pixel_values=test_image, output_hidden_states=True)
    
    # 关键维度检查
    embeddings = outputs.last_hidden_state
    print(f"""
    === 视觉模型维度验证 ===
    输入尺寸: {test_image.shape}
    特征嵌入尺寸: {embeddings.shape} (应如 [1, 257, 768])
    CLS token尺寸: {outputs.pooler_output.shape}
    位置编码尺寸: {model.vision_model.embeddings.patch_embedding.weight.shape}
    卷积核尺寸: {model.vision_model.embeddings.position_embedding.weight.shape}
    """)

    # 卷积核尺寸验证
    current_kernel = model.vision_model.embeddings.patch_embedding.weight
    expected_kernel_size = model.vision_model.embeddings.patch_size
    assert current_kernel.shape[-2:] == (expected_kernel_size, expected_kernel_size), \
        f"卷积核尺寸{current_kernel.shape}与配置{expected_kernel_size}不符"
    
    # 位置编码验证
    num_positions = (model.image_size // model.patch_size)**2 + 1
    assert model.vision_model.embeddings.position_embedding.num_embeddings == num_positions, \
        f"位置编码长度{model.vision_model.embeddings.position_embedding.num_embeddings}应等于{num_positions}"

    # 在validate_distributed_shapes函数中添加
    #print(f"当前patch_size: {model.patch_size}")

    #print(f"卷积核尺寸验证: {model.vision_model.embeddings.patch_embedding.weight.shape}")
    assert model.vision_model.embeddings.patch_embedding.weight.shape[-2:] == (16, 16), "卷积核尺寸错误"

    pos_embed = model.vision_model.embeddings.position_embedding.weight
    #print(f"位置编码范数验证 - 均值: {pos_embed.mean().item():.4f} 标准差: {pos_embed.std().item():.4f}")

    assert embeddings.shape[1] == (image_size // model.patch_size)**2 + 1, "特征图维度不匹配！"

        # 在 validate_distributed_shapes 函数中添加
    print("Class embedding shape:", model.vision_model.embeddings.class_embedding.shape)
    # 应输出：torch.Size([768])

    print("Patch embedding shape:", model.vision_model.embeddings.patch_embedding.weight.shape)
    # 应输出：torch.Size([768, 3, 16, 16])

    print("Position embedding shape:", model.vision_model.embeddings.position_embedding.weight.shape)
    # 应输出：torch.Size([197, 768])

def load_adapted_weights(model, pretrained_path):
    
    # 从DDP模型中获取原始模型
    raw_model = model.module if isinstance(model, DDP) else model

    # 获取当前设备
    device = raw_model.vision_model.embeddings.patch_embedding.weight.device

        # 添加尺寸验证
    current_kernel = raw_model.vision_model.embeddings.patch_embedding.weight
    print(f"当前模型卷积核尺寸: {current_kernel.shape}")  # 应为[768,3,16,16]

    #----32x32 → 16x16----
    original_weights = torch.load(pretrained_path, map_location=device)
    #original_weights = torch.load(pretrained_path, map_location="cpu", weights_only=True)

        # 键名兼容处理
    key_alias = {
        'embeddings.position_embedding.weight': 'vision_model.embeddings.position_embedding.weight',
        'vision_model.position_embedding.weight': 'vision_model.embeddings.position_embedding.weight'
    }

    for alias, true_key in key_alias.items():
        if alias in original_weights and true_key not in original_weights:
            original_weights[true_key] = original_weights.pop(alias)

    # 处理视觉层权重

    original_patch_weights = original_weights['vision_model.embeddings.patch_embedding.weight'].to(device)
    
    # 当预训练是32x32而当前是16x16时，裁剪中心部分
    target_patch_size = model.vision_model.embeddings.patch_size  # 从模型获取实际配置
    original_size = original_patch_weights.shape[-1]  # 原始卷积核尺寸
    start = (original_size - target_patch_size) // 2
    cropped_kernel = original_patch_weights[
        :, :, 
        start:start+target_patch_size, 
        start:start+target_patch_size
    ]

    # 添加尺寸验证
    assert cropped_kernel.shape[-2:] == (target_patch_size, target_patch_size), \
        f"裁剪后尺寸{cropped_kernel.shape}与目标{target_patch_size}不匹配"
    
    # 加载到模型
    model.vision_model.embeddings.patch_embedding.weight.data.copy_(cropped_kernel)
    '''
    start = (32 - 16) // 2  # 计算中心16x16区域 
    cropped_kernel = original_patch_weights[:, :, start:start+16, start:start+16]
    cropped_kernel = original_weights['vision_model.embeddings.patch_embedding.weight']
    '''
        
    # 加载时添加断言
    assert cropped_kernel.shape == current_kernel.shape, f"卷积核尺寸不匹配！权重:{cropped_kernel.shape} 模型:{current_kernel.shape}"

    # 只加载文本层和投影层权重
    # 因为配置文件中patch_size=32，与预训练权重一致
    # 创建适配后的权重字典
    new_state_dict = {
        'vision_model.embeddings.patch_embedding.weight': cropped_kernel.to(device),
        'text_projection.weight': original_weights.get(  # 使用圆括号
            'text_projection.weight', 
            raw_model.text_projection.weight  # 使用原始模型参数作为默认
        ).to(device)
    }


    # ==== 加载其他兼容权重 ====
    compatible_keys = [
        'text_model', 
        #'text_projection.weight',
        #'visual_projection.weight'
    ]
    for key in original_weights:
        if any(k in key for k in compatible_keys):
            new_state_dict[key] = original_weights[key].to(device)
    
    # ==== 加载适配后的权重 ====
    missing, unexpected = raw_model.load_state_dict(new_state_dict, strict=False)
    print(f"成功加载权重 | 缺失键: {missing} | 意外键: {unexpected}")

    # ==== 初始化新增层 ====
    nn.init.kaiming_normal_(raw_model.vision_model.embeddings.patch_embedding.weight)
    nn.init.normal_(raw_model.vision_projection.weight, std=0.02)



    # 处理位置编码
    original_pos_embed = original_weights['vision_model.embeddings.position_embedding.weight'].to(device)
    current_pos_embed = raw_model.vision_model.embeddings.position_embedding.weight
    if original_pos_embed.shape[0] < current_pos_embed.shape[0]:
        # 插值扩展位置编码
        new_embed = F.interpolate(
            original_pos_embed.permute(1, 0).unsqueeze(0),  # [hidden, num_positions] → [1, 1, hidden, num_positions]
            size=current_pos_embed.shape[0],
            mode='linear',
            align_corners=True
        ).squeeze().permute(1, 0)  # 恢复形状
        current_pos_embed.data.copy_(new_embed)


    # 冻结文本模型
    '''
    for param in raw_model.text_model.parameters():
        param.requires_grad = False
    raw_model.text_projection.requires_grad = True  # 允许微调
    '''
    

def train_clip(epochs=10):
    args = parse_args()

    # 使用 torchrun 提供的环境变量（关键修改）
    local_rank = int(os.environ["LOCAL_RANK"])  # 必须从环境变量获取
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0

    # ==== 新增：分布式初始化 ====
    # 在train_clip函数开头添加（在parse_args之后）
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    torch.distributed.init_process_group(backend='nccl', init_method='env://')  # 简化初始化


    # 绑定到指定GPU（关键步骤）
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    """完整的训练流程"""
    # ==== 初始化模型 ====
    # 模型必须在绑定设备后初始化
    # 模型加载
    model = LowMemCLIP.from_pretrained(
        Config.MODEL_DIR,
        ignore_mismatched_sizes=True,  # 关键参数
        local_files_only=True,          # 确保使用本地模型
        ).to(device)

    load_adapted_weights(model, "/home/lijiaji/RAG/clip-vit-base-patch32/original_weights.pth")

    # 分布式包装
    model = DDP(model, 
            device_ids=[local_rank],
            find_unused_parameters=True,  # 避免梯度同步失败
            gradient_as_bucket_view=True,
            #static_graph=True  # 静态图优化
            )  # 节省内存
    
    # 确保权重加载到对应设备
    raw_model = model.module if hasattr(model, 'module') else model # 获取原始模型
    device = raw_model.text_projection.weight.device

        # 1. 冻结参数部分
    #for param in raw_model.vision_model.parameters():
        #param.requires_grad = False


    # ==== 数据加载 ====
    print("\n\033[1;34m=== 正在加载数据集 ===\033[0m")
    dataset = CLIPDataset(split='train')
    #train_data = load_json_data(Config.ANNOTATION_PATH, split="train")
    #dataset = CLIPDataset(train_data)


    # 检查模型初始化状态
    # ==== 修改后的权重检查 ====
    if local_rank == 0:

        validate_distributed_shapes(model = raw_model, dataset=dataset, image_size=model.module.image_size)  # 注意使用.module

        original_weights = torch.load('/home/lijiaji/RAG/clip-vit-base-patch32/original_weights.pth', map_location='cpu', weights_only=True)
        
        # 获取当前模型权重
        current_patch_weights = model.module.vision_model.embeddings.patch_embedding.weight.cpu()
         # 原始权重形状 [768, 3, 32, 32] → 提取中心 16x16 区域
        original_patch_weights = original_weights['vision_model.embeddings.patch_embedding.weight']
        original_size = original_patch_weights.shape[-1]
        start = (original_size - 16) // 2
        cropped_original = original_patch_weights[:, :, start:start+16, start:start+16]


        # 对比中心区域
        vision_match = torch.allclose(
            current_patch_weights,
            cropped_original.to(current_patch_weights.device),  # 与权重文件键名完全一致
            atol=1e-3
        )
        
        print(f"视觉层中心区域匹配: {vision_match}")


    loaded_weight = torch.load('/home/lijiaji/RAG/clip-vit-base-patch32/original_weights.pth', map_location=device)['text_projection.weight']
    print("文本投影层是否加载预训练权重:",
        torch.allclose(model.module.text_projection.weight,
                        loaded_weight.to(device), atol=1e-3))
    # 预期输出：True


    # ==== 优化器参数设置 ====
    optimizer_params = [
        {'params': model.module.vision_model.parameters(), 'lr': 5e-6}, # 1e-5新初始化层需要更高的学习率
        #{'params': model.module.text_model.parameters(), 'lr': 3e-5*2},
        {'params': model.module.visual_projection.parameters(), 'lr': 1e-5},#5e-5 *2
        {'params': model.module.text_projection.parameters(), 'lr': 3e-6} #1e-4 *2
    ]


    # ==== 初始化优化器 ====
    optimizer = torch.optim.AdamW(optimizer_params)

    
    # 打印当前学习率（假设 optimizer 有多个参数组）：
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i} learning rate: {group['lr']}")

    scaler = torch.amp.GradScaler(enabled=args.fp16)  # 混合精度训练


    # ==== 数据加载 ====
    # 在数据加载完成后添加
    dist.barrier()
    print(f"Rank {dist.get_rank()} 数据加载完成")

    # 查看展开后的数据样本数量
    '''
    print(f"展开后数据量: {len(dataset)}")  # 应该=原始量×每报告图像数
    sample = dataset[0]
    print(f"样本结构：{ {k: v.shape for k, v in sample.items() if isinstance(v, torch.Tensor)} }")
    '''

    # ==== 修改数据加载器 ====
    train_sampler = DistributedSampler(
        dataset,
        shuffle=True,
        num_replicas=dist.get_world_size(),
        rank=local_rank
    )

    args.local_batch = args.batch_size // dist.get_world_size()
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=train_sampler,  # 使用分布式采样器
        collate_fn=collate_fn,
        num_workers=4, # 适当调整
        pin_memory=True,
        persistent_workers=False,  # 避免内存泄漏
        multiprocessing_context=None,
        prefetch_factor=2#None # 减少预加载批次 
        #multiprocessing_context='spawn' 
    )

    # 保持 OneCycleLR 调度器，但调整 max_lr 参数（例如增大 max_lr）：
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-5,  # 提高最大学习率5e-4 *2
        total_steps=len(train_loader) * Config.EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos'
    )



    # 在训练循环前添加
    # 在训练脚本中
    sample_batch = next(iter(train_loader))
    print(f"输入形状验证:")
    print(f"- pixel_values: {sample_batch['pixel_values'].shape} → [batch*2, 3, 224, 224]")
    print(f"- input_ids: {sample_batch['input_ids'].shape} → [batch, seq_len]")

    # ==== 新增：训练前验证 ====
    if Config.VALIDATE_BEFORE_TRAIN and local_rank == 0:
        print("\n\033[1;35m=== 执行分布式形状验证 ===\033[0m")
        validate_distributed_shapes(raw_model, dataset)  # 注意使用.module

    # 在训练循环开始前添加
    torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')  # 优化显存碎片
    torch.cuda.memory.set_per_process_memory_fraction(0.8)  # 限制显存使用

    print(f"[Rank {dist.get_rank()}] CUDA设备: {torch.cuda.current_device()}")
    print(f"[Rank {dist.get_rank()}] 模型设备: {next(model.parameters()).device}")


    # ==== 训练循环 ====
    print(f"\033[1;32m启动训练，共 {epochs} 个epoch，批量大小：{Config.BATCH_SIZE}\033[0m")

    accum_steps = 2  # 新增梯度累积
    total_steps = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_sampler.set_epoch(epoch)  # 重要！设置epoch保证shuffle正确
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        total_loss = 0.0

        loss_fn = nn.CrossEntropyLoss()

        # 前3个epoch只训练投影层
        if epoch < 5:
            for param in model.module.vision_model.parameters():
                param.requires_grad = False
        # 在训练 loop 中，当 epoch 达到某个阈值时解冻视觉模型，并重新初始化 optimizer：
        # 示例：在某个 epoch 后解冻文本模型后几层
        if epoch == 5:
            for name, param in model.module.text_model.named_parameters():
                if "encoder.layers.4" in name or "encoder.layers.5" in name:
                    print(name, param.requires_grad)
                # 例如只解冻最后一层
                #if "encoder.layers.11" in name:
                    #param.requires_grad = True
            # 重新构造优化器（加入已解冻的参数）
            optimizer = torch.optim.AdamW([
                {'params': model.module.vision_model.parameters(), 'lr': 1e-4},
                {'params': model.module.text_model.parameters(), 'lr': 3e-5},
                {'params': model.module.visual_projection.parameters(), 'lr': 5e-5},
                {'params': model.module.text_projection.parameters(), 'lr': 1e-4},
            ])
            print("解冻部分文本模型参数，并更新优化器。")



        for step, batch in enumerate(progress_bar):

            # ==== 混合精度训练 ====
            with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16, enabled=args.fp16):
                
                # 前向传播
                logits_per_image, logits_per_text = model(
                    pixel_values=batch['pixel_values'].to(device),
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )

                logits_per_image = logits_per_image / Config.TEMPERATURE
                logits_per_text = logits_per_text / Config.TEMPERATURE

                                # ==== 手动计算对比损失 ====
                batch_size = logits_per_image.shape[0] #自动适配实际batch大小
                #labels = torch.arange(batch_size, device=device)
                #global_batch_size = batch_size * dist.get_world_size()
                labels = torch.arange(batch_size, device=device)
                
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_txt = F.cross_entropy(logits_per_text, labels)
                
                # 类内一致性正则
                #intra_loss = F.mse_loss(logits_per_image, logits_per_text.T.detach())

                loss = (loss_img + loss_txt) / 2 # + 0.1 * intra_loss
                
                # 梯度累积
                loss = loss / accum_steps


            # ==== 反向传播 ====
            scaler.scale(loss).backward()

            # 累积梯度更新
            if (step + 1) % accum_steps == 0:
                # 打印视觉投影层和文本投影层的梯度范数
                grad_norm_text = model.module.text_projection.weight.grad.norm().item() if model.module.text_projection.weight.grad is not None else 0.0
                print(f"Step {total_steps}: text_projection grad norm = {grad_norm_text:.4f}")
    
    
                # 梯度同步
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 梯度更新
                # 增加梯度裁剪（train.py训练循环中）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 添加严格的梯度裁剪
                scaler.step(optimizer)
                scaler.update()

                scheduler.step() # 更新学习率

                optimizer.zero_grad() # 清空梯度
                total_steps += 1
                
                # 分布式日志（仅rank0打印）
                if local_rank == 0 and total_steps % 10 == 0:
                    print(f"Step {total_steps} Loss: {loss.item():.4f}")

            # 调试时打印 logits 和标签的数值范围
            if step % 10 == 0:
                print(f"Step {total_steps}: logits_per_image range: {logits_per_image.min().item():.4f} ~ {logits_per_image.max().item():.4f}")
                print(f"Step {total_steps}: labels: {labels}")

            # ==== 更新进度 ====
            total_loss += loss.item()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss/(progress_bar.n+1):.4f}"
            })

        # 每个epoch结束后添加
        torch.cuda.empty_cache() # 释放显存

        print(f"\n\033[1;36mEpoch {epoch+1} 完成 | 平均损失: {total_loss/len(train_loader):.4f}\033[0m")
    
    # ==== 新增：分布式环境清理 ====
    dist.destroy_process_group()

    # 在train_clip函数末尾添加，保存模型：
    if local_rank == 0:
        # 保存完整模型（结构+权重）
        model.module.save_pretrained(
            Config.MODEL_SAVE_DIR,
            save_config=True,  # 确保保存配置文件
            state_dict=model.module.state_dict()
        )

        # 单独保存processor
        processor = CLIPProcessor.from_pretrained(Config.MODEL_DIR)
        processor.save_pretrained(Config.MODEL_SAVE_DIR)
        # 额外保存PyTorch格式（可选）
        torch.save(
            model.module.state_dict(),
            os.path.join(Config.MODEL_SAVE_DIR, "pytorch_model.bin")
        )


if __name__ == "__main__":
    args = parse_args()
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.NUM_WORKERS = args.num_workers
    Config.USE_MMAP = args.use_mmap

    # 必须放在其他导入之前
    mp.set_start_method('spawn', force=True)  # 替代默认的fork

        # 加载数据集
    dataset = CLIPDataset(split='train')
    #train_data = load_json_data(Config.ANNOTATION_PATH, split="train")
    #dataset = CLIPDataset(train_data)
        # 数据验证
    print(f"展开后总样本数: {len(dataset)}")
    print(f"样本结构验证：")
    for i in [0, 100, len(dataset)-1]:
        sample = dataset[i]
        assert sample['input_ids'].shape == torch.Size([77]), \
            f"样本{i}文本维度错误: {sample['input_ids'].shape}"
        assert sample['pixel_values'].shape == torch.Size([2, 3, 224, 224]), \
            f"样本{i}图像维度错误: {sample['pixel_values'].shape}"
    print("所有样本维度验证通过 ✅")

    print("\033[1;33m=== 训练配置验证 ===\033[0m")
    print(f"使用GPU: {Config.GPU_IDS}")
    print(f"混合精度: {'启用' if Config.FP16 else '禁用'}")
    print(f"初始学习率: Vision分支 5e-5 | Text分支 3e-5")
    

        # ...其他代码...
    # 添加数据集完整性检查
    print(f"展开后样本数量验证: {len(dataset)}")
    assert len(dataset) > 0, "数据集为空！请检查数据路径"
    '''
    # 抽样检查前10个样本
    for i in range(10):
        try:
            sample = dataset[i]
            raw_sample_data = dataset.samples[i]
            # ==== 正确访问字段 ====
            for img_path in raw_sample_data['image_paths']:
                assert os.path.exists(img_path), f"图像路径不存在: {img_path}"
                print(f"验证通过: {img_path}")
            assert sample['input_ids'].numel() > 0, "文本嵌入为空"
        except Exception as e:
            print(f"样本{i}验证失败: {str(e)}")
            raise'''

    train_clip()
    print("\n\033[1;32m=== 训练完成 ===\033[0m")
