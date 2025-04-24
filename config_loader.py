# ADDED: config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any
import torch

class HardwareConfig:
    def __init__(self, config_path: str = "RAG/configs/hardware.yaml"):
        self.config = self._load_config(config_path)
        self._init_memory_management()  # 新增

    def _init_memory_management(self):
        if self.gpu_settings.get('enabled', False):
            torch.cuda.empty_cache()
            # 设置每卡显存上限
            torch.cuda.set_per_process_memory_fraction(
                self.gpu_settings.get('max_mem_fraction', 0.8)
            )
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(Path(__file__).parent.parent / path, "r") as f:
            return yaml.safe_load(f)
    
    @property
    def gpu_settings(self):
        return self.config.get('gpu', {})
    
    @property
    def cpu_settings(self):
        return self.config.get('cpu', {})

# 初始化配置
hw_config = HardwareConfig()

# 在config_loader.py中增加
import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_name', type=str, help='Parameter name')
    args, _ = parser.parse_known_args()
    return vars(args)

def get(param_name: str, default=None):
    # 1. 尝试从YAML获取
    value = hw_config.config.get(param_name, None)
    
    # 2. 尝试从命令行参数获取
    if value is None:
        value = parse_cli_args().get(param_name)
    
    # 3. 使用默认值
    return value or default