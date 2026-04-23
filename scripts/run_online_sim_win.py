"""Windows DLL 加载修复 + 在线仿真启动器。"""
import os, sys

# 在 import torch 之前注册 torch/lib 为 DLL 搜索目录
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_torch_lib)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, yaml
from src.online.loop import run_online_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

run_online_simulation(cfg)
