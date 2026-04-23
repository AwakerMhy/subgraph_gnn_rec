"""预处理 CollegeMsg 数据集到 data/processed/college_msg/"""
import sys
sys.path.insert(0, ".")
from src.dataset.real.college_msg import CollegeMsgDataset

ds = CollegeMsgDataset(raw_dir="data/raw", processed_dir="data/processed")
ds.process()
print("预处理完成")
