import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset.real.college_msg import CollegeMsgDataset
ds = CollegeMsgDataset(raw_dir="data/raw", processed_dir="data/processed")
ds.load(force_reprocess=False)
print("meta:", ds.meta)
train, val, test = ds.get_splits()
print(f"train={len(train)} val={len(val)} test={len(test)}")
