import yaml, sys
from pathlib import Path
sys.path.insert(0, ".")
cfg = yaml.safe_load(Path("configs/online/bitcoin_alpha_thr_node_emb_pool_neg.yaml").read_text())
from src.online.loop import run_online_simulation
run_online_simulation(cfg)
