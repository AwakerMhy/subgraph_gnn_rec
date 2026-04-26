import yaml, sys
from pathlib import Path
sys.path.insert(0, ".")
cfg = yaml.safe_load(Path("configs/online/bitcoin_alpha_thr_node_emb_pool_neg.yaml").read_text())
cfg["total_rounds"] = 5
cfg["runtime"]["log_every"] = 1
from src.online.loop import run_online_simulation
run_online_simulation(cfg)
