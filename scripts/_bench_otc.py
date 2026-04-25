"""Quick 10-round bench for bitcoin_otc."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.online.loop import run_online_simulation

cfg_path = "configs/online/bitcoin_otc_full.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

cfg["total_rounds"] = 10
cfg["runtime"]["log_every"] = 1

t0 = time.time()
run_online_simulation(cfg)
elapsed = time.time() - t0
print(f"\n10 rounds total: {elapsed:.1f}s  ({elapsed/10:.2f}s/round)")
