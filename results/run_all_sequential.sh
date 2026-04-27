#!/bin/bash
PYTHON=/c/conda/envs/gnn/python.exe
cd /c/Users/12143/Desktop/pythonProject/gnn

echo '>>> Running: epinions_v2_random_topk10'
mkdir -p results/online/epinions_v2_random_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_random_topk10.yaml > results/online/epinions_v2_random_topk10.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_random_topk20'
mkdir -p results/online/epinions_v2_random_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_random_topk20.yaml > results/online/epinions_v2_random_topk20.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_topk10'
mkdir -p results/online/epinions_v2_gnn_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_topk10.yaml > results/online/epinions_v2_gnn_topk10.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_topk20'
mkdir -p results/online/epinions_v2_gnn_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_topk20.yaml > results/online/epinions_v2_gnn_topk20.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_node_emb_topk10'
mkdir -p results/online/epinions_v2_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_node_emb_topk10.yaml > results/online/epinions_v2_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_node_emb_topk20'
mkdir -p results/online/epinions_v2_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_node_emb_topk20.yaml > results/online/epinions_v2_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_node_emb_topk10'
mkdir -p results/online/epinions_v2_gnn_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_node_emb_topk10.yaml > results/online/epinions_v2_gnn_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_node_emb_topk20'
mkdir -p results/online/epinions_v2_gnn_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_node_emb_topk20.yaml > results/online/epinions_v2_gnn_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_random_topk10'
mkdir -p results/online/sx_askubuntu_v2_random_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_random_topk10.yaml > results/online/sx_askubuntu_v2_random_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_random_topk20'
mkdir -p results/online/sx_askubuntu_v2_random_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_random_topk20.yaml > results/online/sx_askubuntu_v2_random_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_topk10'
mkdir -p results/online/sx_askubuntu_v2_gnn_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_topk10.yaml > results/online/sx_askubuntu_v2_gnn_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_topk20'
mkdir -p results/online/sx_askubuntu_v2_gnn_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_topk20.yaml > results/online/sx_askubuntu_v2_gnn_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_node_emb_topk10'
mkdir -p results/online/sx_askubuntu_v2_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_node_emb_topk10.yaml > results/online/sx_askubuntu_v2_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_node_emb_topk20'
mkdir -p results/online/sx_askubuntu_v2_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_node_emb_topk20.yaml > results/online/sx_askubuntu_v2_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_node_emb_topk10'
mkdir -p results/online/sx_askubuntu_v2_gnn_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_node_emb_topk10.yaml > results/online/sx_askubuntu_v2_gnn_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_node_emb_topk20'
mkdir -p results/online/sx_askubuntu_v2_gnn_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_node_emb_topk20.yaml > results/online/sx_askubuntu_v2_gnn_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_random_topk10'
mkdir -p results/online/sx_superuser_v2_random_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_random_topk10.yaml > results/online/sx_superuser_v2_random_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_random_topk20'
mkdir -p results/online/sx_superuser_v2_random_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_random_topk20.yaml > results/online/sx_superuser_v2_random_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_topk10'
mkdir -p results/online/sx_superuser_v2_gnn_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_topk10.yaml > results/online/sx_superuser_v2_gnn_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_topk20'
mkdir -p results/online/sx_superuser_v2_gnn_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_topk20.yaml > results/online/sx_superuser_v2_gnn_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_node_emb_topk10'
mkdir -p results/online/sx_superuser_v2_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_node_emb_topk10.yaml > results/online/sx_superuser_v2_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_node_emb_topk20'
mkdir -p results/online/sx_superuser_v2_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_node_emb_topk20.yaml > results/online/sx_superuser_v2_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_node_emb_topk10'
mkdir -p results/online/sx_superuser_v2_gnn_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_node_emb_topk10.yaml > results/online/sx_superuser_v2_gnn_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_node_emb_topk20'
mkdir -p results/online/sx_superuser_v2_gnn_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_node_emb_topk20.yaml > results/online/sx_superuser_v2_gnn_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_random_topk10'
mkdir -p results/online/sbm10k_v2_random_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_random_topk10.yaml > results/online/sbm10k_v2_random_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_random_topk20'
mkdir -p results/online/sbm10k_v2_random_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_random_topk20.yaml > results/online/sbm10k_v2_random_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_topk10'
mkdir -p results/online/sbm10k_v2_gnn_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_topk10.yaml > results/online/sbm10k_v2_gnn_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_topk20'
mkdir -p results/online/sbm10k_v2_gnn_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_topk20.yaml > results/online/sbm10k_v2_gnn_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_node_emb_topk10'
mkdir -p results/online/sbm10k_v2_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_node_emb_topk10.yaml > results/online/sbm10k_v2_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_node_emb_topk20'
mkdir -p results/online/sbm10k_v2_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_node_emb_topk20.yaml > results/online/sbm10k_v2_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_node_emb_topk10'
mkdir -p results/online/sbm10k_v2_gnn_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_node_emb_topk10.yaml > results/online/sbm10k_v2_gnn_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_node_emb_topk20'
mkdir -p results/online/sbm10k_v2_gnn_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_node_emb_topk20.yaml > results/online/sbm10k_v2_gnn_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_random_topk10'
mkdir -p results/online/sbm20k_v2_random_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_random_topk10.yaml > results/online/sbm20k_v2_random_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_random_topk20'
mkdir -p results/online/sbm20k_v2_random_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_random_topk20.yaml > results/online/sbm20k_v2_random_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_topk10'
mkdir -p results/online/sbm20k_v2_gnn_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_topk10.yaml > results/online/sbm20k_v2_gnn_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_topk20'
mkdir -p results/online/sbm20k_v2_gnn_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_topk20.yaml > results/online/sbm20k_v2_gnn_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_node_emb_topk10'
mkdir -p results/online/sbm20k_v2_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_node_emb_topk10.yaml > results/online/sbm20k_v2_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_node_emb_topk20'
mkdir -p results/online/sbm20k_v2_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_node_emb_topk20.yaml > results/online/sbm20k_v2_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_node_emb_topk10'
mkdir -p results/online/sbm20k_v2_gnn_node_emb_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_node_emb_topk10.yaml > results/online/sbm20k_v2_gnn_node_emb_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_node_emb_topk20'
mkdir -p results/online/sbm20k_v2_gnn_node_emb_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_node_emb_topk20.yaml > results/online/sbm20k_v2_gnn_node_emb_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm5k_v2_gnn_lc_topk10'
mkdir -p results/online/sbm5k_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm5k_v2_gnn_lc_topk10.yaml > results/online/sbm5k_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm5k_v2_gnn_lc_topk20'
mkdir -p results/online/sbm5k_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm5k_v2_gnn_lc_topk20.yaml > results/online/sbm5k_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm5k_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/sbm5k_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm5k_v2_gnn_node_emb_lc_topk10.yaml > results/online/sbm5k_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm5k_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/sbm5k_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm5k_v2_gnn_node_emb_lc_topk20.yaml > results/online/sbm5k_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_alpha_v2_gnn_lc_topk10'
mkdir -p results/online/bitcoin_alpha_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_alpha_v2_gnn_lc_topk10.yaml > results/online/bitcoin_alpha_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_alpha_v2_gnn_lc_topk20'
mkdir -p results/online/bitcoin_alpha_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_alpha_v2_gnn_lc_topk20.yaml > results/online/bitcoin_alpha_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_alpha_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/bitcoin_alpha_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_alpha_v2_gnn_node_emb_lc_topk10.yaml > results/online/bitcoin_alpha_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_alpha_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/bitcoin_alpha_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_alpha_v2_gnn_node_emb_lc_topk20.yaml > results/online/bitcoin_alpha_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_otc_v2_gnn_lc_topk10'
mkdir -p results/online/bitcoin_otc_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_otc_v2_gnn_lc_topk10.yaml > results/online/bitcoin_otc_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_otc_v2_gnn_lc_topk20'
mkdir -p results/online/bitcoin_otc_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_otc_v2_gnn_lc_topk20.yaml > results/online/bitcoin_otc_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_otc_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/bitcoin_otc_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_otc_v2_gnn_node_emb_lc_topk10.yaml > results/online/bitcoin_otc_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: bitcoin_otc_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/bitcoin_otc_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/bitcoin_otc_v2_gnn_node_emb_lc_topk20.yaml > results/online/bitcoin_otc_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_lc_topk10'
mkdir -p results/online/epinions_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_lc_topk10.yaml > results/online/epinions_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_lc_topk20'
mkdir -p results/online/epinions_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_lc_topk20.yaml > results/online/epinions_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/epinions_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_node_emb_lc_topk10.yaml > results/online/epinions_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: epinions_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/epinions_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/epinions_v2_gnn_node_emb_lc_topk20.yaml > results/online/epinions_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_lc_topk10'
mkdir -p results/online/sx_askubuntu_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_lc_topk10.yaml > results/online/sx_askubuntu_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_lc_topk20'
mkdir -p results/online/sx_askubuntu_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_lc_topk20.yaml > results/online/sx_askubuntu_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/sx_askubuntu_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_node_emb_lc_topk10.yaml > results/online/sx_askubuntu_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_askubuntu_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/sx_askubuntu_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_askubuntu_v2_gnn_node_emb_lc_topk20.yaml > results/online/sx_askubuntu_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_lc_topk10'
mkdir -p results/online/sx_superuser_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_lc_topk10.yaml > results/online/sx_superuser_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_lc_topk20'
mkdir -p results/online/sx_superuser_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_lc_topk20.yaml > results/online/sx_superuser_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/sx_superuser_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_node_emb_lc_topk10.yaml > results/online/sx_superuser_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sx_superuser_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/sx_superuser_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sx_superuser_v2_gnn_node_emb_lc_topk20.yaml > results/online/sx_superuser_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_lc_topk10'
mkdir -p results/online/sbm10k_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_lc_topk10.yaml > results/online/sbm10k_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_lc_topk20'
mkdir -p results/online/sbm10k_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_lc_topk20.yaml > results/online/sbm10k_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/sbm10k_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_node_emb_lc_topk10.yaml > results/online/sbm10k_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm10k_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/sbm10k_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm10k_v2_gnn_node_emb_lc_topk20.yaml > results/online/sbm10k_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_lc_topk10'
mkdir -p results/online/sbm20k_v2_gnn_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_lc_topk10.yaml > results/online/sbm20k_v2_gnn_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_lc_topk20'
mkdir -p results/online/sbm20k_v2_gnn_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_lc_topk20.yaml > results/online/sbm20k_v2_gnn_lc_topk20.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_node_emb_lc_topk10'
mkdir -p results/online/sbm20k_v2_gnn_node_emb_lc_topk10
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_node_emb_lc_topk10.yaml > results/online/sbm20k_v2_gnn_node_emb_lc_topk10.log 2>&1
echo '    done.'
echo '>>> Running: sbm20k_v2_gnn_node_emb_lc_topk20'
mkdir -p results/online/sbm20k_v2_gnn_node_emb_lc_topk20
$PYTHON scripts/run_online_sim_win.py --config configs/online/sbm20k_v2_gnn_node_emb_lc_topk20.yaml > results/online/sbm20k_v2_gnn_node_emb_lc_topk20.log 2>&1
echo '    done.'
