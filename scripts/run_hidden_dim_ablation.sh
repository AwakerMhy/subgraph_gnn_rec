#!/usr/bin/env bash
# 等待子图设计对比实验完成后，自动启动 hidden_dim 消融实验
# 用法：bash scripts/run_hidden_dim_ablation.sh

PYTHON="C:/conda/envs/gnn/python.exe"
LOG_DIR="results/logs"

WATCH_LOGS=(
  "$LOG_DIR/btc_bfs2hop_30ep.log"
  "$LOG_DIR/btc_egocn_30ep.log"
  "$LOG_DIR/eu_bfs2hop_30ep.log"
  "$LOG_DIR/eu_egocn_30ep.log"
)

echo "[$(date '+%H:%M:%S')] 等待子图设计对比实验完成..."

for log in "${WATCH_LOGS[@]}"; do
  echo "  监控: $log"
  until grep -qE "(训练完成|Early stopping|Epoch 30)" "$log" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] 完成: $log"
done

echo ""
echo "[$(date '+%H:%M:%S')] 所有对比实验完成！开始 hidden_dim 消融实验..."
echo "固定配置: CollegeMsg × ego_cn × simulated_recall × 30ep × seed=42"
echo ""

DIMS=(4 8 16 32 64)

for dim in "${DIMS[@]}"; do
  run_name="ablation_hdim${dim}_collegmsg"
  log_file="$LOG_DIR/ablation_hdim${dim}.log"
  echo "[$(date '+%H:%M:%S')] 启动 hidden_dim=$dim  →  $log_file"
  nohup "$PYTHON" -m src.train \
    --data_dir data/processed/college_msg \
    --run_name "$run_name" \
    --protocol simulated_recall \
    --subgraph_type ego_cn \
    --epochs 30 \
    --max_samples 2000 \
    --hidden_dim "$dim" \
    --num_layers 2 \
    --patience 10 \
    --seed 42 \
    --recall_method common_neighbors \
    --recall_top_k 100 \
    --device cpu \
    > "$log_file" 2>&1 &
  echo "  PID: $!"
  # 串行运行，避免内存/CPU 竞争
  wait $!
  echo "[$(date '+%H:%M:%S')] hidden_dim=$dim 完成"
done

echo ""
echo "[$(date '+%H:%M:%S')] 所有 hidden_dim 实验完成！"
echo "结果路径: results/logs/ablation_hdim*.log"
