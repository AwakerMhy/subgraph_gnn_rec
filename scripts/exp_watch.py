#!/usr/bin/env python3
"""Real-time experiment progress visualization."""
import sqlite3
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).parent.parent
ORCH_DIR = ROOT / "results" / "orchestrator"

def find_latest_sweep():
    """Find sweep with most recent experiments.db."""
    latest = None
    latest_time = 0
    for db_path in ORCH_DIR.glob("*/experiments.db"):
        mtime = db_path.stat().st_mtime
        if mtime > latest_time:
            latest_time = mtime
            latest = db_path.parent.name
    return latest

def read_cells(sweep_name):
    """Read cells from DB."""
    db_path = ORCH_DIR / sweep_name / "experiments.db"
    conn = sqlite3.connect(db_path)
    cells = pd.read_sql("SELECT * FROM cells", conn)
    conn.close()
    return cells

def format_progress_bar(done, total, width=20):
    """ASCII progress bar: # = completed, - = running, . = pending/hold."""
    if total == 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"

def parse_round_from_log(log_path):
    """Parse latest Round line from log file."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        # Find lines with "Round"
        round_lines = [l for l in lines if 'Round' in l]
        if not round_lines:
            return None
        # Parse last Round line: "Round  22/30  coverage=0.419  uauc=0.551  (12.3s)"
        last = round_lines[-1].strip()
        m = re.search(r'Round\s+(\d+)/(\d+).*?uauc=([0-9.]+|-)\s*\(([0-9.]+)s', last)
        if m:
            return {
                'round': f"{m.group(1)}/{m.group(2)}",
                'uauc': m.group(3),
                'time_per_round': m.group(4)
            }
    except:
        pass
    return None

def main():
    sweep_name = sys.argv[1] if len(sys.argv) > 1 else find_latest_sweep()
    if not sweep_name:
        print("No experiments found.")
        return

    cells = read_cells(sweep_name)

    # === Overall progress ===
    done = len(cells[cells['status'] == 'completed'])
    total = len(cells)
    pct = 100 * done / total

    # ETA: average duration of last 20 completed cells * remaining
    remaining = len(cells[cells['status'].isin(['pending', 'hold'])])
    avg_duration = cells[cells['status'] == 'completed']['duration_s'].tail(20).mean()
    if pd.isna(avg_duration):
        eta_str = "unknown"
    else:
        eta_sec = int(avg_duration * remaining)
        if eta_sec < 3600:
            eta_str = f"~{eta_sec//60}m"
        else:
            eta_str = f"~{eta_sec//3600}h{(eta_sec%3600)//60}m"

    print(f"=== {sweep_name}  {done}/{total} ({pct:.1f}%) ===")
    print(f"{format_progress_bar(done, total)}  {pct:.1f}%   ETA {eta_str}")
    print()

    # === Per-dataset progress ===
    print("Per-dataset:")
    for ds in sorted(cells['dataset'].unique()):
        ds_cells = cells[cells['dataset'] == ds]
        ds_done = len(ds_cells[ds_cells['status'] == 'completed'])
        ds_total = len(ds_cells)

        running = len(ds_cells[ds_cells['status'] == 'running'])
        hold = len(ds_cells[ds_cells['status'] == 'hold'])

        status_str = ""
        if ds_done == ds_total:
            status_str = "OK"
        else:
            parts = []
            if running > 0:
                parts.append(f"{running} running")
            if hold > 0:
                parts.append(f"{hold} hold")
            status_str = ", ".join(parts) if parts else "pending"

        print(f"  {ds:20s} {format_progress_bar(ds_done, ds_total)}  {ds_done:3d}/{ds_total:3d}  {status_str}")
    print()

    # === Running cells ===
    running_cells = cells[cells['status'] == 'running']
    if len(running_cells) > 0:
        print(f"Running ({len(running_cells)} cells):")
        for _, cell in running_cells.iterrows():
            log_path = cell['log_path']
            round_info = parse_round_from_log(log_path)
            if round_info:
                info_str = f"Round {round_info['round']}  uauc={round_info['uauc']}  ({round_info['time_per_round']}s/round)"
            else:
                info_str = "(starting...)"
            print(f"  {cell['dataset']:20s} / {cell['method']:15s} {cell['seed']}  {info_str}")
        print()

    # === Recently completed ===
    completed = cells[cells['status'] == 'completed'].copy()
    if len(completed) > 0:
        completed['finished_at'] = pd.to_datetime(completed['finished_at'], errors='coerce')
        recent = completed.nlargest(5, 'finished_at')
        print("Recently completed (last 5):")
        for _, cell in recent.iterrows():
            dur = int(cell['duration_s']) if pd.notna(cell['duration_s']) else 0
            print(f"  {cell['dataset']:20s} / {cell['method']:15s} {cell['seed']}  {dur:3d}s")
        print()

    # === Phase info ===
    if len(cells[cells['status'] == 'hold']) > 0:
        n_hold = len(cells[cells['status'] == 'hold'])
        print(f"Phase2 (hold): {n_hold} cells — will unlock after running phase completes")

if __name__ == "__main__":
    main()
