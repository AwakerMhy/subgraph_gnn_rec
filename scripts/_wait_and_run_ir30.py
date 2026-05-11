"""等待 ir60 完成后自动启动 ir30（8 workers）。"""
import sqlite3, time, subprocess, sys, os

db = 'results/orchestrator/ir60_constlr5e5_multiseed_bidir/experiments.db'
print('Waiting for ir60 to finish (checking every 60s)...', flush=True)

while True:
    conn = sqlite3.connect(db)
    counts = dict(conn.execute('SELECT status, COUNT(*) FROM cells GROUP BY status').fetchall())
    conn.close()
    running = counts.get('running', 0)
    done = counts.get('completed', 0)
    total = sum(counts.values())
    print(f'  [{time.strftime("%H:%M:%S")}] {done}/{total} done, {running} running', flush=True)
    if running == 0:
        break
    time.sleep(60)

print(f'\n[{time.strftime("%H:%M:%S")}] ir60 complete! Launching ir30 with 8 workers...', flush=True)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONPATH'] = '.'

proc = subprocess.run(
    [sys.executable, '-u', 'scripts/orchestrator.py',
     '--spec', 'configs/sweep_spec_ir30_constlr5e5_multiseed_bidir.yaml',
     '--workers', '8'],
    env=env,
)
print(f'\nOrchestrator exited with code {proc.returncode}', flush=True)
