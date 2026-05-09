import sqlite3
from pathlib import Path

DB = Path(__file__).parent.parent / "results" / "orchestrator" / "ir50_constlr5e5_multiseed_bidir" / "experiments.db"
conn = sqlite3.connect(DB)
n = conn.execute(
    "UPDATE cells SET status='pending', started_at=NULL, finished_at=NULL, duration_s=NULL, exit_code=NULL WHERE status='completed'"
).rowcount
conn.commit()
print(f"Reset {n} cells to pending")
verify = conn.execute("SELECT status, COUNT(*) FROM cells GROUP BY status").fetchall()
print(verify)
conn.close()
