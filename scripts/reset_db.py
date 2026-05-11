"""Reset all completed cells to pending in a sweep DB."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sweep_name = sys.argv[1]
DB = ROOT / "results" / "orchestrator" / sweep_name / "experiments.db"

conn = sqlite3.connect(DB)
n = conn.execute(
    "UPDATE cells SET status='pending', started_at=NULL, finished_at=NULL, "
    "duration_s=NULL, exit_code=NULL WHERE status='completed'"
).rowcount
conn.commit()
print(f"Reset {n} cells to pending")
print(conn.execute("SELECT status, COUNT(*) FROM cells GROUP BY status").fetchall())
conn.close()
