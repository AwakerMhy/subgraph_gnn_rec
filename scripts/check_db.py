import sqlite3, sys
db, dataset = sys.argv[1], sys.argv[2]
with sqlite3.connect(db) as c:
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT cell_id, status, exit_code, error_msg FROM cells WHERE dataset=?", (dataset,)
    ).fetchall()
    for r in rows:
        msg = (r["error_msg"] or "")[:60]
        print(f"{r['cell_id']:<40} {r['status']:<12} rc={r['exit_code']}  {msg}")
