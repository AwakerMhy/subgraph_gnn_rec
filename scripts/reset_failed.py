import sqlite3, sys
db = sys.argv[1]
with sqlite3.connect(db) as c:
    rows = c.execute("UPDATE cells SET status='pending' WHERE status='failed'").rowcount
    print(f"Reset {rows} failed cells to pending")
