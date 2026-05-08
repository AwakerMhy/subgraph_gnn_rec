"""Two-phase resume: first finish wiki_vote, then launch all remaining with workers=8."""
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB = ROOT / "results" / "orchestrator" / "ir40_constlr5e5_multiseed_bidir" / "experiments.db"
SPEC = ROOT / "configs" / "sweep_spec_ir40_constlr5e5_multiseed_bidir.yaml"
PYTHON = sys.executable

PHASE2_WORKERS = 8


def db_set_status(conn, from_status: str, to_status: str, dataset_filter=None):
    if dataset_filter:
        placeholders = ",".join("?" * len(dataset_filter))
        sql = f"UPDATE cells SET status=? WHERE status=? AND dataset IN ({placeholders})"
        params = [to_status, from_status] + list(dataset_filter)
    else:
        sql = "UPDATE cells SET status=? WHERE status=?"
        params = [to_status, from_status]
    n = conn.execute(sql, params).rowcount
    conn.commit()
    return n


def count_pending(conn):
    return conn.execute("SELECT COUNT(*) FROM cells WHERE status='pending'").fetchone()[0]


def main():
    conn = sqlite3.connect(DB)

    # ── Phase 1: hold everything except wiki_vote ──────────────────────────────
    big_datasets = ["epinions", "slashdot", "sx_askubuntu", "sx_mathoverflow", "sx_superuser"]
    n_held = db_set_status(conn, "pending", "hold", big_datasets)
    print(f"[phase1] Held {n_held} cells for big datasets")

    wiki_pending = count_pending(conn)
    conn.close()
    print(f"[phase1] Running wiki_vote ({wiki_pending} cells), workers=1 (serial)")

    ret = subprocess.run(
        [PYTHON, "-u", str(ROOT / "scripts" / "orchestrator.py"),
         "--spec", str(SPEC), "--resume", "--workers", "1"],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    print(f"[phase1] orchestrator exited with code {ret.returncode}")

    # ── Phase 2: restore big datasets and launch with workers=8 ───────────────
    conn = sqlite3.connect(DB)
    n_restored = db_set_status(conn, "hold", "pending")
    remaining = count_pending(conn)
    conn.close()
    print(f"[phase2] Restored {n_restored} cells -> pending  (total pending: {remaining})")
    print(f"[phase2] Launching orchestrator --workers {PHASE2_WORKERS}")

    subprocess.run(
        [PYTHON, "-u", str(ROOT / "scripts" / "orchestrator.py"),
         "--spec", str(SPEC), "--resume", "--workers", str(PHASE2_WORKERS)],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    print("[done] All phases complete.")


if __name__ == "__main__":
    main()
