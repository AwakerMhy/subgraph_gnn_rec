"""scripts/orchestrator.py — Autonomous experiment orchestrator for GNN online learning sweeps.

Usage:
    python scripts/orchestrator.py --spec configs/sweep_spec_smoke.yaml
    python scripts/orchestrator.py --spec configs/sweep_spec_full.yaml --resume
    python scripts/orchestrator.py --spec configs/sweep_spec_full.yaml --dry_run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import itertools
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
PYTHON = sys.executable
ROOT = Path(__file__).parent.parent
ORCH_DIR = ROOT / "results" / "orchestrator"
STATUS_PATH = ROOT / "STATUS.md"

# Per-sweep paths are resolved in Orchestrator.__init__ once sweep name is known
def _sweep_db(name: str) -> Path:
    return ORCH_DIR / name / "experiments.db"

def _sweep_cfg_dir(name: str) -> Path:
    return ROOT / "configs" / "orchestrator" / name

def _sweep_logs_dir(name: str) -> Path:
    return ORCH_DIR / name / "logs"

# Windows-specific crash exit codes (negative on Python side)
_WIN_CRASH_CODES: frozenset[int] = frozenset({
    -1073741819,  # 0xC0000005 access violation
    -1073741515,  # 0xC0000135 DLL not found
    -1073741571,  # 0xC00000FD stack overflow
    -1073740791,  # 0xC0000409 heap corruption
})

_OOM_KEYWORDS = (
    "out of memory",
    "cuda out of memory",
    "memoryerror",
    "cannot allocate memory",
    "runtimeerror: cuda",
)

STATUS_INTERVAL_S = 600  # 10 minutes

# Methods that have no GNN model (no trainer needed)
_NO_MODEL = frozenset({"random", "ground_truth", "cn", "aa", "jaccard", "pa"})


# ── Cell ───────────────────────────────────────────────────────────────────────
class Cell:
    __slots__ = ("cell_id", "dataset", "method", "seed", "n_nodes",
                 "config", "timeout_s", "initial_batch_size")

    def __init__(
        self,
        cell_id: str,
        dataset: str,
        method: str,
        seed: int,
        n_nodes: int,
        config: dict,
        timeout_s: int,
        initial_batch_size: int,
    ) -> None:
        self.cell_id = cell_id
        self.dataset = dataset
        self.method = method
        self.seed = seed
        self.n_nodes = n_nodes
        self.config = config
        self.timeout_s = timeout_s
        self.initial_batch_size = initial_batch_size


# ── Sweep Spec ─────────────────────────────────────────────────────────────────
class SweepSpec:
    """Parse sweep_spec.yaml and expand into an ordered list of Cell objects."""

    def __init__(self, spec_path: Path) -> None:
        with open(spec_path, encoding="utf-8") as f:
            self.spec = yaml.safe_load(f)
        self.name: str = self.spec.get("name", "sweep")

    def expand(self) -> list[Cell]:
        base = self.spec.get("base_config", {})
        grid = self.spec.get("grid", {})
        timeout_s: int = self.spec.get("timeouts", {}).get("per_cell_seconds", 3600)

        grid_keys = list(grid.keys())
        grid_vals = [v if isinstance(v, list) else [v] for v in grid.values()]
        combos = list(itertools.product(*grid_vals))
        multi_seed = len([v for v in grid_vals if len(v) > 1 and "seed" in grid_keys]) > 0

        cells: list[Cell] = []
        for combo in combos:
            gd = dict(zip(grid_keys, combo))
            seed = int(gd.get("seed", base.get("runtime", {}).get("seed", 42)))
            total_rounds = int(gd.get("total_rounds", base.get("total_rounds", 100)))
            hidden_dim = int(gd.get("hidden_dim", 32))

            for ds_spec in self.spec["datasets"]:
                ds_name: str = ds_spec["name"]
                n_nodes: int = ds_spec.get("n_nodes", 999999)

                for method in self.spec["methods"]:
                    cell_id = f"{ds_name}_{method}_s{seed}"

                    # out_dir: per-dataset subdir, method as leaf (plot-script compatible)
                    method_dir = method if not multi_seed else f"{method}_s{seed}"
                    out_dir = (f"results/orchestrator/runs/{self.name}"
                               f"/{ds_name}/{method_dir}")

                    cfg = self._build_config(
                        ds_spec, method, seed, total_rounds, hidden_dim, base, out_dir, n_nodes
                    )
                    batch_size = cfg.get("trainer", {}).get("min_batch_size", 4)
                    cells.append(Cell(
                        cell_id=cell_id,
                        dataset=ds_name,
                        method=method,
                        seed=seed,
                        n_nodes=n_nodes,
                        config=cfg,
                        timeout_s=timeout_s,
                        initial_batch_size=batch_size,
                    ))

        # Smallest-dataset-first, then method
        cells.sort(key=lambda c: (c.n_nodes, c.dataset, c.method))
        return cells

    def _build_config(
        self,
        ds_spec: dict,
        method: str,
        seed: int,
        total_rounds: int,
        hidden_dim: int,
        base: dict,
        out_dir: str,
        n_nodes: int,
    ) -> dict:
        cfg: dict[str, Any] = copy.deepcopy(base)

        # Dataset
        cfg["dataset"] = {"type": ds_spec["name"], "path": ds_spec["path"]}

        # Model
        if method == "random":
            cfg["model"] = {"type": "random"}
        elif method == "ground_truth":
            cfg["model"] = {"type": "ground_truth"}
        elif method in ("cn", "aa", "jaccard", "pa"):
            cfg["model"] = {"type": method}
        elif method == "mlp":
            cfg["model"] = {"type": "mlp", "hidden_dim": hidden_dim}
        elif method == "node_emb":
            cfg["model"] = {"type": "node_emb", "emb_dim": hidden_dim, "hidden_dim": hidden_dim}
        elif method == "gnn":
            cfg["model"] = {"type": "gnn", "hidden_dim": 8, "num_layers": 3,
                            "encoder_type": "last", "node_feat_dim": 0}
        elif method == "gnn_h32":
            cfg["model"] = {"type": "gnn", "hidden_dim": 32, "num_layers": 3,
                            "encoder_type": "last", "node_feat_dim": 0}
        elif method == "gnn_concat":
            cfg["model"] = {"type": "gnn", "hidden_dim": hidden_dim, "num_layers": 3,
                            "encoder_type": "layer_concat", "node_feat_dim": 0}
        elif method == "gnn_concat_h8":
            cfg["model"] = {"type": "gnn", "hidden_dim": 8, "num_layers": 3,
                            "encoder_type": "layer_concat", "node_feat_dim": 0}
        elif method == "gnn_sum":
            cfg["model"] = {"type": "gnn", "hidden_dim": hidden_dim, "num_layers": 3,
                            "encoder_type": "layer_sum", "node_feat_dim": 0}
        elif method == "gnn_sum_h8":
            cfg["model"] = {"type": "gnn", "hidden_dim": 8, "num_layers": 3,
                            "encoder_type": "layer_sum", "node_feat_dim": 0}
        elif method == "graphsage_emb":
            cfg["model"] = {"type": "graphsage_emb", "hidden_dim": hidden_dim,
                            "emb_dim": hidden_dim, "num_layers": 3}
        elif method == "gat_emb":
            cfg["model"] = {"type": "gat_emb", "hidden_dim": hidden_dim,
                            "emb_dim": hidden_dim, "num_layers": 3, "num_heads": 4}
        elif method == "seal":
            cfg["model"] = {"type": "seal", "hidden_dim": hidden_dim,
                            "num_layers": 3, "label_dim": 16}
        else:
            cfg["model"] = {"type": method}

        cfg["total_rounds"] = total_rounds

        # Scale sample_ratio by dataset size
        sel = cfg.setdefault("user_selector", {})
        if "sample_ratio" not in sel:
            sel["sample_ratio"] = 0.1 if n_nodes < 10_000 else 0.01
        if "strategy" not in sel:
            sel["strategy"] = "uniform"
        for k, v in {"alpha": 0.5, "beta": 2.0, "gamma": 2.0,
                     "lam": 0.1, "w": 3}.items():
            sel.setdefault(k, v)

        cfg.setdefault("runtime", {})
        cfg["runtime"]["seed"] = seed
        cfg["runtime"]["out_dir"] = out_dir
        cfg["runtime"].setdefault("device", "cpu")
        cfg["runtime"].setdefault("log_every", 1)

        return cfg


# ── Ledger ─────────────────────────────────────────────────────────────────────
class Ledger:
    """SQLite-backed persistent ledger. Thread-safe via a write lock."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cells (
        cell_id       TEXT PRIMARY KEY,
        dataset       TEXT    NOT NULL,
        method        TEXT    NOT NULL,
        seed          INTEGER NOT NULL,
        n_nodes       INTEGER NOT NULL,
        status        TEXT    NOT NULL DEFAULT 'pending',
        config_path   TEXT,
        out_dir       TEXT,
        log_path      TEXT,
        batch_size    INTEGER NOT NULL DEFAULT 4,
        oom_retried   INTEGER NOT NULL DEFAULT 0,
        started_at    TEXT,
        finished_at   TEXT,
        duration_s    REAL,
        exit_code     INTEGER,
        error_msg     TEXT
    )
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(self._SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30)

    # ── writes ─────────────────────────────────────────────────────────────────
    def register(self, cells: list[Cell]) -> None:
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO cells "
                "(cell_id,dataset,method,seed,n_nodes,status,batch_size) "
                "VALUES (?,?,?,?,?,'pending',?)",
                [(cell.cell_id, cell.dataset, cell.method, cell.seed,
                  cell.n_nodes, cell.initial_batch_size) for cell in cells],
            )

    def set_running(self, cell_id: str, cfg_path: str, out_dir: str, log_path: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE cells SET status='running', config_path=?, out_dir=?, log_path=?, "
                "started_at=? WHERE cell_id=?",
                (cfg_path, out_dir, log_path, _now(), cell_id),
            )

    def set_completed(self, cell_id: str, duration_s: float) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE cells SET status='completed', finished_at=?, duration_s=?, exit_code=0 "
                "WHERE cell_id=?",
                (_now(), duration_s, cell_id),
            )

    def set_failed(self, cell_id: str, exit_code: int, msg: str, duration_s: float) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE cells SET status='failed', finished_at=?, duration_s=?, "
                "exit_code=?, error_msg=? WHERE cell_id=?",
                (_now(), duration_s, exit_code, msg[:2000], cell_id),
            )

    def set_oom_and_requeue(self, cell_id: str, new_batch: int, msg: str, duration_s: float) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE cells SET status='pending', finished_at=?, duration_s=?, "
                "oom_retried=oom_retried+1, batch_size=?, error_msg=? WHERE cell_id=?",
                (_now(), duration_s, new_batch, msg[:500], cell_id),
            )

    def set_timed_out(self, cell_id: str, duration_s: float) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE cells SET status='failed', finished_at=?, duration_s=?, "
                "exit_code=-999, error_msg='TIMEOUT' WHERE cell_id=?",
                (_now(), duration_s, cell_id),
            )

    def mark_stale_running_pending(self) -> None:
        """On resume, any 'running' cell was interrupted — reset to pending."""
        with self._lock, self._conn() as c:
            c.execute("UPDATE cells SET status='pending' WHERE status='running'")

    # ── reads ──────────────────────────────────────────────────────────────────
    def get(self, cell_id: str) -> dict | None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
        return dict(row) if row else None

    def all_rows(self) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(
                "SELECT * FROM cells ORDER BY n_nodes,dataset,method"
            ).fetchall()]

    def summary(self) -> dict[str, int]:
        with self._conn() as c:
            return {s: n for s, n in c.execute(
                "SELECT status, COUNT(*) FROM cells GROUP BY status"
            ).fetchall()}


# ── Runner ─────────────────────────────────────────────────────────────────────
def _is_oom(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _OOM_KEYWORDS)


def _is_win_crash(code: int) -> bool:
    return code in _WIN_CRASH_CODES or (code < -1 and code != -999 and code != -998)


def run_cell(
    cell: Cell,
    ledger: Ledger,
    dry_run: bool = False,
    cfg_dir: Path | None = None,
    logs_dir: Path | None = None,
) -> str:
    """Execute one cell. Returns 'completed' | 'failed' | 'oom' | 'timeout'."""
    _cfg_dir = cfg_dir or (ROOT / "configs" / "orchestrator")
    _logs_dir = logs_dir or (ORCH_DIR / "logs")

    row = ledger.get(cell.cell_id)
    current_batch: int = row["batch_size"] if row else cell.initial_batch_size

    _cfg_dir.mkdir(parents=True, exist_ok=True)
    _logs_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(cell.config)
    cfg.setdefault("trainer", {})["min_batch_size"] = current_batch

    cfg_path = _cfg_dir / f"{cell.cell_id}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    out_dir_abs = ROOT / cfg["runtime"]["out_dir"]
    out_dir_abs.mkdir(parents=True, exist_ok=True)
    log_path = _logs_dir / f"{cell.cell_id}.log"

    ledger.set_running(cell.cell_id, str(cfg_path), str(out_dir_abs), str(log_path))

    if dry_run:
        print(f"  [DRY_RUN] {cell.cell_id}", flush=True)
        ledger.set_completed(cell.cell_id, 0.01)
        return "completed"

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()

    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(f"=== {cell.cell_id}  started {_now()} ===\n\n")
        fp.flush()
        try:
            proc = subprocess.run(
                [PYTHON, "-u", "-m", "src.online.loop", "--config", str(cfg_path)],
                stdout=fp,
                stderr=fp,
                timeout=cell.timeout_s,
                cwd=str(ROOT),
                env=env,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            dur = time.time() - t0
            fp.write(f"\n=== TIMEOUT after {dur:.0f}s ===\n")
            ledger.set_timed_out(cell.cell_id, dur)
            return "timeout"
        except Exception as exc:
            dur = time.time() - t0
            fp.write(f"\n=== EXCEPTION: {exc} ===\n")
            ledger.set_failed(cell.cell_id, -998, str(exc)[:500], dur)
            return "failed"

    dur = time.time() - t0
    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:]

    if _is_oom(log_tail):
        return "oom"  # caller handles requeue

    if exit_code == 0:
        ledger.set_completed(cell.cell_id, dur)
        return "completed"

    if _is_win_crash(exit_code):
        ledger.set_failed(cell.cell_id, exit_code,
                          f"Windows crash (0x{exit_code & 0xFFFFFFFF:08X})", dur)
        return "failed"

    ledger.set_failed(cell.cell_id, exit_code, log_tail[-500:], dur)
    return "failed"


# ── Results Aggregator ─────────────────────────────────────────────────────────
class ResultsAggregator:
    """Appends final-round metrics to results.csv; regenerates comparison table and plots."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def ingest(self, cell: Cell) -> bool:
        out_dir = ROOT / cell.config["runtime"]["out_dir"]
        rounds_csv = out_dir / "rounds.csv"
        if not rounds_csv.exists():
            return False
        try:
            df = pd.read_csv(rounds_csv)
            if df.empty:
                return False
            last = df.iloc[-1].to_dict()
            row: dict[str, Any] = {
                "cell_id": cell.cell_id,
                "dataset": cell.dataset,
                "method": cell.method,
                "seed": cell.seed,
                "n_nodes": cell.n_nodes,
                "total_rounds": len(df),
                "ingested_at": _now(),
            }
            for col in df.columns:
                if col != "round":
                    row[f"final_{col}"] = last.get(col)
            new_row = pd.DataFrame([row])

            with self._lock:
                if self.csv_path.exists():
                    existing = pd.read_csv(self.csv_path)
                    existing = existing[existing["cell_id"] != cell.cell_id]
                    out = pd.concat([existing, new_row], ignore_index=True)
                else:
                    out = new_row
                out.to_csv(self.csv_path, index=False)
            return True
        except Exception as exc:
            print(f"  [agg] ERROR ingesting {cell.cell_id}: {exc}", flush=True)
            return False

    def regenerate_table(self, sweep_name: str) -> None:
        if not self.csv_path.exists():
            return
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return
            col_map = {
                "coverage": "final_coverage",
                "mrr@1": "final_mrr@1",
                "mrr@3": "final_mrr@3",
                "mrr@5": "final_mrr@5",
                "mrr@10": "final_mrr@10",
                "hits@5": "final_hits@5",
                "hit_rate@1": "final_hit_rate@1",
            }
            lines = [
                f"# Results — {sweep_name}",
                f"\n> 最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
                "| dataset | method | coverage | mrr@1 | mrr@3 | mrr@5 | mrr@10 | hits@5 | hit_rate@1 |",
                "|---------|--------|----------|-------|-------|-------|--------|--------|-----------|",
            ]
            for _, r in df.sort_values(["n_nodes", "dataset", "method"]).iterrows():
                cells_vals = []
                for k, col in col_map.items():
                    v = r.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        cells_vals.append("—")
                    elif k == "coverage":
                        cells_vals.append(f"{float(v):.2%}")
                    else:
                        cells_vals.append(f"{float(v):.4f}")
                lines.append(f"| {r['dataset']} | {r['method']} | {' | '.join(cells_vals)} |")

            tbl_path = ORCH_DIR / sweep_name / f"{sweep_name}_comparison.md"
            tbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"  [agg] ERROR generating table: {exc}", flush=True)

    def regenerate_plots(self, sweep_name: str) -> None:
        """Call plot_algo_sweep.py once per dataset that has at least one rounds.csv."""
        runs_root = ROOT / "results" / "orchestrator" / "runs" / sweep_name
        if not runs_root.exists():
            return
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        for ds_dir in sorted(runs_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            if not any((ds_dir / m / "rounds.csv").exists() for m in ds_dir.iterdir()
                       if (ds_dir / m).is_dir()):
                continue
            out_png = ds_dir / "curves.png"
            try:
                subprocess.run(
                    [PYTHON, "scripts/plot_algo_sweep.py",
                     "--sweep_dir", str(ds_dir),
                     "--out", str(out_png)],
                    cwd=str(ROOT), env=env, timeout=60,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass


# ── Status Reporter ────────────────────────────────────────────────────────────
class StatusReporter(threading.Thread):
    """Daemon thread: writes STATUS.md every STATUS_INTERVAL_S seconds."""

    def __init__(self, ledger: Ledger, sweep_name: str) -> None:
        super().__init__(daemon=True, name="StatusReporter")
        self.ledger = ledger
        self.sweep_name = sweep_name
        self._stop = threading.Event()
        self._t0 = time.time()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(timeout=STATUS_INTERVAL_S):
            self._write()
        self._write()

    def _write(self) -> None:
        try:
            rows = self.ledger.all_rows()
            summary = self.ledger.summary()
            total = len(rows)
            done = summary.get("completed", 0)
            running = summary.get("running", 0)
            failed = summary.get("failed", 0)
            oom = summary.get("oom", 0)
            pending = summary.get("pending", 0)

            elapsed = time.time() - self._t0
            if done > 0:
                eta_s = int(elapsed / done * (total - done))
                eta = str(timedelta(seconds=eta_s))
            else:
                eta = "—"

            bar = "█" * int(20 * done / max(total, 1)) + "░" * (20 - int(20 * done / max(total, 1)))
            ds_rows: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                ds_rows[r["dataset"]].append(r)

            lines = [
                f"# STATUS — {self.sweep_name}",
                f"\n> 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                f"**Progress**: `[{bar}]` {done}/{total}",
                f"- completed: **{done}** | running: {running} | pending: {pending} "
                f"| failed: {failed} | oom: {oom}",
                f"- elapsed: {str(timedelta(seconds=int(elapsed)))} | ETA: {eta}",
                "",
                "## Per-Dataset",
                "",
                "| dataset | n_nodes | done | fail | oom |",
                "|---------|---------|------|------|-----|",
            ]
            for ds in sorted(ds_rows):
                dr = ds_rows[ds]
                n = dr[0]["n_nodes"]
                d = sum(1 for r in dr if r["status"] == "completed")
                f = sum(1 for r in dr if r["status"] == "failed")
                o = sum(1 for r in dr if r["status"] == "oom")
                lines.append(f"| {ds} | {n} | {d}/{len(dr)} | {f} | {o} |")

            lines += [
                "",
                "## All Cells",
                "",
                "| cell_id | status | dur | rc |",
                "|---------|--------|-----|----|",
            ]
            for r in rows:
                dur = f"{r['duration_s']:.0f}s" if r["duration_s"] else "—"
                rc = str(r["exit_code"]) if r["exit_code"] is not None else "—"
                lines.append(f"| `{r['cell_id']}` | {r['status']} | {dur} | {rc} |")

            STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"  [status] write error: {exc}", flush=True)


# ── Sanity Check ───────────────────────────────────────────────────────────────
def _sanity_check_first_round(out_dir: Path, cell_id: str) -> list[str]:
    """Return list of warning strings if first-round metrics look pathological."""
    rounds_csv = out_dir / "rounds.csv"
    if not rounds_csv.exists():
        return [f"rounds.csv missing in {out_dir}"]
    try:
        df = pd.read_csv(rounds_csv)
        if df.empty:
            return ["rounds.csv is empty"]
        row = df.iloc[0]
        warnings: list[str] = []
        # ground_truth oracle legitimately achieves MRR=1.0 — skip this check
        is_oracle = "ground_truth" in cell_id
        for col in ("mrr@1", "mrr@3", "mrr@5", "mrr@10"):
            if col in row and not is_oracle:
                v = float(row[col])
                if v >= 1.0:
                    warnings.append(f"  WARN [{cell_id}] {col}=1.0 — possible data leakage")
        if "coverage" in row:
            cov = float(row["coverage"])
            if cov <= 0.0:
                warnings.append(f"  WARN [{cell_id}] coverage=0.0 — no candidates accepted")
            elif cov >= 1.0:
                warnings.append(f"  WARN [{cell_id}] coverage=1.0 — all users accepted (p_pos=1.0?)")
        return warnings
    except Exception:
        return []


# ── Orchestrator ───────────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self, spec_path: Path, resume: bool = True, dry_run: bool = False,
                 workers: int = 1) -> None:
        self.spec = SweepSpec(spec_path)
        self.resume = resume
        self.dry_run = dry_run
        self.workers = max(1, workers)
        self.cells = self.spec.expand()
        self.ledger = Ledger(_sweep_db(self.spec.name))
        _csv = ORCH_DIR / self.spec.name / "results.csv"
        self.aggregator = ResultsAggregator(_csv)

        self.ledger.register(self.cells)
        if resume:
            self.ledger.mark_stale_running_pending()

        print(
            f"[orchestrator] sweep={self.spec.name}  "
            f"cells={len(self.cells)}  resume={resume}  dry_run={dry_run}  workers={self.workers}",
            flush=True,
        )
        summary = self.ledger.summary()
        print(f"[orchestrator] ledger: {summary}", flush=True)

    @property
    def _cfg_dir(self) -> Path:
        return _sweep_cfg_dir(self.spec.name)

    @property
    def _logs_dir(self) -> Path:
        return _sweep_logs_dir(self.spec.name)

    def run(self) -> None:
        reporter = StatusReporter(self.ledger, self.spec.name)
        reporter.start()
        try:
            self._loop()
        finally:
            reporter.stop()
            self.aggregator.regenerate_table(self.spec.name)
            self.aggregator.regenerate_plots(self.spec.name)
            summary = self.ledger.summary()
            print(f"\n[orchestrator] DONE — {summary}", flush=True)
            self._print_final_table()

    def _run_with_oom(self, cell: Cell) -> str:
        """运行单个 cell，含 OOM 一次重试；返回最终 result 字符串。"""
        result = run_cell(cell, self.ledger, self.dry_run,
                          cfg_dir=self._cfg_dir, logs_dir=self._logs_dir)
        if result == "oom":
            row2 = self.ledger.get(cell.cell_id)
            retried = row2["oom_retried"] if row2 else 0
            old_bs = row2["batch_size"] if row2 else cell.initial_batch_size
            if retried < 1:
                new_bs = max(1, old_bs // 2)
                log_file = self._logs_dir / f"{cell.cell_id}.log"
                log_tail = log_file.read_text(encoding="utf-8", errors="replace")[-500:] \
                           if log_file.exists() else ""
                self.ledger.set_oom_and_requeue(cell.cell_id, new_bs, log_tail,
                                                row2.get("duration_s", 0) if row2 else 0)
                print(f"  [oom-retry] {cell.cell_id}  batch {old_bs}→{new_bs}", flush=True)
                result = run_cell(cell, self.ledger, self.dry_run,
                                  cfg_dir=self._cfg_dir, logs_dir=self._logs_dir)
            else:
                self.ledger.set_failed(cell.cell_id, -2, "OOM after retry", 0)
                result = "failed"
        return result

    def _on_completed(self, cell: Cell) -> None:
        out_dir = ROOT / cell.config["runtime"]["out_dir"]
        for warn in _sanity_check_first_round(out_dir, cell.cell_id):
            print(warn, flush=True)
        self.aggregator.ingest(cell)
        self.aggregator.regenerate_table(self.spec.name)

    def _loop(self) -> None:
        pending = [
            cell for cell in self.cells
            if not (self.resume
                    and (row := self.ledger.get(cell.cell_id))
                    and row["status"] == "completed")
        ]
        skipped = len(self.cells) - len(pending)
        if skipped:
            print(f"  [skip ] {skipped} already-completed cells", flush=True)

        if self.workers <= 1:
            # 串行（原逻辑）
            for cell in pending:
                print(f"\n  [queue] {cell.cell_id}  n_nodes={cell.n_nodes}", flush=True)
                result = self._run_with_oom(cell)
                print(f"  [{result:<9}] {cell.cell_id}  ({_elapsed_str(cell, self.ledger)})",
                      flush=True)
                if result == "completed":
                    self._on_completed(cell)
        else:
            # 并行模式：ThreadPoolExecutor（每个 cell 是独立子进程，Windows 安全）
            total = len(pending)
            done_count = 0
            print(f"  [parallel] {total} cells  workers={self.workers}", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                future_to_cell = {
                    pool.submit(self._run_with_oom, cell): cell for cell in pending
                }
                for fut in concurrent.futures.as_completed(future_to_cell):
                    cell = future_to_cell[fut]
                    done_count += 1
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = "failed"
                        print(f"  [EXC] {cell.cell_id}: {exc}", flush=True)
                    print(f"  [{result:<9}] {cell.cell_id}  "
                          f"({done_count}/{total}, {_elapsed_str(cell, self.ledger)})",
                          flush=True)
                    if result == "completed":
                        self._on_completed(cell)

    def _print_final_table(self) -> None:
        csv_path = ORCH_DIR / self.spec.name / "results.csv"
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return
            cols = ["dataset", "method"]
            for c in ("final_coverage", "final_mrr@5", "final_mrr@10", "final_hits@5"):
                if c in df.columns:
                    cols.append(c)
            print("\n" + df[cols].sort_values(["dataset", "method"]).to_string(index=False),
                  flush=True)
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _elapsed_str(cell: Cell, ledger: Ledger) -> str:
    row = ledger.get(cell.cell_id)
    if not row or not row.get("duration_s"):
        return "?"
    return f"{row['duration_s']:.1f}s"


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="GNN experiment orchestrator")
    parser.add_argument("--spec", required=True, help="Path to sweep_spec.yaml")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip completed cells (default: True)")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--dry_run", action="store_true",
                        help="Register cells but do not execute")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发实验数（默认 1=串行，用 ThreadPoolExecutor）")
    args = parser.parse_args()

    orch = Orchestrator(Path(args.spec), resume=args.resume, dry_run=args.dry_run,
                        workers=args.workers)
    orch.run()


if __name__ == "__main__":
    main()
