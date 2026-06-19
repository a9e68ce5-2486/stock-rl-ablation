"""Save / load eval results to CSV (for quick compare) + JSON (full detail)."""
from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "runs").mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "runs.csv"
CSV_FIELDS = [
    "timestamp", "run_id", "config", "period",
    "start", "end", "steps",
    "cum_return", "sharpe", "max_dd", "avg_turnover",
    "spy_return", "spy_sharpe",
    "max_avg_weight", "cash_avg_weight",
    "model_path",
]


def append_to_csv(row: dict):
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        # Only write the recognized fields, in order
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def save_run_json(run_id: str, payload: dict):
    json_path = RESULTS_DIR / "runs" / f"{run_id}.json"
    # If file exists, merge (e.g., adding OOS to existing IS)
    existing = {}
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)

    # Merge by period
    if "results" in existing and "results" in payload:
        existing["results"].update(payload["results"])
        payload["results"] = existing["results"]
    elif "results" in existing and "results" not in payload:
        payload["results"] = existing["results"]

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_eval_result(
    run_id: str,
    config_name: str,
    period_label: str,
    period_start: str,
    period_end: str,
    metrics: dict,
    avg_weights: dict,
    model_path: str,
    full_returns: list[float] | None = None,
):
    # CSV row (one per period)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "config": config_name,
        "period": period_label,
        "start": period_start,
        "end": period_end,
        "steps": metrics["steps"],
        "cum_return": f"{metrics['cum_return']:.6f}",
        "sharpe": f"{metrics['sharpe']:.6f}",
        "max_dd": f"{metrics['max_dd']:.6f}",
        "avg_turnover": f"{metrics['avg_turnover']:.6f}",
        "spy_return": f"{metrics['spy_return']:.6f}",
        "spy_sharpe": f"{metrics['spy_sharpe']:.6f}",
        "max_avg_weight": f"{max(avg_weights.values()):.4f}",
        "cash_avg_weight": f"{avg_weights.get('CASH', 0):.4f}",
        "model_path": model_path,
    }
    append_to_csv(row)

    # JSON full detail
    payload = {
        "run_id": run_id,
        "config_name": config_name,
        "model_path": model_path,
        "results": {
            period_label: {
                "start": period_start,
                "end": period_end,
                "metrics": metrics,
                "avg_weights": avg_weights,
                "daily_returns": full_returns if full_returns else None,
            }
        },
    }
    save_run_json(run_id, payload)


def load_runs_table():
    """Return list of dicts representing all eval rows so far."""
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def print_compare_table():
    """Pretty-print a compact comparison of all runs so far."""
    rows = load_runs_table()
    if not rows:
        print("(no runs yet — train + eval a model first)")
        return

    cols = ["run_id", "config", "period", "sharpe", "cum_return",
            "max_dd", "avg_turnover", "max_avg_weight", "cash_avg_weight"]
    widths = {c: max(len(c), max(len(r.get(c, "")) for r in rows)) for c in cols}

    def fmt_row(r):
        return "  ".join(str(r.get(c, ""))[: widths[c]].ljust(widths[c]) for c in cols)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("─" * len(header))
    for r in rows:
        print(fmt_row(r))


if __name__ == "__main__":
    print_compare_table()
