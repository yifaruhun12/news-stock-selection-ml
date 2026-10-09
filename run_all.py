from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data() -> list[dict]:
    manifest_path = ROOT / "data" / "DATA_MANIFEST.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    checks = []
    for item in manifest:
        path = ROOT / "data" / item["file"]
        exists = path.is_file()
        size_matches = exists and path.stat().st_size == int(item["bytes"])
        hash_matches = exists and sha256(path) == item["sha256"]
        checks.append(
            {
                "file": item["file"],
                "exists": exists,
                "size_matches": size_matches,
                "sha256_matches": hash_matches,
                "passed": bool(exists and size_matches and hash_matches),
            }
        )
    if not all(item["passed"] for item in checks):
        raise RuntimeError("data manifest verification failed")
    return checks


def run(script: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "code" / script)],
        cwd=ROOT,
        check=True,
    )


def load_summary(folder: str) -> dict:
    path = OUTPUT_DIR / folder / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    data_checks = verify_data()
    run("run_base_v0_5.py")
    run("run_logistic_l2.py")

    base = load_summary("base_v0_5")
    logistic = load_summary("logistic_l2_industry_cap_30pct")
    rows = []
    for payload in (base, logistic):
        summary = payload["summary"]
        rows.append(
            {
                "strategy": payload["strategy"],
                "start": summary["oos_start"],
                "end": summary["oos_end"],
                "annualized_return": summary["strategy_annualized_return"],
                "cumulative_return": summary["strategy_cumulative_return"],
                "sharpe": summary["strategy_sharpe"],
                "max_drawdown": summary["strategy_max_drawdown"],
                "win_rate": summary["realized_trade_win_rate"],
                "entry_trades": summary["entry_trades"],
                "validation_pass": payload["validation"]["all_checks_pass"],
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "final_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status = {
        "status": "passed"
        if all(row["validation_pass"] for row in rows)
        else "failed",
        "data_manifest_pass": all(item["passed"] for item in data_checks),
        "data_files_checked": len(data_checks),
        "strategies": rows,
    }
    (OUTPUT_DIR / "reproduction_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
