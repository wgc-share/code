from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = ROOT / "code" / "ablation"
RESULT_ROOT = ROOT / "results" / "ablation" / "batch_temp_input_adaln_10x_logs"

METHOD_SETS = {
    "filtered": [
        ("no_adaln_temp_input_no_ah", "train_ablation_no_adaln_temp_input_no_ah.py"),
        ("temp_input_adaln_no_ah", "train_ablation_temp_input_adaln_no_ah.py"),
    ],
    "extrapolation": [
        ("no_adaln_temp_input_no_ah", "train_ablation_temp_extrap_no_adaln_temp_input_no_ah.py"),
        ("temp_input_adaln_no_ah", "train_ablation_temp_extrap_temp_input_adaln_no_ah.py"),
    ],
}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def stream(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8-sig") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Run the temp-input vs temp-input+AdaLN comparison for multiple rounds.")
    parser.add_argument("--scope", choices=["filtered", "extrapolation", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--show-tqdm", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")

    run_name = args.run_name or f"temp_input_adaln_{args.scope}_{args.rounds}x_{datetime.now().strftime('%m%d_%H%M%S')}"
    scopes = ["filtered", "extrapolation"] if args.scope == "both" else [args.scope]
    print("=== Temp-input vs Temp-input+AdaLN batch plan ===")
    print(f"SCOPES   : {', '.join(scopes)}")
    print(f"ROUNDS   : {args.rounds}")
    print(f"SEEDS    : {args.seed_start}..{args.seed_start + args.rounds - 1}")

    for scope in scopes:
        methods = METHOD_SETS[scope]
        run_root = RESULT_ROOT / run_name / scope
        run_root.mkdir(parents=True, exist_ok=True)

        plan_rows = []
        step = 0
        for round_idx in range(1, args.rounds + 1):
            seed = args.seed_start + round_idx - 1
            for method, script in methods:
                step += 1
                plan_rows.append(
                    {
                        "step": step,
                        "round": round_idx,
                        "scope": scope,
                        "method": method,
                        "seed": seed,
                        "script": str(ABLATION_DIR / script),
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "val_batch_size": args.val_batch_size,
                        "soc_batch_size": args.soc_batch_size,
                        "disable_tqdm": not args.show_tqdm,
                    }
                )

        plan_path = run_root / f"{scope}_plan.csv"
        status_path = run_root / f"{scope}_status.csv"
        write_csv(plan_path, plan_rows)

        print("\n" + "-" * 100)
        print(f"SCOPE    : {scope}")
        print(f"RUN_ROOT : {run_root}")
        print(f"PLAN     : {plan_path}")
        print(f"STATUS   : {status_path}")
        print(f"METHODS  : {', '.join(m for m, _ in methods)}")
        print(f"TOTAL    : {len(plan_rows)} training runs")

        for row in plan_rows:
            method = row["method"]
            round_idx = int(row["round"])
            seed = int(row["seed"])
            script_path = Path(row["script"])
            log_path = run_root / "logs" / f"round{round_idx:02d}_{method}_seed{seed}.log"
            cmd = [
                sys.executable,
                "-u",
                str(script_path),
                "--mode",
                "train",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--val-batch-size",
                str(args.val_batch_size),
                "--soc-batch-size",
                str(args.soc_batch_size),
                "--seed",
                str(seed),
            ]
            if args.cpu:
                cmd.append("--cpu")
            if not args.show_tqdm:
                cmd.append("--disable-tqdm")

            started_at = datetime.now().isoformat(timespec="seconds")
            print("\n" + "=" * 100)
            print(f"Step {row['step']}/{len(plan_rows)} | round={round_idx} | method={method} | seed={seed}")
            print(f"LOG: {log_path}")
            append_csv(
                status_path,
                {
                    **row,
                    "status": "started",
                    "started_at": started_at,
                    "finished_at": "",
                    "return_code": "",
                    "log_path": str(log_path),
                },
            )

            return_code = stream(cmd, log_path)
            finished_at = datetime.now().isoformat(timespec="seconds")
            status = "complete" if return_code == 0 else "failed"
            append_csv(
                status_path,
                {
                    **row,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "return_code": return_code,
                    "log_path": str(log_path),
                },
            )
            print(f"Step finished | status={status} | return_code={return_code}")
            if return_code != 0 and args.stop_on_error:
                raise RuntimeError(f"Failed at round={round_idx}, method={method}. See {log_path}")

    print("\nAll scheduled comparison runs finished.")


if __name__ == "__main__":
    main()
