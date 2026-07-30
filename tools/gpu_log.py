#!/usr/bin/env python3
# gpu_log.py
# 일정 간격으로 GPU별 온도/사용률/점유 여부를 측정해서 CSV 또는 JSON Lines로 누적 기록한다.
#
# 사용법:
#   python3 tools/gpu_log.py                          # 30분 간격, tools/logs/gpu_log.csv에 기록
#   python3 tools/gpu_log.py --interval-minutes 10     # 간격 변경
#   python3 tools/gpu_log.py --format json             # tools/logs/gpu_log.jsonl로 기록
#   python3 tools/gpu_log.py --once                    # 한 번만 측정하고 종료 (테스트용)
#
# 터미널을 닫아도 계속 돌게 하려면:
#   nohup python3 tools/gpu_log.py > tools/logs/gpu_log.out 2>&1 &
#
# 돌고 있는지 확인 / 끄기:
#   pgrep -af gpu_log.py       # PID랑 실행 커맨드 확인
#   kill <PID>                 # 끄기
#
# 나중에 데이터 분석할 때 (pandas):
#   import pandas as pd
#   df = pd.read_csv("tools/logs/gpu_log.csv")                 # --format csv (기본)
#   df = pd.read_json("tools/logs/gpu_log.jsonl", lines=True)  # --format json

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

FIELDNAMES = [
    "timestamp", "gpu_index", "gpu_name", "temperature_c",
    "utilization_percent", "memory_used_mib", "memory_total_mib",
    "process_count", "in_use",
]


def run_csv(query_type, fields):
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-{query_type}={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        rows.append([c.strip() for c in line.split(",")])
    return rows


def sample():
    now = datetime.datetime.now().isoformat(timespec="seconds")
    gpus = run_csv("gpu", "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total")
    uuid_map = {row[1]: row[0] for row in run_csv("gpu", "index,uuid")}

    try:
        apps = run_csv("compute-apps", "pid,gpu_uuid")
    except subprocess.CalledProcessError:
        apps = []

    proc_count = {}
    for pid, gpu_uuid in apps:
        idx = uuid_map.get(gpu_uuid, "?")
        proc_count[idx] = proc_count.get(idx, 0) + 1

    records = []
    for idx, name, temp, util, mem_used, mem_total in gpus:
        n_proc = proc_count.get(idx, 0)
        records.append({
            "timestamp": now,
            "gpu_index": int(idx),
            "gpu_name": name,
            "temperature_c": int(temp),
            "utilization_percent": int(util),
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
            "process_count": n_proc,
            "in_use": n_proc > 0 or int(util) > 0,
        })
    return records


def write_csv(records, path):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(records)


def write_jsonl(records, path):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="GPU 온도/사용 여부를 주기적으로 기록한다.")
    parser.add_argument("--interval-minutes", type=float, default=30)
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--out", default=None, help="출력 경로 (기본: tools/logs/gpu_log.csv 또는 .jsonl)")
    parser.add_argument("--once", action="store_true", help="한 번만 측정하고 종료 (테스트용)")
    args = parser.parse_args()

    ext = "csv" if args.format == "csv" else "jsonl"
    out_path = args.out or os.path.join(os.path.dirname(__file__), "logs", f"gpu_log.{ext}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write = write_csv if args.format == "csv" else write_jsonl

    print(f"GPU 로깅 시작 -> {out_path} ({args.interval_minutes}분 간격, Ctrl+C로 종료)")
    try:
        while True:
            try:
                records = sample()
                write(records, out_path)
                print(f"[{records[0]['timestamp']}] {len(records)}개 GPU 기록 완료", flush=True)
            except Exception as e:
                print(f"측정 실패: {e}", file=sys.stderr, flush=True)

            if args.once:
                break
            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
