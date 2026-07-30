#!/usr/bin/env python3
"""Consolidated GPU/process status across local + pod1/pod2/theta5090.

Usage:
    python otter/scripts/check_gpu_status.py
    python otter/scripts/check_gpu_status.py --hosts local,theta5090
    python otter/scripts/check_gpu_status.py --log-lines 8
"""
import argparse
import subprocess

HOSTS = ["local", "pod1", "pod2", "theta5090"]

GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
    "--format=csv,noheader"
)
PROC_QUERY = (
    "ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu "
    "| grep -E 'phase1_|belebele|xnli_eval' | grep -v grep | head -8"
)


def run_local(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr).strip()


def run_remote(host: str, cmd: str, timeout: int = 25) -> str:
    r = subprocess.run(
        ["ssh", host, cmd], capture_output=True, text=True, timeout=timeout
    )
    return (r.stdout + r.stderr).strip()


def report(host: str, log_lines: int) -> None:
    print(f"\n=== {host} ===")
    remote_cmd = (
        f"{GPU_QUERY}; echo '---procs---'; {PROC_QUERY}; "
        f"echo '---latest log---'; "
        f"f=$(ls -t /root/*.log 2>/dev/null | head -1); "
        f"if [ -n \"$f\" ]; then echo \"$f\"; tail -n {log_lines} \"$f\"; "
        f"else echo '(no /root/*.log found)'; fi"
    )
    try:
        if host == "local":
            print(run_local(GPU_QUERY))
            print("---procs---")
            print(run_local(PROC_QUERY) or "(none running)")
        else:
            print(run_remote(host, remote_cmd))
    except subprocess.TimeoutExpired:
        print(f"[unreachable / timed out: {host}]")
    except Exception as e:
        print(f"[error: {host}: {e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default=",".join(HOSTS),
                     help="comma-separated subset of: " + ",".join(HOSTS))
    ap.add_argument("--log-lines", type=int, default=5)
    args = ap.parse_args()

    for host in args.hosts.split(","):
        report(host.strip(), args.log_lines)


if __name__ == "__main__":
    main()
