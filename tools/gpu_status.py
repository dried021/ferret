#!/usr/bin/env python3
# gpu_status.py
# GPU별로 지금 누가(내 작업 포함) 뭘 몇 시간째 돌리고 있는지 한 번에 보여준다.
#
# 사용법: venv/bin/python gpu_status.py  (또는 python3 gpu_status.py, 외부 의존성 없음)

import getpass
import subprocess


def run_csv(query):
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-{query[0]}={query[1]}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        rows.append([c.strip() for c in line.split(",")])
    return rows


def get_process_info(pid):
    try:
        out = subprocess.check_output(
            ["ps", "-o", "user=,etime=,cmd=", "-p", str(pid)],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            return None
        parts = out.split(None, 2)
        user = parts[0]
        etime = parts[1] if len(parts) > 1 else "?"
        cmd = parts[2] if len(parts) > 2 else "?"
    except subprocess.CalledProcessError:
        return None

    # 다른 사용자 프로세스는 /proc/<pid>/cwd 읽기 권한이 없을 수 있음(ptrace 제한) -> best-effort.
    try:
        cwd = subprocess.check_output(["readlink", "-f", f"/proc/{pid}/cwd"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        cwd = "?"

    return {"user": user, "etime": etime, "cmd": cmd, "cwd": cwd}


def main():
    me = getpass.getuser()

    gpus = run_csv(("gpu", "index,name,memory.used,memory.total,utilization.gpu"))
    uuid_map = {row[1]: row[0] for row in run_csv(("gpu", "index,uuid"))}

    try:
        apps = run_csv(("compute-apps", "pid,used_memory,gpu_uuid"))
    except subprocess.CalledProcessError:
        apps = []

    by_gpu = {}
    for pid, used_mem, gpu_uuid in apps:
        idx = uuid_map.get(gpu_uuid, "?")
        by_gpu.setdefault(idx, []).append((pid, used_mem))

    print(f"{'GPU':<4}{'Mem':<16}{'Util':<7}{'PID':<9}{'User':<10}{'Elapsed':<12}Cmd (cwd)")
    print("-" * 100)

    for idx, name, mem_used, mem_total, util in gpus:
        procs = by_gpu.get(idx, [])
        mem_str = f"{mem_used}/{mem_total}MB"
        if not procs:
            print(f"{idx:<4}{mem_str:<16}{util+'%':<7}{'(free)':<9}")
            continue
        for i, (pid, used_mem) in enumerate(procs):
            info = get_process_info(pid)
            gpu_col = idx if i == 0 else ""
            mem_col = mem_str if i == 0 else ""
            util_col = (util + "%") if i == 0 else ""
            if info is None:
                print(f"{gpu_col:<4}{mem_col:<16}{util_col:<7}{pid:<9}{'(exited?)':<10}")
                continue
            # tag = " <== 나" if info["user"] == me else ""
            cmd_short = info["cmd"]
            if len(cmd_short) > 60:
                cmd_short = cmd_short[:57] + "..."
            print(f"{gpu_col:<4}{mem_col:<16}{util_col:<7}{pid:<9}{info['user']:<10}"
                  f"{info['etime']:<12}{cmd_short}  ({info['cwd']})")

                #   f"{info['etime']:<12}{cmd_short}  ({info['cwd']}){tag}")


if __name__ == "__main__":
    main()
