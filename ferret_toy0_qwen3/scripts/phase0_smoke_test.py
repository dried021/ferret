"""Phase 0: confirm Qwen3-30B-A3B forwards stably on 2x RTX 3090 (physical GPU 2,3) in INT8.

Run with:
    CUDA_VISIBLE_DEVICES=2,3 python3 phase0_smoke_test.py

Ramps seq_len 64 -> 128 -> 256 per methodology section 6, recording per-GPU peak
memory, forward time, and any OOM/cross-device errors. Does not touch routing or
expert weights -- that's Phase 1+.
"""
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
SEQ_LENS = [64, 128, 256]
BATCH_SIZE = 1
PEAK_RESERVED_BUDGET_GB = 22.5

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "smoke_test.log"
RESULT_PATH = ROOT / "results" / "phase0_smoke_test.json"
DEVICE_MAP_CONFIG = ROOT / "configs" / "device_map.json"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def gpu_mem_snapshot():
    snap = {}
    for i in range(torch.cuda.device_count()):
        snap[f"cuda:{i}"] = {
            "allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(i) / 1e9, 2),
        }
    return snap


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log(f"visible devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
    assert torch.cuda.device_count() == 2, (
        f"expected 2 visible GPUs (physical 2,3 via CUDA_VISIBLE_DEVICES), "
        f"found {torch.cuda.device_count()}"
    )

    results = {"model": MODEL_NAME, "runs": []}

    log(f"loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log("loading model in INT8 across cuda:0,1 (balanced device_map)")
    with open(DEVICE_MAP_CONFIG) as f:
        dm_cfg = json.load(f)
    max_memory = {(int(k) if k.isdigit() else k): v for k, v in dm_cfg["max_memory"].items()}
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=dm_cfg["strategy"],
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    load_time = time.time() - t0
    log(f"model loaded in {load_time:.1f}s")
    log(f"post-load memory: {json.dumps(gpu_mem_snapshot())}")

    vocab_size = model.config.vocab_size

    for seq_len in SEQ_LENS:
        log(f"--- seq_len={seq_len} ---")
        torch.cuda.reset_peak_memory_stats()
        input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, seq_len), device="cuda:0")
        try:
            t0 = time.time()
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    output_router_logits=False,
                )
            torch.cuda.synchronize()
            fwd_time = time.time() - t0
            mem = gpu_mem_snapshot()
            ok = all(v["peak_reserved_gb"] < PEAK_RESERVED_BUDGET_GB for v in mem.values())
            log(f"forward ok, time={fwd_time:.3f}s, logits shape={tuple(out.logits.shape)}")
            log(f"memory: {json.dumps(mem)}  within_budget={ok}")
            results["runs"].append({
                "seq_len": seq_len,
                "status": "ok",
                "forward_time_s": round(fwd_time, 3),
                "memory": mem,
                "within_budget": ok,
            })
            del out
        except torch.cuda.OutOfMemoryError as e:
            log(f"OOM at seq_len={seq_len}: {e}")
            results["runs"].append({"seq_len": seq_len, "status": "oom", "error": str(e)})
            torch.cuda.empty_cache()
            break
        except Exception as e:
            log(f"error at seq_len={seq_len}: {type(e).__name__}: {e}")
            results["runs"].append({"seq_len": seq_len, "status": "error", "error": str(e)})
            break
        finally:
            del input_ids
            gc.collect()
            torch.cuda.empty_cache()

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"results written to {RESULT_PATH}")

    all_ok = all(r["status"] == "ok" and r["within_budget"] for r in results["runs"])
    log(f"PHASE 0 {'PASS' if all_ok and len(results['runs']) == len(SEQ_LENS) else 'INCOMPLETE/FAIL'}")


if __name__ == "__main__":
    main()
