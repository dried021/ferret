"""Central config for otter's Toy0 (see ../README.md).

Reuses the Qwen3-30B-A3B setup already verified in ferret_kr and
ferret_toy0_qwen3_ver1 on this host -- same batched-3D-weight expert module
shape, same device_map probing strategy. See scripts/moe_hooks.py for the
verified module shape assertion.
"""
import getpass
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
HF_HOME = "/mnt/HDD/minjeong/hf_cache"
os.environ.setdefault("HF_HOME", HF_HOME)

DEVICE_MAP_CONFIG = json.loads((PROJECT_ROOT / "configs" / "device_map.json").read_text())
LANGUAGES_YAML = PROJECT_ROOT / "data" / "languages.yaml"


def _safe_gpu_indices():
    """Checked at every run, never hardcoded: queries nvidia-smi for which
    GPU each running compute process sits on, then ps for who owns that
    process. Any GPU with a process owned by someone other than the current
    user is excluded outright -- this is a shared 4x3090 host and another
    user's job must never be touched, regardless of how much free memory
    _probe_max_memory() thinks is available on it (incident 2026-07-23: a
    stale static cuda_visible_devices list let a run reach into a GPU held
    by another user's active job)."""
    me = getpass.getuser()

    idx_out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    uuid_to_index = {}
    for line in idx_out.strip().splitlines():
        idx, uuid = (x.strip() for x in line.split(","))
        uuid_to_index[uuid] = idx
    all_indices = sorted(uuid_to_index.values(), key=int)

    proc_out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout

    excluded = {}
    for line in proc_out.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = (x.strip() for x in line.split(","))
        gpu_idx = uuid_to_index.get(uuid)
        if gpu_idx is None or gpu_idx in excluded:
            continue
        owner = subprocess.run(
            ["ps", "-o", "user=", "-p", pid],
            capture_output=True, text=True,
        ).stdout.strip()
        if owner and owner != me:
            excluded[gpu_idx] = (owner, pid)

    for gpu_idx, (owner, pid) in excluded.items():
        print(f"[config] GPU {gpu_idx}: process owned by '{owner}' (pid {pid}), not '{me}' -- excluding entirely")

    safe = [i for i in all_indices if i not in excluded]
    if not safe:
        raise RuntimeError(
            "no GPU is free of other users' processes -- refusing to run. "
            "Check `nvidia-smi` manually before retrying."
        )
    print(f"[config] GPUs safe to use this run (no other-user process on them): {safe}")
    return safe


# Must be set before any torch.cuda call -- see _safe_gpu_indices docstring.
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_safe_gpu_indices())

DATA_DIR = PROJECT_ROOT / "data"
FACTORS_DIR = PROJECT_ROOT / "factors"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
for d in (DATA_DIR, FACTORS_DIR, RESULTS_DIR, LOGS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Toy0 only needs a cheap early/mid/late signal, not the full 6-layer sweep
# ferret_kr uses -- subset of ferret_toy0_qwen3_ver1's layer_selection.
TOY_LAYER_INDICES = [8, 22, 38]

EVAL_BATCH_SIZE = 15  # sentences per batched forward

ROUTING_STATS_JSON = RESULTS_DIR / "toy0_routing_fisher_stats.json"
GATE_SUMMARY_JSON = RESULTS_DIR / "toy0_gate_summary.json"
DIVERGENCE_CSV = RESULTS_DIR / "toy0_condition_divergence.csv"
FISHER_CORR_CSV = RESULTS_DIR / "toy0_fisher_rank_correlation.csv"

# --- Phase 0.5 (and later, same-shape reruns like the layer-locality check)
# share one generic loader/path scheme, keyed by an output `prefix` so
# multiple runs of the same probe (different yaml config, different layer
# set) don't clobber each other's results. ---
def load_spec(yaml_path):
    """Returns a spec dict straight from the given yaml (seeds, layer_indices,
    total_token_budget, conditions, languages) -- same shape as
    data/phase0_5_config.yaml. Generic so 01b/02b/03b can be pointed at any
    yaml built to this shape (e.g. data/layer_locality_config.yaml) via
    --config, instead of forking a new script per follow-up probe."""
    import yaml
    return yaml.safe_load(Path(yaml_path).read_text())


def stats_json(prefix, seed):
    return RESULTS_DIR / f"{prefix}_stats_seed{seed}.json"


def gate_json(prefix):
    return RESULTS_DIR / f"{prefix}_gate_summary.json"


def seed_layer_csv(prefix):
    return RESULTS_DIR / f"{prefix}_seed_layer_table.csv"


# --- Phase 0.5: reproducibility check (see 00_docs/02_Toy_실험.md) ---
PHASE0_5_YAML = DATA_DIR / "phase0_5_config.yaml"


def phase0_5_stats_json(seed):
    return stats_json("phase0_5", seed)


PHASE0_5_GATE_JSON = gate_json("phase0_5")
PHASE0_5_SEED_LAYER_CSV = seed_layer_csv("phase0_5")


def load_phase0_5_spec():
    """Returns the Phase 0.5 spec dict straight from data/phase0_5_config.yaml
    (seeds, layer_indices, total_token_budget, conditions, languages)."""
    return load_spec(PHASE0_5_YAML)


# --- Layer-locality follow-up (00_docs/02_Toy_실험.md "0) 레이어 38 국소성
# 검증"): same probe, denser late-layer sweep, no new language/model/analysis. ---
LAYER_LOCALITY_YAML = DATA_DIR / "layer_locality_config.yaml"


def load_conditions():
    """Returns (source, split, column_prefix, conditions) where conditions is
    {condition_id: [(lang_code, n_sentences, offset), ...]}, straight from
    data/languages.yaml -- see that file's comments for why offsets are
    disjoint across conditions. This is the Toy0 (2026-07-23) config; Phase
    0.5 uses load_phase0_5_spec() instead (token-budget, not fixed sentence
    counts)."""
    import yaml
    spec = yaml.safe_load(LANGUAGES_YAML.read_text())
    conditions = {
        cond_id: [tuple(row) for row in rows]
        for cond_id, rows in spec["conditions"].items()
    }
    return spec["source"], spec["split"], spec["column_prefix"], conditions


def _parse_size(s):
    s = str(s).strip()
    if s.endswith("GiB"):
        return int(float(s[:-3]) * 1024 ** 3)
    if s.endswith("MiB"):
        return int(float(s[:-3]) * 1024 ** 2)
    return int(s)


def _probe_max_memory(safety_margin_gib=6.0, min_usable_gib=4.0):
    """Caps each visible GPU's max_memory at min(configured cap, actually-free
    memory right now - safety margin) -- see ferret_kr/scripts/00_config.py,
    copied verbatim since this is the same shared host."""
    import torch

    configured = DEVICE_MAP_CONFIG["max_memory"]
    safety_bytes = int(safety_margin_gib * 1024 ** 3)
    min_usable_bytes = int(min_usable_gib * 1024 ** 3)

    max_memory = {}
    for i in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        usable = max(free_bytes - safety_bytes, 0)
        cap = _parse_size(configured.get(str(i), "0GiB"))
        chosen = min(usable, cap)
        if chosen < min_usable_bytes:
            print(f"[config] GPU {i}: only {free_bytes / 1024**3:.1f}GiB free, excluding from this run")
            continue
        max_memory[i] = int(chosen)
        print(f"[config] GPU {i}: {free_bytes / 1024**3:.1f}GiB free -> using up to {chosen / 1024**3:.1f}GiB")

    if "cpu" in configured:
        max_memory["cpu"] = configured["cpu"]
    return max_memory


def load_model_and_tokenizer(layer_indices=None):
    """Loads Qwen3-30B-A3B in bf16 across whatever GPU memory is actually
    free right now, and asserts the MoE block shape matches what
    moe_hooks.py expects, for every layer in `layer_indices` (defaults to
    TOY_LAYER_INDICES -- Phase 0.5 passes its own wider layer list)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    if layer_indices is None:
        layer_indices = TOY_LAYER_INDICES

    max_memory = _probe_max_memory()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE_MAP_CONFIG["strategy"],
        max_memory=max_memory,
    )
    model.eval()

    for layer_id in layer_indices:
        mlp = model.model.layers[layer_id].mlp
        if not isinstance(mlp, Qwen3MoeSparseMoeBlock):
            raise RuntimeError(
                f"layer {layer_id}: expected Qwen3MoeSparseMoeBlock, got "
                f"{type(mlp).__name__}. Model/transformers structure has "
                f"drifted -- re-inspect print(model) and update moe_hooks.py."
            )
        experts = mlp.experts
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            raise RuntimeError(
                f"layer {layer_id}: experts module missing expected batched "
                f"gate_up_proj/down_proj tensors -- structure has drifted."
            )

    return model, tokenizer
