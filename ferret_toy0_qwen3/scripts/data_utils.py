"""Build calibration/eval sequences for Qwen3-30B-A3B from wikitext-2-raw-v1.

split_by_document=true (config): each wikitext article (a contiguous run of
non-empty lines between blank-line boundaries) is tokenized independently and
assigned wholly to either calib or eval, so no chunk straddles the
calibration/eval boundary and no chunk mixes tokens from two articles.
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
SEQ_LEN = 256
N_CALIB = 16
N_EVAL = 16
OUT_PATH = "/home/minjeong/project/FERRET/ferret_toy0_qwen3/traces/_sequences.pt"


def _iter_documents(raw_lines):
    """wikitext-2-raw-v1 marks article boundaries with blank lines and
    ' = Title = ' headers. Group consecutive non-empty lines into one
    document each."""
    doc_lines = []
    for line in raw_lines:
        if line.strip() == "":
            if doc_lines:
                yield "\n".join(doc_lines)
                doc_lines = []
        else:
            doc_lines.append(line)
    if doc_lines:
        yield "\n".join(doc_lines)


def build_sequences(seq_len=SEQ_LEN, n_calib=N_CALIB, n_eval=N_EVAL, seed=0):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    n_needed = n_calib + n_eval
    chunks = []
    for doc in _iter_documents(ds["text"]):
        if len(chunks) >= n_needed:
            break
        ids = tok(doc, return_tensors="pt").input_ids[0]
        if ids.numel() < seq_len:
            continue
        chunks.append(ids[:seq_len].clone())

    assert len(chunks) >= n_needed, (
        f"not enough documents with >= {seq_len} tokens: found {len(chunks)}, need {n_needed}"
    )

    calib = chunks[:n_calib]
    eval_ = chunks[n_calib:n_calib + n_eval]
    return calib, eval_, tok


if __name__ == "__main__":
    calib, eval_, tok = build_sequences()
    print("calib sequences:", len(calib), "eval sequences:", len(eval_))
    print("seq shape:", calib[0].shape, calib[0].dtype)
    torch.save({"calib": calib, "eval": eval_}, OUT_PATH)
    print("saved sequences to", OUT_PATH)
