import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
SEQ_LEN = 256
N_CALIB = 16
N_EVAL = 16


def build_sequences(seq_len=SEQ_LEN, n_calib=N_CALIB, n_eval=N_EVAL, seed=0):
    """Tokenize wikitext-2-raw-v1 and chunk into fixed-length, non-overlapping
    sequences. Returns (calib_ids, eval_ids): lists of 1D LongTensors, plus
    a sample_id per sequence (index into the combined list) for bookkeeping."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 0])
    ids = tok(text, return_tensors="pt").input_ids[0]

    n_needed = n_calib + n_eval
    total_needed_tokens = n_needed * seq_len
    assert ids.numel() >= total_needed_tokens, f"not enough tokens: {ids.numel()} < {total_needed_tokens}"

    # Skip the very first chunk (often front-matter/headers) for cleaner text.
    offset = seq_len
    chunks = []
    for i in range(n_needed):
        start = offset + i * seq_len
        chunks.append(ids[start:start + seq_len].clone())

    calib = chunks[:n_calib]
    eval_ = chunks[n_calib:n_calib + n_eval]
    return calib, eval_, tok


if __name__ == "__main__":
    calib, eval_, tok = build_sequences()
    print("calib sequences:", len(calib), "eval sequences:", len(eval_))
    print("seq shape:", calib[0].shape, calib[0].dtype)
    torch.save({"calib": calib, "eval": eval_}, "/home/minjeong/project/FERRET/ferret_toy0/traces/_sequences.pt")
    print("saved sequences.")
