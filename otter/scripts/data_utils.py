"""FLORES-200 sentence loading + per-sentence/per-corpus PPL helpers, copied
unchanged from ferret_kr/scripts/data_utils.py (same dataset source, same
model). Toy0 (01_calibration_stats.py) only uses load_flores_sentences here --
the PPL helpers are kept for parity and for a later phase that needs a
held-out LM-likelihood check per calibration condition.

Uses israel/flores-parallel (ungated community re-upload of FLORES-200
devtest) instead of the gated facebook/flores -- see data/languages.yaml for
the rationale and how to switch back once access is granted.
"""
import torch
import torch.nn.functional as F


def load_flores_sentences(source, split, column_prefix, lang_code, n, offset=0):
    """Returns the first `n` sentences (from `offset`) for `lang_code`, in
    the dataset's natural devtest order -- deterministic across scripts/runs
    so different calibration conditions can be given disjoint offsets to
    avoid overlap (see 00_config.py CONDITIONS)."""
    from datasets import load_dataset

    ds = load_dataset(source, split=split)
    col = f"{column_prefix}{lang_code}"
    if col not in ds.column_names:
        raise KeyError(f"{col!r} not found in {source} ({split}); available columns include: "
                        f"{[c for c in ds.column_names if c.startswith(column_prefix)][:10]}...")
    sentences = ds[col][offset:offset + n]
    if len(sentences) < n:
        raise ValueError(f"{source} only has {len(sentences)} sentences for {lang_code} from offset {offset}, need {n}")
    return sentences


def load_full_column(source, split, column_prefix, lang_code):
    """Returns every sentence for `lang_code` (all ~1012 FLORES-200 devtest
    rows) -- the pool that Phase 0.5's token-budget sampling draws seed-
    shuffled, disjoint chunks from (see token_budget_chunks)."""
    from datasets import load_dataset

    ds = load_dataset(source, split=split)
    col = f"{column_prefix}{lang_code}"
    if col not in ds.column_names:
        raise KeyError(f"{col!r} not found in {source} ({split}); available columns include: "
                        f"{[c for c in ds.column_names if c.startswith(column_prefix)][:10]}...")
    return list(ds[col])


def token_budget_chunks(tokenizer, pool, seed, budgets):
    """Shuffles `pool` (a language's full sentence list) with a seed-derived
    RNG, then walks the shuffled order once, carving out consecutive,
    non-overlapping chunks -- one per entry in `budgets` -- each stopped as
    soon as its cumulative tokenizer token count reaches that budget.

    This is what makes english_a/english_b/balanced's EN share disjoint
    *within one seed* (three budgets requested from the same English pool),
    while still being controlled by token count rather than sentence count
    (Phase 0.5's main fix over Toy0 -- see data/phase0_5_config.yaml).
    Different seeds reshuffle independently, so chunks *may* overlap across
    seeds -- that's fine, each seed is an independent replicate.
    """
    import random

    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    chunks = []
    pos = 0
    for budget in budgets:
        chunk = []
        total = 0
        while total < budget:
            if pos >= len(order):
                raise ValueError(
                    f"language pool exhausted (size {len(pool)}) before reaching "
                    f"token budget {budget} (accumulated {total} tokens, "
                    f"{len(chunk)} sentences) -- raise the pool size or lower the budget."
                )
            sent = pool[order[pos]]
            pos += 1
            n_tok = len(tokenizer(sent)["input_ids"])
            chunk.append(sent)
            total += n_tok
        chunks.append(chunk)
    return chunks


@torch.no_grad()
def batch_sentence_nll(model, tokenizer, sentences, device=None):
    """Runs one padded batched forward over `sentences`, returns per-sentence
    (nll_sum, n_tokens) lists computed with standard shifted causal-LM loss,
    ignoring pad positions."""
    if device is None:
        device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    enc = tokenizer(list(sentences), return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    label_mask = attention_mask[:, 1:].bool()

    token_nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    token_nll = token_nll.masked_fill(~label_mask, 0.0)

    nll_sums = token_nll.sum(dim=1).tolist()
    n_tokens = label_mask.sum(dim=1).tolist()
    return nll_sums, n_tokens


def evaluate_sentences(model, tokenizer, sentences, batch_size):
    """Chunks `sentences` into batches of `batch_size` and concatenates
    per-sentence (nll_sum, n_tokens) across chunks -- equivalent to one giant
    batch, just bounded peak memory."""
    nll_sums, n_tokens = [], []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        chunk_nll, chunk_tok = batch_sentence_nll(model, tokenizer, chunk)
        nll_sums.extend(chunk_nll)
        n_tokens.extend(chunk_tok)
    return nll_sums, n_tokens


def corpus_ppl(nll_sums, n_tokens_list):
    total_nll = sum(nll_sums)
    total_tokens = sum(n_tokens_list)
    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))
