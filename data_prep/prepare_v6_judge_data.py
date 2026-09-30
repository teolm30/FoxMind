
from pathlib import Path
import json, random
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).parent
BASE = ROOT / "checkpoints" / "v5_4h_balanced" / "final_hf"
OUT = ROOT / "data_v6_judges"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260918
LETTERS = "ABCDEFGH"
tok = PreTrainedTokenizerFast.from_pretrained(BASE)

def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def fmt_hella(r):
    endings = list(r["endings"])
    options = "\n".join(f"{LETTERS[i]}) {x.strip()}" for i, x in enumerate(endings))
    prompt = f"Choose the most plausible continuation.\nContext: {r['ctx'].strip()}\n{options}"
    return {"prompt": prompt, "answer": LETTERS[int(r["label"])], "source": "hellaswag_train"}

def fmt_arc(r):
    labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
    answer_key = str(r["answerKey"])
    idx = labels.index(answer_key)
    options = "\n".join(f"{LETTERS[i]}) {t.strip()}" for i, t in enumerate(texts))
    prompt = f"Choose the correct answer.\nQuestion: {r['question'].strip()}\n{options}"
    return {"prompt": prompt, "answer": LETTERS[idx], "source": "arc_easy_train"}

def fmt_piqa(r):
    goal = r.get("goal", r.get("question", "")).strip()
    a = r.get("sol1", r.get("choice_a", "")).strip()
    b = r.get("sol2", r.get("choice_b", "")).strip()
    label = int(r.get("label", r.get("answer", 0)))
    prompt = f"Choose the physically sensible solution.\nGoal: {goal}\nA) {a}\nB) {b}"
    return {"prompt": prompt, "answer": "A" if label == 0 else "B", "source": "piqa_train"}

def fmt_wino(r):
    sentence = r["sentence"].strip()
    a, b = r["option1"].strip(), r["option2"].strip()
    answer = "A" if str(r["answer"]) == "1" else "B"
    prompt = f"Choose the option that best fills the blank.\nSentence: {sentence}\nA) {a}\nB) {b}"
    return {"prompt": prompt, "answer": answer, "source": "winogrande_train"}

def build_sft():
    specs = [
        ("Rowan/hellaswag", None, fmt_hella),
        ("allenai/ai2_arc", "ARC-Easy", fmt_arc),
        ("lighteval/piqa", None, fmt_piqa),
        ("allenai/winogrande", "winogrande_xl", fmt_wino),
    ]
    train_rows, holdout_rows = [], []
    rng = random.Random(SEED)
    source_counts = {}
    for name, config, formatter in specs:
        ds = load_dataset(name, config, split="train")
        rows = [formatter(r) for r in ds]
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * 0.03))
        holdout_rows.extend(rows[:cut])
        train_rows.extend(rows[cut:])
        source_counts[name] = {"train": len(rows) - cut, "holdout": cut}
    rng.shuffle(train_rows)
    rng.shuffle(holdout_rows)
    save_jsonl(OUT / "judge_train.jsonl", train_rows)
    save_jsonl(OUT / "judge_holdout.jsonl", holdout_rows)
    return source_counts, len(train_rows), len(holdout_rows)
def build_wikitext(max_train_tokens=35_000_000, max_holdout_tokens=1_000_000):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)
    train_path = OUT / "wikitext_train.bin"
    holdout_path = OUT / "wikitext_holdout.bin"
    train_tokens = 0
    holdout_tokens = 0
    with train_path.open("wb") as ft, holdout_path.open("wb") as fh:
        for i, row in enumerate(ds):
            text = row["text"].strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
            arr = np.asarray(ids, dtype=np.uint16)
            if i % 20 == 0 and holdout_tokens < max_holdout_tokens:
                arr.tofile(fh)
                holdout_tokens += len(arr)
            elif train_tokens < max_train_tokens:
                arr.tofile(ft)
                train_tokens += len(arr)
            if train_tokens >= max_train_tokens and holdout_tokens >= max_holdout_tokens:
                break
    return train_tokens, holdout_tokens
def main():
    source_counts, train_n, holdout_n = build_sft()
    wiki_train, wiki_holdout = build_wikitext()
    manifest = {
        "base_model": "v5_4h_balanced/final_hf",
        "seed": SEED,
        "judge_train_rows": train_n,
        "judge_holdout_rows": holdout_n,
        "source_counts": source_counts,
        "wikitext_train_tokens": wiki_train,
        "wikitext_local_holdout_tokens": wiki_holdout,
        "strict_split_policy": "Only official train splits are used. No benchmark validation/test examples enter training.",
        "benchmarks_targeted": ["HellaSwag", "ARC-Easy", "PIQA", "WinoGrande", "WikiText-103"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
