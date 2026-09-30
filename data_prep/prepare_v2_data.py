
from pathlib import Path
import json, os, random
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ROOT = Path(__file__).parent
DATA = ROOT / "data_v2"
TOK = ROOT / "tokenizer_v2"
DATA.mkdir(exist_ok=True)
TOK.mkdir(exist_ok=True)
VOCAB_SIZE = 8192
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
random.seed(20260917)

SOURCES = {
    "fineweb": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"),
}
TARGETS = {"fineweb": 60_000_000, "finemath": 25_000_000, "cosmopedia": 15_000_000, "validation": 1_000_000}
def row_text(row):
    return (row.get("text") or row.get("content") or "").strip()


def stream_source(name):
    repo, config = SOURCES[name]
    return load_dataset(repo, config, split="train", streaming=True)


def load_sft_pairs():
    pairs = []
    with (ROOT / "data4h" / "sft.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("prompt") and row.get("answer"):
                pairs.append(row)
    random.Random(424242).shuffle(pairs)
    cut = int(len(pairs) * 0.95)
    train, val = pairs[:cut], pairs[cut:]
    for path, rows in [(DATA / "sft_train.jsonl", train), (DATA / "sft_val.jsonl", val)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train, val


def tokenizer_texts(sft_train):
    for row in sft_train:
        yield f"User: {row['prompt'].strip()}\nAssistant: {row['answer'].strip()}"
    limits = {"fineweb": 22000, "finemath": 18000, "cosmopedia": 18000}
    for name, limit in limits.items():
        print(f"Tokenizer sample: {name} ({limit:,} docs max)", flush=True)
        seen = 0
        for row in stream_source(name):
            text = row_text(row)
            if len(text) < 120:
                continue
            yield text[:20000]
            seen += 1
            if seen >= limit:
                break


def build_tokenizer(sft_train):
    print("Training NEW 8,192-token BPE on the actual V2 corpus...", flush=True)
    raw = Tokenizer(BPE(unk_token="<unk>"))
    raw.normalizer = NFKC()
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,
                         special_tokens=SPECIAL,
                         initial_alphabet=ByteLevel.alphabet(),
                         show_progress=True)
    raw.train_from_iterator(tokenizer_texts(sft_train), trainer=trainer)
    raw.save(str(TOK / "tokenizer.json"))
    fast = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>",
                                   bos_token="<bos>", eos_token="<eos>",
                                   unk_token="<unk>")
    fast.model_max_length = 2048
    fast.save_pretrained(TOK)
    if len(fast) != VOCAB_SIZE:
        raise RuntimeError(f"Tokenizer size is {len(fast)}, expected {VOCAB_SIZE}")
    return fast
def write_tokens(name, tok, train_limit, val_limit=0):
    ds = iter(stream_source(name))
    eos = tok.eos_token_id
    val_count = 0
    val_file = None
    if val_limit:
        val_file = (DATA / "val.bin").open("wb")
    train_file = (DATA / f"{name}.bin").open("wb")
    train_count = 0
    report = 5_000_000
    try:
        for row in ds:
            text = row_text(row)
            if len(text) < 120:
                continue
            ids = tok.encode(text, add_special_tokens=False) + [eos]
            pos = 0
            if val_count < val_limit:
                take = min(len(ids), val_limit - val_count)
                np.asarray(ids[:take], dtype=np.uint16).tofile(val_file)
                val_count += take
                pos += take
                if val_count < val_limit:
                    continue
            if pos < len(ids) and train_count < train_limit:
                ids2 = ids[pos:]
                take = min(len(ids2), train_limit - train_count)
                np.asarray(ids2[:take], dtype=np.uint16).tofile(train_file)
                train_count += take
            if train_count >= report:
                print(f"{name}: {train_count:,}/{train_limit:,} tokens", flush=True)
                report += 5_000_000
            if train_count >= train_limit:
                break
    finally:
        train_file.close()
        if val_file:
            val_file.close()
    if train_count != train_limit or val_count != val_limit:
        raise RuntimeError(f"{name} ended early: train={train_count:,} val={val_count:,}")
    return train_count, val_count
def main():
    ready = DATA / "manifest.json"
    if ready.exists() and (TOK / "tokenizer.json").exists():
        print("V2 data/tokenizer already prepared; skipping.", flush=True)
        return
    sft_train, sft_val = load_sft_pairs()
    print(f"SFT split: {len(sft_train):,} train / {len(sft_val):,} held-out", flush=True)
    tok = build_tokenizer(sft_train)
    counts = {}
    counts["fineweb"], counts["validation"] = write_tokens(
        "fineweb", tok, TARGETS["fineweb"], TARGETS["validation"])
    counts["finemath"], _ = write_tokens("finemath", tok, TARGETS["finemath"])
    counts["cosmopedia"], _ = write_tokens("cosmopedia", tok, TARGETS["cosmopedia"])
    manifest = {
        "tokenizer_vocab": len(tok),
        "token_counts": counts,
        "sft_train_pairs": len(sft_train),
        "sft_val_pairs": len(sft_val),
        "sources": SOURCES,
        "notes": [
            "Tokenizer trained from scratch on the same modern corpus plus SFT text.",
            "FineWeb validation is taken before FineWeb training tokens in one streaming pass.",
            "No pretrained model weights are used."
        ]
    }
    ready.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("V2 DATA READY", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
