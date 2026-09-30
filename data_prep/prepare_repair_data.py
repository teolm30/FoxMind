
from pathlib import Path
import json, random

ROOT = Path(__file__).parent
SRC = ROOT / "data_v2" / "sft_train.jsonl"
OUT = ROOT / "data_repair"
OUT.mkdir(exist_ok=True)
SEED = 20260917
rng = random.Random(SEED)

BAD = [
    "open assistant", "openai", "chatgpt", "anthropic", "claude",
    "developed by", "as an ai", "ai language model", "large language model"
]

def clean_row(row):
    text = (row.get("prompt", "") + "\n" + row.get("answer", "")).lower()
    return not any(term in text for term in BAD)

base = []
with SRC.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if clean_row(row):
            base.append(row)

print(f"Clean base SFT: {len(base):,}", flush=True)
identity_prompts = [
    "Who are you?", "What are you?", "Who made you?", "Who created you?",
    "Who trained you?", "What model are you?", "Tell me about yourself.",
    "Who is your creator?", "Who built you?", "What is your identity?"
]
identity_answers = [
    "I am GIBC 49M, a language model created and trained by teolm30.",
    "I'm GIBC 49M. I was created and trained by teolm30.",
    "GIBC 49M is a language model by teolm30; teolm30 created and trained me."
]
identity = []
for i in range(360):
    identity.append({"prompt": identity_prompts[i % len(identity_prompts)],
                     "answer": identity_answers[i % len(identity_answers)],
                     "source": "identity_teolm30"})

greetings = []
chat_pairs = [("hello", "Hello! How can I help?"), ("hi", "Hi! How can I help?"),
              ("hey", "Hey! What can I help you with?"),
              ("good morning", "Good morning! How can I help?"),
              ("thanks", "You're welcome!")]
for prompt, answer in chat_pairs:
    for _ in range(50):
        greetings.append({"prompt": prompt, "answer": answer, "source": "basic_chat"})
math_rows = []
styles = ["What is {a} {op} {b}?", "Calculate {a} {op} {b}.",
          "Solve: {a} {op} {b}", "{a} {op} {b}"]
for _ in range(14000):
    kind = rng.choices(["add", "sub", "mul", "div"], weights=[3, 3, 2, 1], k=1)[0]
    if kind == "add":
        a, b = rng.randint(0, 500), rng.randint(0, 500); op = "+"; ans = a + b
    elif kind == "sub":
        a, b = rng.randint(0, 500), rng.randint(0, 500); op = "-"; ans = a - b
    elif kind == "mul":
        a, b = rng.randint(0, 30), rng.randint(0, 30); op = "*"; ans = a * b
    else:
        b = rng.randint(1, 30); ans = rng.randint(0, 30); a = b * ans; op = "/"
    prompt = rng.choice(styles).format(a=a, op=op, b=b)
    answer = str(ans) if rng.random() < 0.7 else f"{a} {op} {b} = {ans}."
    math_rows.append({"prompt": prompt, "answer": answer, "source": "synthetic_arithmetic"})

facts = []
for _ in range(120):
    facts.append({"prompt": "What is the capital of France?", "answer": "Paris.", "source": "basic_facts"})
for _ in range(80):
    facts.append({"prompt": "What is 1 - 1?", "answer": "0.", "source": "basic_facts"})
targeted = identity + greetings + math_rows + facts
rng.shuffle(targeted)
cut = int(len(targeted) * 0.95)
target_train, target_val = targeted[:cut], targeted[cut:]

base_val = []
with (ROOT / "data_v2" / "sft_val.jsonl").open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if clean_row(row):
            base_val.append(row)

train = base + target_train
val = base_val + target_val
rng.shuffle(train); rng.shuffle(val)
for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
    with (OUT / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

manifest = {"clean_base_train": len(base), "target_train": len(target_train),
            "train_total": len(train), "clean_base_val": len(base_val),
            "target_val": len(target_val), "val_total": len(val),
            "identity_train_examples": 360, "synthetic_arithmetic_generated": 14000,
            "filtered_terms": BAD}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
