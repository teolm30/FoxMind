
from pathlib import Path
import json, random
from datasets import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "data_v10_reasoning"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 26092210
rng = random.Random(SEED)

def boxed_answer(s):
    pos = s.rfind(r"\boxed")
    if pos < 0:
        return ""
    i = s.find("{", pos)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i+1:j].strip()
    return ""

rows = []
gsm = load_dataset("openai/gsm8k", "main", split="train")
for r in gsm:
    ans = r["answer"].strip()
    final = ans.split("####")[-1].strip() if "####" in ans else ans.splitlines()[-1].strip()
    rationale = ans.split("####")[0].strip()
    rows.append({"source":"gsm8k","question":r["question"].strip(),"rationale":rationale,"answer":final})

for cfg in ["prealgebra","algebra","intermediate_algebra","number_theory","counting_and_probability"]:
    ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
    for r in ds:
        sol = r["solution"].strip()
        final = boxed_answer(sol)
        if final:
            rows.append({"source":"math_"+cfg,"question":r["problem"].strip(),"rationale":sol,"answer":final})

sv = load_dataset("ChilleD/SVAMP", split="train")
for r in sv:
    q = (str(r.get("Body","")).strip()+" "+str(r.get("Question","")).strip()).strip()
    equation = str(r.get("Equation","")).strip()
    answer = str(r.get("Answer","")).strip()
    rationale = ("Equation: "+equation) if equation else ""
    rows.append({"source":"svamp","question":q,"rationale":rationale,"answer":answer})
by = {}
for r in rows:
    by.setdefault(r["source"], []).append(r)

train, hold = [], []
counts = {}
for src, items in by.items():
    rng.shuffle(items)
    frac = .10 if src == "svamp" else .05
    cut = max(20, int(len(items) * frac))
    cut = min(cut, max(1, len(items)//5))
    hold.extend(items[:cut])
    train.extend(items[cut:])
    counts[src] = {"train":len(items)-cut,"holdout":cut,"total":len(items)}

rng.shuffle(train)
rng.shuffle(hold)
for name, data in [("math_train.jsonl", train), ("math_holdout.jsonl", hold)]:
    with (OUT / name).open("w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
manifest = {
    "seed": SEED,
    "policy": "Only official TRAIN splits are used; benchmark validation/test splits are excluded from gradient training.",
    "math_train_rows": len(train),
    "math_holdout_rows": len(hold),
    "sources": counts,
    "notes": "GSM8K and MATH retain human-written solutions; SVAMP uses provided equation and answer supervision."
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
