
from pathlib import Path
import json, random
from datasets import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "data_v9_generalize"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 26092026
rng = random.Random(SEED)

def add(rows, source, question, options, correct):
    options = [str(x).strip() for x in options]
    q = str(question).strip()
    if not q or len(options) < 2 or correct < 0 or correct >= len(options):
        return
    rows.append({"source": source, "question": q, "options": options, "correct": int(correct)})

def choices_index(choices, answer_key):
    labels = [str(x) for x in choices["label"]]
    key = str(answer_key)
    return labels.index(key)

def build():
    by_source = {}

    ds = load_dataset("Rowan/hellaswag", split="train")
    rows=[]
    for r in ds:
        add(rows,"hellaswag", "Choose the most plausible continuation. Context: " + r["ctx"], r["endings"], int(r["label"]))
    by_source["hellaswag"]=rows

    ds = load_dataset("allenai/ai2_arc","ARC-Easy",split="train")
    rows=[]
    for r in ds:
        try: idx=choices_index(r["choices"],r["answerKey"])
        except: continue
        add(rows,"arc_easy",r["question"],r["choices"]["text"],idx)
    by_source["arc_easy"]=rows

    ds = load_dataset("lighteval/piqa",split="train")
    rows=[]
    for r in ds:
        add(rows,"piqa","Choose the physically sensible solution. Goal: "+r.get("goal",r.get("question","")),
            [r.get("sol1",r.get("choice_a","")),r.get("sol2",r.get("choice_b",""))],
            int(r.get("label",r.get("answer",0))))
    by_source["piqa"]=rows

    ds = load_dataset("allenai/winogrande","winogrande_xl",split="train")
    rows=[]
    for r in ds:
        add(rows,"winogrande","Choose the option that best fills the blank. Sentence: "+r["sentence"],
            [r["option1"],r["option2"]],0 if str(r["answer"])=="1" else 1)
    by_source["winogrande"]=rows

    ds = load_dataset("allenai/ai2_arc","ARC-Challenge",split="train")
    rows=[]
    for r in ds:
        try: idx=choices_index(r["choices"],r["answerKey"])
        except: continue
        add(rows,"arc_challenge",r["question"],r["choices"]["text"],idx)
    by_source["arc_challenge"]=rows

    ds = load_dataset("allenai/openbookqa","main",split="train")
    rows=[]
    for r in ds:
        try: idx=choices_index(r["choices"],r["answerKey"])
        except: continue
        add(rows,"openbookqa",r["question_stem"],r["choices"]["text"],idx)
    by_source["openbookqa"]=rows

    ds = load_dataset("tau/commonsense_qa",split="train")
    rows=[]
    for r in ds:
        try: idx=choices_index(r["choices"],r["answerKey"])
        except: continue
        add(rows,"commonsenseqa",r["question"],r["choices"]["text"],idx)
    by_source["commonsenseqa"]=rows

    ds = load_dataset("allenai/sciq",split="train")
    rows=[]
    for r in ds:
        opts=[r["correct_answer"],r["distractor1"],r["distractor2"],r["distractor3"]]
        add(rows,"sciq",r["question"],opts,0)
    by_source["sciq"]=rows

    ds = load_dataset("google/boolq",split="train")
    rows=[]
    for r in ds:
        q="Read the passage and answer the question. Passage: "+r["passage"]+" Question: "+r["question"]
        add(rows,"boolq",q,["yes","no"],0 if bool(r["answer"]) else 1)
    by_source["boolq"]=rows

    train, holdout = [], []
    counts={}
    for src,rows in by_source.items():
        rng.shuffle(rows)
        cut=max(50,int(len(rows)*0.05))
        cut=min(cut,max(1,len(rows)//5))
        holdout.extend(rows[:cut])
        train.extend(rows[cut:])
        counts[src]={"train":len(rows)-cut,"holdout":cut,"total":len(rows)}

    rng.shuffle(train); rng.shuffle(holdout)
    with (OUT/"mc_train.jsonl").open("w",encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with (OUT/"mc_holdout.jsonl").open("w",encoding="utf-8") as f:
        for r in holdout: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    manifest={
        "seed":SEED,
        "policy":"Only official TRAIN splits are used for gradient training. Validation/test splits are excluded.",
        "dynamic_option_shuffle":"Trainer reshuffles answer options every sample to prevent answer-letter memorization.",
        "train_rows":len(train),
        "holdout_rows":len(holdout),
        "sources":counts
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    build()
