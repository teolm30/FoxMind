
from pathlib import Path
import csv, json, math, random, re, time, zlib
import numpy as np
import torch, torch_directml
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).parent
BASE = ROOT / "checkpoints" / "v15_recovery_consolidation" / "final_hf"
DATA = ROOT / "data_v2"
REPAIR = ROOT / "data_repair"
JUDGE = ROOT / "data_v6_judges"
GEN = ROOT / "data_v9_generalize"
MATHDATA = ROOT / "data_v10_reasoning"
OUT = ROOT / "checkpoints" / "v16_final_competition"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ACTIVE = 6 * 60 * 60
MIN_TARGET_TIME = 99 * 60 * 60
PLATEAU_MIN_TIME = 99 * 60 * 60
EVAL_EVERY = 20 * 60
SAVE_EVERY = 20 * 60
BATCH, RANK_BATCH, SEQ = 24, 12, 256
SEED = 26092716
random.seed(SEED)
torch.manual_seed(SEED)
LETTERS = "ABCDEFGH"

tok = PreTrainedTokenizerFast.from_pretrained(BASE)
pad_id = tok.pad_token_id

fineweb = np.memmap(DATA / "fineweb.bin", dtype=np.uint16, mode="r")
finemath = np.memmap(DATA / "finemath.bin", dtype=np.uint16, mode="r")
cosmo = np.memmap(DATA / "cosmopedia.bin", dtype=np.uint16, mode="r")
wikitext = np.memmap(JUDGE / "wikitext_train.bin", dtype=np.uint16, mode="r")
wiki_hold = np.memmap(JUDGE / "wikitext_holdout.bin", dtype=np.uint16, mode="r")

def load_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

mc_train = load_jsonl(GEN / "mc_train.jsonl")
mc_hold = load_jsonl(GEN / "mc_holdout.jsonl")
repair_train = load_jsonl(REPAIR / "train.jsonl")
repair_val = load_jsonl(REPAIR / "val.jsonl")
math_train = load_jsonl(MATHDATA / "math_train.jsonl")
math_hold = load_jsonl(MATHDATA / "math_holdout.jsonl")
math_hold_simple = [r for r in math_hold if r["source"] in {"gsm8k","svamp"}]
general_pool = [r for r in repair_train if r["source"] in {"dolly15k","oasst1","gsm8k"}]
general_val = [r for r in repair_val if r["source"] in {"dolly15k","oasst1","gsm8k"}]
identity_pool = [r for r in repair_train if r["source"] in {"identity_teolm30","basic_chat","basic_facts"}]
identity_val = [r for r in repair_val if r["source"] in {"identity_teolm30","basic_chat","basic_facts"}]

mc_by_source={}
hold_by_source={}
for r in mc_train: mc_by_source.setdefault(r["source"],[]).append(r)
for r in mc_hold: hold_by_source.setdefault(r["source"],[]).append(r)
SOURCES=sorted(mc_by_source)
BASE_SOURCE_WEIGHTS={
    "hellaswag":.10,"arc_easy":.16,"piqa":.08,"winogrande":.08,
    "arc_challenge":.12,"openbookqa":.16,"commonsenseqa":.14,"sciq":.08,"boolq":.08,
}
source_weights={s:BASE_SOURCE_WEIGHTS.get(s,1/len(SOURCES)) for s in SOURCES}

letter_token={}
for c in LETTERS:
    ids=tok.encode(" "+c,add_special_tokens=False)
    letter_token[c]=ids[0] if ids else None

MUST_PASS=[
 ("What is 1 - 1?","0"),("What is 17 - 9?","8"),("What is 7 * 8?","56"),
 ("What is 9 * 9?","81"),("What is 12 * 12?","144"),("What is 20 / 4?","5"),
 ("What number comes next: 3, 6, 9, 12?","15"),("Solve for x: x + 7 = 19.","12"),
 ("Which number is larger: 37 or 42?","42"),
 ("All cats are animals. Milo is a cat. Is Milo an animal? Answer yes or no.","yes"),
]

def make_mask(batch, seq, device):
    m=torch.zeros((seq,seq),dtype=torch.float32)
    m[torch.triu(torch.ones((seq,seq),dtype=torch.bool),diagonal=1)] = -1e4
    return m[None,None].repeat(batch,1,1,1).to(device)

def lr_at(t):
    if t < 1200:
        return 6.0e-8 + (2.2e-7-6.0e-8)*(t/1200)
    if t < 3*3600:
        q=(t-1200)/(3*3600-1200)
        return 1.2e-7 + .5*(2.2e-7-1.2e-7)*(1+math.cos(math.pi*q))
    q=min(1.0,(t-3*3600)/(3*3600))
    return 4.0e-8 + .5*(1.2e-7-4.0e-8)*(1+math.cos(math.pi*q))

def mode_weights(t):
    # Phase 1: restore benchmark strength while retaining language quality.
    if t < 2*3600:
        return {"mc":.28,"rank":.18,"lm":.18,"general":.14,"math_solve":.08,
                "math_answer":.03,"reason":.04,"identity":.03,"precision":.03,"instr":.01}
    # Phase 2: push reasoning/math without abandoning benchmark retention.
    if t < 4*3600:
        return {"mc":.20,"rank":.14,"lm":.17,"general":.12,"math_solve":.18,
                "math_answer":.07,"reason":.06,"precision":.03,"identity":.02,"instr":.01}
    # Phase 3: consolidate MC/rank/perplexity while preserving reasoning gains.
    return {"mc":.23,"rank":.17,"lm":.24,"general":.15,"math_solve":.10,
            "math_answer":.03,"reason":.03,"identity":.03,"instr":.01,"precision":.01}

def choose_mode(t):
    w=mode_weights(t)
    return random.choices(list(w),weights=list(w.values()),k=1)[0]

def choose_lm_source():
    return random.choices([wikitext,fineweb,finemath],weights=[.35,.35,.30],k=1)[0]

def encode_sft(prompt, answer):
    pids=tok.encode("User: "+str(prompt).strip()+"\nAssistant:",add_special_tokens=False)
    aids=tok.encode(" "+str(answer).strip(),add_special_tokens=False)+[tok.eos_token_id]
    max_prompt=max(24,SEQ-min(len(aids),144)-1)
    pids=pids[-max_prompt:]
    aids=aids[:SEQ-len(pids)]
    ids=pids+aids
    labels=[-100]*len(pids)+aids.copy()
    pad=SEQ-len(ids)
    return ids+[pad_id]*pad, labels+[-100]*pad

def generic_sft_batch(pool,device):
    packed=[encode_sft(r["prompt"],r["answer"]) for r in random.choices(pool,k=BATCH)]
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),
            torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

def encode_math_solution(row):
    q=row["question"].strip()
    pids=tok.encode("User: Solve carefully and show the reasoning.\n"+q+"\nAssistant:",add_special_tokens=False)
    final=tok.encode("\nFinal answer: "+str(row["answer"]).strip(),add_special_tokens=False)+[tok.eos_token_id]
    rationale=tok.encode(" "+str(row["rationale"]).strip(),add_special_tokens=False)
    pids=pids[-min(len(pids),120):]
    budget=max(0,SEQ-len(pids)-len(final))
    rationale=rationale[:budget]
    aids=rationale+final
    if len(pids)+len(aids)>SEQ:
        final=final[-max(1,SEQ-len(pids)):]
        aids=final
    ids=pids+aids
    labels=[-100]*len(pids)+aids.copy()
    pad=SEQ-len(ids)
    return ids+[pad_id]*pad,labels+[-100]*pad

def math_solve_batch(device):
    packed=[encode_math_solution(r) for r in random.choices(math_train,k=BATCH)]
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),
            torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

def math_answer_batch(device):
    packed=[]
    for r in random.choices(math_train,k=BATCH):
        prompt=random.choice([
          "Answer with only the final answer.\n"+r["question"],
          "Give only the answer, with no explanation.\n"+r["question"],
          r["question"]+"\nFinal answer only:"
        ])
        packed.append(encode_sft(prompt,r["answer"]))
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),
            torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

def render_mc(row,rng):
    n=len(row["options"])
    perm=list(range(n)); rng.shuffle(perm)
    new_correct=perm.index(int(row["correct"]))
    opts=[row["options"][i] for i in perm]
    lines="\n".join(f"{LETTERS[i]}) {x}" for i,x in enumerate(opts))
    prompt="Answer with only the option letter.\n"+row["question"].strip()+"\n"+lines
    return prompt, LETTERS[new_correct], list(LETTERS[:n])

def choose_mc_row():
    src=random.choices(SOURCES,weights=[source_weights[s] for s in SOURCES],k=1)[0]
    return random.choice(mc_by_source[src])

def encode_rank_sequence(prompt, option):
    pids=tok.encode(str(prompt).strip()+"\nAnswer:",add_special_tokens=False)
    oids=tok.encode(" "+str(option).strip(),add_special_tokens=False)
    oids=oids[:96]
    max_prompt=max(16,SEQ-len(oids))
    pids=pids[-max_prompt:]
    ids=(pids+oids)[:SEQ]
    cand_start=max(0,len(pids)-1)
    cand_end=min(SEQ-1,cand_start+len(oids))
    cmask=[0.0]*(SEQ-1)
    for i in range(cand_start,cand_end):
        cmask[i]=1.0
    ids=ids+[pad_id]*(SEQ-len(ids))
    return ids,cmask

def rank_step_loss(model,device):
    ids=[]; cmasks=[]
    for _ in range(RANK_BATCH):
        row=choose_mc_row()
        correct=int(row["correct"])
        wrong=[i for i in range(len(row["options"])) if i!=correct]
        if random.random()<.65:
            target_len=len(str(row["options"][correct]))
            neg=min(wrong,key=lambda i:abs(len(str(row["options"][i]))-target_len))
        else:
            neg=random.choice(wrong)
        for idx in (correct,neg):
            a,b=encode_rank_sequence(row["question"],row["options"][idx])
            ids.append(a); cmasks.append(b)
    x=torch.tensor(ids,dtype=torch.long,device=device)
    cm=torch.tensor(cmasks,dtype=torch.float32,device=device)
    logits=model(input_ids=x,attention_mask=make_mask(len(ids),SEQ,device),use_cache=False).logits[:,:-1,:]
    targets=x[:,1:]
    vocab=logits.shape[-1]
    nll=F.cross_entropy(logits.reshape(-1,vocab),targets.reshape(-1),reduction="none").view(targets.shape)
    score=-(nll*cm).sum(1)/cm.sum(1).clamp_min(1.0)
    pos=score[0::2]; neg=score[1::2]
    ranking=F.softplus(-(pos-neg)*2.0).mean()
    positive_nll=-pos.mean()
    return ranking+.03*positive_nll,x.numel()

def mc_batch(device):
    packed=[]
    for _ in range(BATCH):
        prompt,ans,_=render_mc(choose_mc_row(),random)
        packed.append(encode_sft(prompt,ans))
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),
            torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

def lm_batch(data,device):
    rows=[]
    for _ in range(BATCH):
        s=random.randrange(0,len(data)-SEQ-1)
        rows.append(torch.from_numpy(np.array(data[s:s+SEQ],dtype=np.int64)))
    x=torch.stack(rows).to(device)
    return x,x.clone()

def reason_row():
    kind=random.choice(["arith","linear","seq","rate","percent","compare","logic","equal","convert","physical"])
    if kind=="arith":
        op=random.choice(["+","-","*","/"])
        if op=="+":
            a,b=random.randint(-500,500),random.randint(-500,500); ans=a+b
        elif op=="-":
            a,b=random.randint(-500,500),random.randint(-500,500); ans=a-b
        elif op=="*":
            a,b=random.randint(-30,30),random.randint(-20,20); ans=a*b
        else:
            b=random.randint(1,25); ans=random.randint(-30,30); a=b*ans
        q=random.choice([f"Calculate {a} {op} {b}.",f"What is {a} {op} {b}?",f"Give the result of {a} {op} {b}."])
        return q,str(ans)
    if kind=="linear":
        x=random.randint(-25,25)
        if random.random()<.5:
            a=random.randint(-30,30); c=x+a
            return random.choice([f"Find x: x + {a} = {c}.",f"What value of x makes x + {a} equal {c}?"]),str(x)
        a=random.choice([2,3,4,5,6,7,8,9]); b=random.randint(-20,20); c=a*x+b
        return random.choice([f"Solve for x: {a}*x + {b} = {c}.",f"If {a}x + {b} = {c}, what is x?"]),str(x)
    if kind=="seq":
        if random.random()<.6:
            start=random.randint(-30,30); step=random.randint(2,15)
            vals=[start+i*step for i in range(4)]; ans=vals[-1]+step
        else:
            start=random.choice([1,2,3,4,5]); ratio=random.choice([2,3])
            vals=[start*(ratio**i) for i in range(4)]; ans=vals[-1]*ratio
        return random.choice(["Continue the sequence: ","What number comes next: "])+", ".join(map(str,vals))+"?",str(ans)
    if kind=="rate":
        speed=random.choice([20,30,40,50,60,70,80,90]); hours=random.randint(2,6); ans=speed*hours
        return random.choice([f"A vehicle travels at {speed} km/h for {hours} hours. How far does it go?",f"At {speed} km per hour, how many km are covered in {hours} hours?"]),str(ans)
    if kind=="percent":
        p=random.choice([10,20,25,50]); n=random.choice([20,40,60,80,100,120,160,200]); ans=n*p//100
        return random.choice([f"What is {p}% of {n}?",f"Calculate {p} percent of {n}."]),str(ans)
    if kind=="compare":
        a,b=random.sample(range(-500,501),2)
        return f"Which number is larger: {a} or {b}?",str(max(a,b))
    if kind=="logic":
        noun=random.choice(["glip","norb","tavin","zor","mep"])
        sup=random.choice(["objects","creatures","things","items"])
        name=random.choice(["Rin","Tavi","Milo","Kora"])
        return f"Every {noun} is one of the {sup}. {name} is a {noun}. Is {name} one of the {sup}? Answer yes or no.","yes"
    if kind=="equal":
        a,b=random.choice([("1 kg of feathers","1 kg of iron"),("2 kg of sand","2 kg of water"),("500 g of wood","500 g of steel")])
        return f"Which is heavier: {a} or {b}? Answer 'same' if they weigh the same.","same"
    if kind=="convert":
        n=random.randint(1,20)
        unit=random.choice(["cm","g","minutes"])
        if unit=="cm": return f"How many centimeters are in {n} meters?",str(n*100)
        if unit=="g": return f"How many grams are in {n} kilograms?",str(n*1000)
        return f"How many minutes are in {n} hours?",str(n*60)
    cases=[
      ("A glass of water tips over on a dry floor. What most directly makes the floor wet?","spilled water"),
      ("A lamp is switched on in a dark room. What provides the light?","the lamp"),
      ("A plant gets no water for many days and wilts. What most likely caused it?","lack of water"),
    ]
    return random.choice(cases)

def reason_batch(device):
    packed=[encode_sft(*reason_row()) for _ in range(BATCH)]
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),
            torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

WORDS=["BLUE","GREEN","FOX","ORANGE","NOVA","BOLT","RIVER","STONE","ALPHA","GAMMA"]
def precision_row():
    k=random.randrange(10)
    if k==0:
        a,b=random.randint(-250,250),random.randint(-250,250)
        q=random.choice([f"What is {a} + {b}?",f"Calculate {a} plus {b}.",f"Give only the result of {a}+{b}."]); return q,str(a+b)
    if k==1:
        a,b=random.randint(-250,250),random.randint(-250,250)
        q=random.choice([f"What is {a} - {b}?",f"Calculate {a} minus {b}.",f"Give only the result of {a}-{b}."]); return q,str(a-b)
    if k==2:
        a,b=random.randint(-20,20),random.randint(-15,15); return random.choice([f"What is {a} * {b}?",f"Multiply {a} by {b}."]),str(a*b)
    if k==3:
        b=random.randint(1,20); ans=random.randint(-20,20); a=b*ans; return random.choice([f"What is {a} / {b}?",f"Divide {a} by {b}."]),str(ans)
    if k==4:
        x=random.randint(-30,30); c=random.randint(-30,30); return random.choice([f"Solve for x: x + {c} = {x+c}.",f"If x + {c} = {x+c}, what is x?"]),str(x)
    if k==5:
        x=random.randint(-20,20); a=random.randint(2,9); c=random.randint(-15,15); return f"Solve for x: {a}*x + {c} = {a*x+c}.",str(x)
    if k==6:
        n=random.choice([20,40,60,80,100,120,160,200,240]); q=random.choice([10,20,25,50]); return f"What is {q}% of {n}?",str(n*q//100)
    if k==7:
        start=random.randint(-30,30); step=random.randint(2,15); vals=[start+i*step for i in range(4)]; return "What number comes next: "+", ".join(map(str,vals))+"?",str(vals[-1]+step)
    if k==8:
        n=random.randint(1,15); u=random.choice(["meters","kilograms","hours"])
        if u=="meters": return f"How many centimeters are in {n} meters?",str(n*100)
        if u=="kilograms": return f"How many grams are in {n} kilograms?",str(n*1000)
        return f"How many minutes are in {n} hours?",str(n*60)
    speed=random.choice([20,30,40,50,60,70,80,90]); h=random.randint(2,6); return f"A vehicle travels {speed} km/h for {h} hours. How far does it travel?",str(speed*h)

def precision_batch(device):
    packed=[encode_sft(*precision_row()) for _ in range(BATCH)]
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

WORDS=["BLUE","GREEN","FOX","ORANGE","NOVA","BOLT","RIVER","STONE","ALPHA","GAMMA","PIXEL","EMBER","COMET","VIOLET"]
def instr_row():
    k=random.randrange(6)
    if k in (0,1,2):
        w=random.choice(WORDS); prompt=random.choice([f"Reply with exactly {w}.",f"Return exactly the word {w} and nothing else.",f"Your entire response must be {w}.",f"Output only {w}."]); return prompt,w
    if k in (3,4):
        n=random.randint(-999,1999); return random.choice([f"Return only the integer {n}.",f"Respond with exactly {n} and nothing else.",f"Give only the number {n}."]),str(n)
    return random.choice([("Say only YES.","YES"),("Say only NO.","NO"),("Reply only same.","same")])

def instr_batch(device):
    packed=[encode_sft(*instr_row()) for _ in range(BATCH)]
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

COMMON_CASES=[
 ("A cup tips over and water spills onto a table. What made the table wet?","spilled water"),
 ("A glass of water spills on a dry floor. What made the floor wet?","spilled water"),
 ("A lamp is switched on in a dark room. What provides the light?","the lamp"),
 ("A flashlight is switched on in the dark. What provides the light?","the flashlight"),
 ("A plant gets no water for many days and wilts. What most likely caused it?","lack of water"),
 ("A flower gets no water for a week and wilts. What most likely caused it?","lack of water"),
 ("One kilogram of cotton and one kilogram of iron are compared. Which is heavier? Answer same if equal.","same"),
 ("Two kilograms of wood and two kilograms of steel are compared. Which is heavier? Answer same if equal.","same"),
 ("Ice is left in a warm room and turns into liquid water. What caused the ice to melt?","heat"),
 ("A book falls from a shelf to the floor. What pulled it downward?","gravity"),
 ("A person touches a hot pan and quickly pulls their hand away. What caused the pain?","heat"),
 ("Rain falls from clouds and makes the ground wet. What made the ground wet?","rain")
]
def common_row():
    q,a=random.choice(COMMON_CASES); return random.choice([q,"Answer briefly. "+q,q+" Give a short answer."]),a

def common_batch(device):
    packed=[encode_sft(*common_row()) for _ in range(BATCH)]
    return (torch.tensor([x[0] for x in packed],dtype=torch.long,device=device),torch.tensor([x[1] for x in packed],dtype=torch.long,device=device))

def greedy_answer(model,device,q,max_new=24):
    ids=tok.encode("User: "+q.strip()+"\nAssistant:",add_special_tokens=False)
    x=torch.tensor([ids],dtype=torch.long,device=device)
    start=len(ids)
    for _ in range(max_new):
        mask=make_mask(1,x.shape[1],device)
        with torch.no_grad():
            z=model(input_ids=x,attention_mask=mask,use_cache=False).logits[0,-1]
        nxt=int(torch.argmax(z).detach().cpu())
        x=torch.cat([x,torch.tensor([[nxt]],dtype=torch.long,device=device)],dim=1)
        if nxt==tok.eos_token_id: break
    return tok.decode(x[0,start:].detach().cpu().tolist(),skip_special_tokens=True).strip()

def norm(s):
    return str(s).strip().lower().strip(" .,!?:;\"'")

def fixed_reason_holdout():
    rng=random.Random(77123)
    rows=[]
    for _ in range(6):
        a,b=rng.randint(-200,200),rng.randint(-200,200)
        rows.append((f"Work out the sum of {a} and {b}.",str(a+b)))
    for _ in range(6):
        x=rng.randint(-20,20); a=rng.choice([2,3,4,5,6,7]); b=rng.randint(-12,12); c=a*x+b
        rows.append((f"The equation is {a}x + {b} = {c}. Give x.",str(x)))
    for _ in range(4):
        s=rng.randint(1,8); ratio=rng.choice([2,3]); vals=[s*ratio**i for i in range(4)]
        rows.append(("Infer the next term in "+", ".join(map(str,vals))+".",str(vals[-1]*ratio)))
    for _ in range(4):
        speed=rng.choice([30,40,50,60,70]); h=rng.randint(2,5)
        rows.append((f"A trip lasts {h} hours at a constant {speed} km/h. State the distance in km.",str(speed*h)))
    rows += [
      ("One kilogram of cotton and one kilogram of metal are compared. Which has greater mass? Answer same if equal.","same"),
      ("Every blick is a creature. Sora is a blick. Is Sora a creature? Answer yes or no.","yes"),
      ("Convert 7 kilograms to grams.","7000"),
      ("Convert 3 hours to minutes.","180"),
    ]
    return rows

REASON_HOLD=fixed_reason_holdout()
INSTR_HOLD=[
 ("Write only BLUE. Do not add punctuation or explanation.","BLUE"),
 ("Your entire response must be GREEN.","GREEN"),
 ("Reply with exactly FOX.","FOX"),
 ("Give only the number 314.","314"),
 ("Your answer must contain nothing except -17.","-17"),
 ("Return one token: NOVA","NOVA"),
 ("Reply only YES.","YES"),("Reply only NO.","NO"),
]

def fixed_arith_holdout():
    rng=random.Random(91337); rows=[]
    for _ in range(8):
        a,b=rng.randint(-220,220),rng.randint(-220,220); rows.append((f"Calculate {a} plus {b}.",str(a+b)))
    for _ in range(8):
        a,b=rng.randint(-220,220),rng.randint(-220,220); rows.append((f"Calculate {a} minus {b}.",str(a-b)))
    for _ in range(6):
        a,b=rng.randint(-18,18),rng.randint(-12,12); rows.append((f"Multiply {a} by {b}.",str(a*b)))
    for _ in range(4):
        b=rng.randint(2,18); ans=rng.randint(-18,18); rows.append((f"Divide {b*ans} by {b}.",str(ans)))
    for _ in range(4):
        x=rng.randint(-25,25); c=rng.randint(-20,20); rows.append((f"If x + {c} = {x+c}, what is x?",str(x)))
    return rows

ARITH_HOLD=fixed_arith_holdout()
EXACT_HOLD=[
 ("Reply with exactly SILVER.","SILVER"),("Your entire response must be MANGO.","MANGO"),
 ("Output only DELTA.","DELTA"),("Return exactly the word CRYSTAL and nothing else.","CRYSTAL"),
 ("Return only the integer 512.","512"),("Give only the number -241.","-241"),
 ("Respond with exactly 903 and nothing else.","903"),("Reply only YES.","YES"),("Say only NO.","NO")
]
COMMON_HOLD=[
 ("A mug falls over and juice pours onto a desk. What made the desk wet?","spilled juice"),
 ("A torch is turned on in a dark cave. What provides the light?","the torch"),
 ("A houseplant receives no water for several days and droops. What caused it?","lack of water"),
 ("Three kilograms of feathers and three kilograms of metal are compared. Which is heavier? Answer same if equal.","same"),
 ("Snow is brought indoors and becomes liquid water. What caused it to melt?","heat"),
 ("An apple drops from a tree toward the ground. What pulled it down?","gravity"),
 ("A hand touches a very hot surface and hurts. What caused the pain?","heat"),
 ("A shower sprays water onto the bathroom floor. What made the floor wet?","water")
]

def eval_generation(model,device,rows,strict=False):
    good=0; details=[]
    model.eval()
    for q,e in rows:
        a=greedy_answer(model,device,q,20)
        ok=(a.strip()==e) if strict else (norm(a.splitlines()[0] if a else "")==norm(e))
        good += int(ok); details.append({"q":q,"expected":e,"answer":a,"ok":ok})
    model.train()
    return good/len(rows),details

def mc_eval(model,device,per_source=20):
    model.eval()
    total=good=0; by={}
    with torch.no_grad():
        for src in SOURCES:
            pool=hold_by_source.get(src,[])
            pick=pool[:min(per_source,len(pool))]
            sg=0
            for row in pick:
                seed=(zlib.crc32(row["question"].encode("utf-8")) ^ 0xA53C9E21) & 0xffffffff
                prompt,ans,cands=render_mc(row,random.Random(seed))
                ids=tok.encode("User: "+prompt+"\nAssistant:",add_special_tokens=False)
                ids=ids[-1900:]
                x=torch.tensor([ids],dtype=torch.long,device=device)
                z=model(input_ids=x,attention_mask=make_mask(1,len(ids),device),use_cache=False).logits[0,-1]
                pred=max(cands,key=lambda c:float(z[letter_token[c]].detach().cpu()))
                sg += int(pred==ans); good += int(pred==ans); total += 1
            by[src]=sg/max(1,len(pick))
    model.train()
    return good/max(1,total),by

def rank_eval(model,device,per_source=16):
    model.eval()
    total=good=0; by={}
    with torch.no_grad():
        for src in SOURCES:
            pick=hold_by_source.get(src,[])[:per_source]
            sg=0
            for row in pick:
                ids=[]; cmasks=[]
                for option in row["options"]:
                    a,b=encode_rank_sequence(row["question"],option)
                    ids.append(a); cmasks.append(b)
                x=torch.tensor(ids,dtype=torch.long,device=device)
                cm=torch.tensor(cmasks,dtype=torch.float32,device=device)
                logits=model(input_ids=x,attention_mask=make_mask(len(ids),SEQ,device),use_cache=False).logits[:,:-1,:]
                targets=x[:,1:]
                tgt=logits.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
                logp=tgt-torch.logsumexp(logits,dim=-1)
                scores=(logp*cm).sum(1)/cm.sum(1).clamp_min(1.0)
                pred=int(torch.argmax(scores).detach().cpu())
                ok=pred==int(row["correct"])
                sg+=int(ok); good+=int(ok); total+=1
            by[src]=sg/max(1,len(pick))
    model.train()
    return good/max(1,total),by

def parse_number(s):
    vals=re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?",str(s))
    if not vals: return None
    v=vals[-1].replace(",","")
    try:
        if "/" in v:
            a,b=v.split("/",1); return float(a)/float(b)
        return float(v)
    except Exception:
        return None

def math_eval(model,device,n=48):
    model.eval(); good=0; details=[]
    for row in math_hold_simple[:n]:
        q="Answer with only the final answer.\n"+row["question"]
        a=greedy_answer(model,device,q,18)
        got=parse_number(a); exp=parse_number(row["answer"])
        ok=(got is not None and exp is not None and abs(got-exp)<=max(1e-6,abs(exp)*1e-6))
        good+=int(ok)
        details.append({"source":row["source"],"q":row["question"],"expected":row["answer"],"answer":a,"ok":ok})
    model.train()
    return good/max(1,len(details)),details

def val_loss(model,device,kind,n=2):
    vals=[]; model.eval(); mask=make_mask(BATCH,SEQ,device)
    with torch.no_grad():
        for _ in range(n):
            if kind=="wiki": x,y=lm_batch(wiki_hold,device)
            elif kind=="general": x,y=generic_sft_batch(general_val,device)
            else: x,y=generic_sft_batch(identity_val,device)
            vals.append(float(model(input_ids=x,attention_mask=mask,labels=y,use_cache=False).loss.detach().cpu()))
    model.train()
    return sum(vals)/len(vals)

def must_pass_eval(model,device):
    acc,details=eval_generation(model,device,MUST_PASS,False)
    return int(round(acc*len(MUST_PASS))),details

def evaluate_bundle(model,device):
    rank,by=rank_eval(model,device,16)
    mc,mcby=mc_eval(model,device,16)
    math_acc,md=math_eval(model,device,48)
    arith_acc,ad=eval_generation(model,device,ARITH_HOLD,False)
    reason_acc,rd=eval_generation(model,device,REASON_HOLD,False)
    instr,idd=eval_generation(model,device,EXACT_HOLD,True)
    common_acc,cd=eval_generation(model,device,COMMON_HOLD,False)
    mp,mpd=must_pass_eval(model,device)
    return {
      "rank_accuracy":rank,"rank_by_source":by,"mc_accuracy":mc,"mc_by_source":mcby,"math_accuracy":math_acc,"arithmetic_accuracy":arith_acc,"reason_accuracy":reason_acc,"instruction_exact":instr,"commonsense_accuracy":common_acc,
      "wiki_loss":val_loss(model,device,"wiki",2),
      "general_loss":val_loss(model,device,"general",2),
      "identity_loss":val_loss(model,device,"identity",1),
      "must_pass":mp,
    }, {"math":md,"arithmetic":ad,"reason":rd,"instruction":idd,"commonsense":cd,"must_pass":mpd}

def quality(m,baseline):
    wiki=min(1.05,baseline["wiki_loss"]/max(m["wiki_loss"],1e-9))
    general=min(1.05,baseline["general_loss"]/max(m["general_loss"],1e-9))
    return (.24*m["mc_accuracy"]+.18*m["rank_accuracy"]+.18*m["math_accuracy"]
            +.10*m["arithmetic_accuracy"]+.10*m["reason_accuracy"]
            +.08*wiki+.05*general+.07*(m["must_pass"]/10))

def target_met(m,baseline):
    return False

def adapt_sources(by,mcby=None):
    global source_weights
    raw={}
    for s in SOURCES:
        base=BASE_SOURCE_WEIGHTS.get(s,1/len(SOURCES))
        acc=by.get(s,.5) if mcby is None else .5*(by.get(s,.5)+mcby.get(s,.5))
        raw[s]=base*(1.0+2.0*(1.0-acc))
    z=sum(raw.values())
    source_weights={s:raw[s]/z for s in SOURCES}

def cpu_copy(obj):
    if torch.is_tensor(obj): return obj.detach().cpu()
    if isinstance(obj,dict): return {k:cpu_copy(v) for k,v in obj.items()}
    if isinstance(obj,list): return [cpu_copy(v) for v in obj]
    if isinstance(obj,tuple): return tuple(cpu_copy(v) for v in obj)
    return obj

def save_resume(model,opt,active,step,tokens,best_score,best_active,no_improve):
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    tmp=OUT/"resume_latest.tmp"
    torch.save({"active":active,"step":step,"tokens":tokens,"model":state,
                "optimizer":cpu_copy(opt.state_dict()),"best_score":best_score,
                "best_active":best_active,"no_improve":no_improve,"source_weights":source_weights},tmp)
    tmp.replace(OUT/"resume_latest.pt")
    print(f"RESUME saved at {active/3600:.2f} active hours",flush=True)

def save_best(model,metrics,active,score):
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({"active":active,"score":score,"metrics":metrics,"model":state},OUT/"best_model.pt")
    print(f"BEST updated | quality={score:.4f} | active={active/3600:.2f}h",flush=True)

def causality_test(model,device):
    a=torch.tensor([[10,11,12,13,14,15]],dtype=torch.long,device=device); b=a.clone(); b[0,5]=16
    m=make_mask(1,6,device)
    model.eval()
    with torch.no_grad():
        la=model(input_ids=a,attention_mask=m,use_cache=False).logits[0,:5].detach().cpu()
        lb=model(input_ids=b,attention_mask=m,use_cache=False).logits[0,:5].detach().cpu()
    model.train(); diff=float((la-lb).abs().max())
    print(f"CAUSAL SELF-TEST {diff:.8f}",flush=True)
    if diff>1e-5: raise RuntimeError("causal mask leak")

def main():
    device=torch_directml.device()
    model=AutoModelForCausalLM.from_pretrained(BASE).to(device)
    model.lm_head.weight=model.model.embed_tokens.weight
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params>50_000_000: raise RuntimeError(f"Parameter cap exceeded: {params:,}")
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-7,betas=(.9,.95),weight_decay=.008)
    mask256=make_mask(BATCH,SEQ,device)
    causality_test(model,device)

    baseline_file=OUT/"baseline.json"
    if baseline_file.exists():
        baseline=json.loads(baseline_file.read_text(encoding="utf-8"))
    else:
        print("Running V16 V15 baseline evaluation...",flush=True)
        baseline,details=evaluate_bundle(model,device)
        baseline_file.write_text(json.dumps(baseline,indent=2),encoding="utf-8")
        (OUT/"baseline_details.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    print("BASELINE "+json.dumps(baseline),flush=True)

    active=0.0; step=0; tokens=0; no_improve=0
    best_score=quality(baseline,baseline); best_active=0.0
    resume=OUT/"resume_latest.pt"
    best_file=OUT/"best_model.pt"
    if resume.exists():
        r=torch.load(resume,map_location="cpu")
        model.load_state_dict(r["model"]); model=model.to(device); model.lm_head.weight=model.model.embed_tokens.weight
        opt.load_state_dict(r["optimizer"])
        active=float(r["active"]); step=int(r["step"]); tokens=int(r["tokens"])
        best_score=float(r.get("best_score",best_score)); best_active=float(r.get("best_active",0)); no_improve=int(r.get("no_improve",0))
        if "source_weights" in r:
            source_weights.update(r["source_weights"])
        print(f"RESUME V16 at {active/3600:.2f} active hours",flush=True)
    else:
        save_best(model,baseline,0.0,best_score)
        print(f"V16 FINAL COMPETITION | {params:,} params | max 6 active hours | base=V15 selected",flush=True)

    next_save=(int(active//SAVE_EVERY)+1)*SAVE_EVERY
    next_eval=(int(active//EVAL_EVERY)+1)*EVAL_EVERY
    stop_reason="max_active_time"
    log=OUT/"train.csv"; mode="a" if active else "w"
    with log.open(mode,newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        if not active: w.writerow(["step","active_seconds","mode","loss","lr","tokens","tok_per_s"])
        while active < MAX_ACTIVE:
            kind=choose_mode(active)
            if kind=="lm": x,y=lm_batch(choose_lm_source(),device)
            elif kind=="math_solve": x,y=math_solve_batch(device)
            elif kind=="math_answer": x,y=math_answer_batch(device)
            elif kind=="instr": x,y=instr_batch(device)
            elif kind=="reason": x,y=reason_batch(device)
            elif kind=="common": x,y=common_batch(device)
            elif kind=="precision": x,y=precision_batch(device)
            elif kind=="mc": x,y=mc_batch(device)
            elif kind=="general": x,y=generic_sft_batch(general_pool,device)
            elif kind=="identity": x,y=generic_sft_batch(identity_pool,device)
            else: x=y=None

            t0=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            if kind=="rank":
                loss,step_tokens=rank_step_loss(model,device)
            else:
                loss=model(input_ids=x,attention_mask=mask256,labels=y,use_cache=False).loss
                step_tokens=x.numel()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.45)
            lr=lr_at(active)
            for g in opt.param_groups: g["lr"]=lr
            opt.step()
            dt=time.perf_counter()-t0
            if dt>20:
                print(f"IGNORED inactive/sleep gap of {dt:.1f}s",flush=True)
                dt=min(dt,2.0)
            active += dt
            step += 1; tokens += step_tokens

            if step%20==0:
                raw=float(loss.detach().cpu()); speed=tokens/max(active,1e-6)
                w.writerow([step,f"{active:.1f}",kind,f"{raw:.6f}",f"{lr:.8f}",tokens,f"{speed:.1f}"]); f.flush()
                print(f"{kind:11s} | step {step:6d} | loss {raw:6.3f} | {speed:7.0f} tok/s | {active/3600:5.2f}/6h active",flush=True)

            if active>=next_save:
                save_resume(model,opt,active,step,tokens,best_score,best_active,no_improve)
                next_save += SAVE_EVERY

            if active>=next_eval:
                metrics,details=evaluate_bundle(model,device)
                score=quality(metrics,baseline)
                record={"active_seconds":active,"quality":score,**metrics}
                with (OUT/"eval.jsonl").open("a",encoding="utf-8") as ef: ef.write(json.dumps(record)+"\n")
                (OUT/"latest_eval_details.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
                print("EVAL "+json.dumps(record),flush=True)
                adapt_sources(metrics["rank_by_source"],metrics["mc_by_source"])
                print("ADAPT_WEIGHTS "+json.dumps(source_weights),flush=True)
                guard=(metrics["must_pass"]>=9 and metrics["rank_accuracy"]>=baseline["rank_accuracy"]-.04
                       and metrics["mc_accuracy"]>=baseline["mc_accuracy"]-.015
                       and metrics["wiki_loss"]<=baseline["wiki_loss"]*1.04
                       and metrics["general_loss"]<=baseline["general_loss"]*1.08)
                if guard and score > best_score + .002:
                    best_score=score; best_active=active; no_improve=0
                    save_best(model,metrics,active,score)
                else:
                    no_improve += 1
                save_resume(model,opt,active,step,tokens,best_score,best_active,no_improve)
                if active>=MIN_TARGET_TIME and target_met(metrics,baseline):
                    stop_reason="proxy_target_met"
                    print("TARGET MET: stopping adaptive training.",flush=True)
                    break
                if active>=PLATEAU_MIN_TIME and no_improve>=6:
                    stop_reason="plateau"
                    print("PLATEAU: no meaningful improvement for 6 evals; stopping.",flush=True)
                    break
                next_eval += EVAL_EVERY

    final_current,final_details=evaluate_bundle(model,device)
    final_current_score=quality(final_current,baseline)
    if (final_current["must_pass"]>=9 and final_current["rank_accuracy"]>=baseline["rank_accuracy"]-.02
        and final_current["mc_accuracy"]>=baseline["mc_accuracy"]-.015
        and final_current["wiki_loss"]<=baseline["wiki_loss"]*1.04
        and final_current["general_loss"]<=baseline["general_loss"]*1.08
        and final_current_score>best_score+.002):
        best_score=final_current_score; best_active=active
        save_best(model,final_current,active,best_score)

    best=torch.load(best_file,map_location="cpu")
    model.load_state_dict(best["model"]); model=model.to(device); model.lm_head.weight=model.model.embed_tokens.weight
    selected_metrics,selected_details=evaluate_bundle(model,device)
    final_dir=OUT/"final_hf"; final_dir.mkdir(parents=True,exist_ok=True)
    cpu_model=AutoModelForCausalLM.from_pretrained(BASE,device_map=None)
    cpu_model.load_state_dict(best["model"]); cpu_model.lm_head.weight=cpu_model.model.embed_tokens.weight
    cpu_model.save_pretrained(final_dir,safe_serialization=True,max_shard_size="5GB")
    tok.save_pretrained(final_dir)
    run={
      "parameters":params,"active_training_seconds":active,"steps":step,"tokens_seen":tokens,
      "active_tok_per_s":tokens/max(active,1e-6),"stop_reason":stop_reason,
      "selected_checkpoint_active_seconds":float(best["active"]),"selection_quality":float(best["score"]),
      "baseline_metrics":baseline,"end_of_run_metrics":final_current,"selected_metrics":selected_metrics,
      "base_model":"v15_recovery_consolidation/final_hf",
      "policy":"Gradient data uses official training splits, human-written math solutions, local corpora, and deterministic instruction templates; official benchmark validation/test examples are excluded.",
      "recipe":"V16: six-hour phased competition run from V15 selected; phase1 MC/rank recovery, phase2 math/reasoning push, phase3 LM/general benchmark consolidation; low LR + adaptive source weighting + 20-minute anti-regression selection; V15 retained as fallback best"
    }
    (OUT/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
    (OUT/"selected_eval_details.json").write_text(json.dumps(selected_details,indent=2),encoding="utf-8")
    print("V16 FINAL COMPETITION TRAINING COMPLETE",flush=True)
    print(json.dumps(run,indent=2),flush=True)

if __name__=="__main__":
    main()
