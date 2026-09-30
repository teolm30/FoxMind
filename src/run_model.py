from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

def main():
    ap = argparse.ArgumentParser(description="Run FoxMind 49M locally")
    ap.add_argument("--model", default="checkpoints/v16_final_competition/final_hf")
    ap.add_argument("--prompt", default="What is 17 - 9?")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model folder not found: {model_path}\n"
            "Place the exported V16 Hugging Face model there before running."
        )

    tok = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    prompt = f"User: {args.prompt.strip()}\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
