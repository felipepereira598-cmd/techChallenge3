"""Assistant-only SFT loss with Transformers Trainer and PEFT LoRA/QLoRA."""
import argparse
import hashlib
import json
from pathlib import Path
from src.llm import MODEL_ID

def encode_example(row, tokenizer, max_length):
    messages=row["messages"]
    prompt=tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix=tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens=tokenizer(full, add_special_tokens=False)["input_ids"]
    if tokens[:len(prefix)] != prefix:
        raise ValueError("Chat template não permite máscara segura do prompt.")
    if len(tokens)>max_length or len(tokens)<=len(prefix): return None
    return {"input_ids":tokens, "attention_mask":[1]*len(tokens),
            "labels":[-100]*len(prefix)+tokens[len(prefix):]}

def main(args):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.pad_token=tokenizer.eos_token
    def dataset(paths):
        rows=[json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        encoded=[encode_example(r,tokenizer,args.max_length) for r in rows]
        valid=[r for r in encoded if r is not None]
        if not valid: raise ValueError("Dataset vazio após tokenização; aumente max-length ou revise os dados.")
        return Dataset.from_list(valid), {"input":len(rows),"retained":len(valid),"dropped":len(rows)-len(valid)}
    train,train_counts=dataset([args.train,args.synthetic]); val,val_counts=dataset([args.validation])
    kwargs={"revision":args.revision,"torch_dtype":torch.float16 if torch.cuda.is_available() else torch.float32}
    if args.qlora:
        if not torch.cuda.is_available(): raise ValueError("QLoRA requer CUDA no Colab.")
        kwargs.update(device_map={"":0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
    model=AutoModelForCausalLM.from_pretrained(args.model,**kwargs)
    if args.qlora: model=prepare_model_for_kbit_training(model)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    model.config.use_cache=False
    def collate(batch):
        length=max(len(r["input_ids"]) for r in batch)
        return {key:torch.tensor([r[key]+[pad]*(length-len(r[key])) for r in batch])
                for key,pad in [("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100)]}
    trainer=Trainer(model=model,train_dataset=train,eval_dataset=val,data_collator=collate,
        args=TrainingArguments(output_dir=str(args.output),num_train_epochs=1,max_steps=args.max_steps,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=2e-4,fp16=torch.cuda.is_available(),logging_steps=10,eval_strategy="epoch",
            save_strategy="epoch",report_to="none",seed=42))
    train_result=trainer.train(); evaluation=trainer.evaluate()
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
    files=[args.train,args.synthetic,args.validation]
    manifest={"model":args.model,"requested_revision":args.revision,"resolved_revision":model.config._commit_hash,
              "qlora":args.qlora,"train_counts":train_counts,"validation_counts":val_counts,
              "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "training":train_result.metrics,"validation":evaluation}
    (args.output/"training_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default=MODEL_ID); p.add_argument("--revision",default="main")
    p.add_argument("--train",type=Path,default=Path("data/processed/medquad_train.jsonl"))
    p.add_argument("--validation",type=Path,default=Path("data/processed/medquad_validation.jsonl"))
    p.add_argument("--synthetic",type=Path,default=Path("data/synthetic/training.jsonl"))
    p.add_argument("--output",type=Path,default=Path("models/adapter")); p.add_argument("--qlora",action="store_true")
    p.add_argument("--max-length",type=int,default=1024); p.add_argument("--max-steps",type=int,default=-1)
    main(p.parse_args())
