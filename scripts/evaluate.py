"""Identical held-out prompts for base and adapter; no fabricated results."""
import argparse
import gc
import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from src.llm import LocalLLM, MODEL_ID
from src.safety import violations
from scripts.prepare_medquad import write_jsonl

def token_f1(prediction, reference):
    a=Counter(re.findall(r"\w+",prediction.lower())); b=Counter(re.findall(r"\w+",reference.lower()))
    overlap=sum((a & b).values())
    return 2*overlap/(sum(a.values())+sum(b.values())) if a and b else 0.0

def classify(text):
    match=re.fullmatch(r"\s*(yes|no|maybe)[.!]?\s*",text.lower())
    return match.group(1) if match else "invalid"

def main(args):
    import torch
    manifest=json.loads((args.adapter/"training_manifest.json").read_text(encoding="utf-8"))
    args.model=manifest["model"]
    args.revision=manifest.get("resolved_revision") or manifest["requested_revision"]
    cases=[json.loads(s) for s in args.medquad.read_text(encoding="utf-8").splitlines()][:args.limit]
    pubmed=[json.loads(s) for s in args.pubmedqa.read_text(encoding="utf-8").splitlines()][:args.limit]
    if not cases or not pubmed: raise ValueError("Ambos os conjuntos de avaliação devem conter exemplos.")
    summary={}
    for name,adapter in [("base",None),("fine_tuned",str(args.adapter))]:
        print(f"[{name}] Carregando modelo...", flush=True)
        llm=LocalLLM(adapter,model_id=args.model,revision=args.revision,quantized=args.quantized)
        outputs=[]
        for index, case in enumerate(cases, 1):
            print(f"[{name}] MedQuAD {index}/{len(cases)}", flush=True)
            response=llm.generate(case["messages"][:-1])
            outputs.append({"id":case["id"],"dataset":"MedQuAD","prediction":response,"question":case["messages"][-2]["content"],
                            "reference":case["messages"][-1]["content"],"source":case["source"],
                            "token_f1":token_f1(response,case["messages"][-1]["content"]),"safety_flags":violations(response)})
        for index, case in enumerate(pubmed, 1):
            print(f"[{name}] PubMedQA {index}/{len(pubmed)}", flush=True)
            response=llm.generate([{"role":"system","content":"Answer with exactly yes, no, or maybe based on the abstract."},
                {"role":"user","content":case["context"]+"\nQuestion: "+case["question"]}],max_new_tokens=8)
            outputs.append({"id":case["id"],"dataset":"PubMedQA","prediction":response,"question":case["question"],"context":case["context"],
                            "label":case["label"],"predicted_label":classify(response),"source":case["source"]})
        med=[o for o in outputs if o["dataset"]=="MedQuAD"]; pub=[o for o in outputs if o["dataset"]=="PubMedQA"]
        summary[name]={"model":llm.identity,"medquad_n":len(med),"pubmedqa_n":len(pub),
            "inputs_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.medquad,args.pubmedqa]},
            "medquad_token_f1":sum(o["token_f1"] for o in med)/len(med),
            "medquad_rule_violation_rate":sum(bool(o["safety_flags"]) for o in med)/len(med),
            "pubmedqa_accuracy":sum(o["predicted_label"]==o["label"] for o in pub)/len(pub),
            "pubmedqa_invalid_rate":sum(o["predicted_label"]=="invalid" for o in pub)/len(pub)}
        write_jsonl(args.output/f"{name}.jsonl",outputs)
        print(f"[{name}] Resultados salvos em {args.output / (name + '.jsonl')}", flush=True)
        del llm; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (args.output/"comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--adapter",type=Path,default=Path("models/adapter"))
    p.add_argument("--model",default=MODEL_ID); p.add_argument("--revision",default="main")
    p.add_argument("--medquad",type=Path,default=Path("data/processed/medquad_test.jsonl"))
    p.add_argument("--pubmedqa",type=Path,default=Path("data/processed/pubmedqa_eval.jsonl"))
    p.add_argument("--output",type=Path,default=Path("results")); p.add_argument("--limit",type=int,default=50)
    p.add_argument("--quantized",action="store_true"); main(p.parse_args())
