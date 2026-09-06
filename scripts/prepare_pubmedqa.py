import argparse
import json
import hashlib
from pathlib import Path
from scripts.prepare_medquad import write_jsonl

def prepare(path, output):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows=[]
    for pmid, item in sorted(data.items()):
        label=item.get("final_decision", "").lower()
        if label not in {"yes","no","maybe"}: continue
        # Never include LONG_ANSWER in prompt: it leaks the target conclusion.
        rows.append({"id":pmid, "dataset":"PubMedQA", "source":f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                     "question":item["QUESTION"], "context":"\n".join(item["CONTEXTS"]), "label":label})
    if not rows: raise ValueError("Nenhum exemplo PQA-L válido.")
    write_jsonl(output, rows)
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "dataset":"PubMedQA PQA-L", "upstream":"https://github.com/pubmedqa/pubmedqa",
        "input_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "records":len(rows), "usage":"evaluation only; context excludes LONG_ANSWER"
    }, indent=2), encoding="utf-8")
    return rows

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/processed/pubmedqa_eval.jsonl"))
    a=p.parse_args(); prepare(a.input,a.output)
