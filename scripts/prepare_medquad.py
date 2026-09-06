"""Curate XML QA, deduplicate questions and split by source document."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from collections import Counter
from defusedxml import ElementTree as ET
from src.safety import SYSTEM, anonymize

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")

def prepare(raw, output, limit=3000, seed=42):
    rows, seen, counts = [], set(), Counter()
    for path in sorted(raw.rglob("*.xml")):
        if path.relative_to(raw).parts[0].startswith(("10_", "11_", "12_")):
            counts["excluded_collection"] += 1; continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            counts["invalid_xml"] += 1; continue
        source = root.attrib.get("url", str(path.relative_to(raw)))
        for pair in root.findall(".//QAPair"):
            qnode, anode = pair.find("Question"), pair.find("Answer")
            q = " ".join("".join(qnode.itertext()).split()) if qnode is not None else ""
            a = " ".join("".join(anode.itertext()).split()) if anode is not None else ""
            if len(q) < 10 or len(a) < 30 or len(a) > 6000:
                counts["invalid_length"] += 1; continue
            q, a = anonymize(q), anonymize(a)
            normalized = q.casefold()
            if normalized in seen:
                counts["duplicate_question"] += 1; continue
            seen.add(normalized)
            # All QA from the same document stay in one partition.
            bucket = int(hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
            rows.append({"id": hashlib.sha256((source+q).encode()).hexdigest()[:20],
                         "source": source, "dataset": "MedQuAD", "split": split,
                         "question_type": qnode.attrib.get("qtype", "unknown"),
                         "messages": [{"role":"system", "content":SYSTEM},
                                      {"role":"user", "content":q}, {"role":"assistant", "content":a}]})
    random.Random(seed).shuffle(rows)
    rows = rows[:limit]
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"medquad_{split}.jsonl", [r for r in rows if r["split"] == split])
    try:
        revision = subprocess.check_output(["git", "-C", str(raw), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unversioned-input"
    manifest = {"dataset":"MedQuAD", "upstream":"https://github.com/abachaa/MedQuAD",
                "revision":revision, "seed":seed, "counts":dict(counts),
                "splits":dict(Counter(r["split"] for r in rows)),
                "curation":"Length filters, excluded copyrighted collections, normalized question deduplication, document split; manual review still required."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/processed")); p.add_argument("--limit", type=int, default=3000)
    a=p.parse_args(); prepare(a.raw,a.output,a.limit)
