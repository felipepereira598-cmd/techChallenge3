import json
from pathlib import Path
from scripts.prepare_medquad import prepare
from scripts.prepare_pubmedqa import prepare as prepare_pubmed
from scripts.evaluate import classify, token_f1
from scripts.train import encode_example

def test_xml_curation_and_group_split(tmp_path):
    raw=tmp_path/"raw"; raw.mkdir()
    (raw/"one.xml").write_text('<Document url="https://example.org/1"><QAPairs><QAPair><Question qtype="symptoms">What symptoms are listed?</Question><Answer>This is a sufficiently long synthetic answer for the fixture.</Answer></QAPair><QAPair><Question>What symptoms are listed?</Question><Answer>This duplicated question should be removed during processing.</Answer></QAPair><QAPair><Question>What observations are listed?</Question><Answer>A second sufficiently long answer from the same document.</Answer></QAPair><QAPair><Question>Missing answer?</Question><Answer /></QAPair></QAPairs></Document>')
    out=tmp_path/"out"; rows=prepare(raw,out)
    assert len(rows)==2 and len({r["split"] for r in rows})==1
    assert prepare(raw,tmp_path/"out2")==rows
    manifest=json.loads((out/"manifest.json").read_text())
    assert manifest["counts"]["duplicate_question"]==1

def test_pubmed_label_not_in_context(tmp_path):
    path=tmp_path/"pqa.json"
    path.write_text(json.dumps({"123":{"QUESTION":"Question?","CONTEXTS":["Abstract"],"LONG_ANSWER":"SECRET TARGET","final_decision":"yes"}}))
    rows=prepare_pubmed(path,tmp_path/"eval.jsonl")
    assert "SECRET TARGET" not in rows[0]["context"]
    assert rows[0]["label"] == "yes"

def test_metrics():
    assert classify("yes.")=="yes"
    assert classify("yes or no")=="invalid"
    assert token_f1("a b","a b")==1
    assert token_f1("x","y")==0

def test_synthetic_training_coverage():
    rows=[json.loads(s) for s in Path("data/synthetic/training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"protocol","faq","report_template","prescription_template","procedure_template"} <= {r["category"] for r in rows}

def test_loss_mask_and_oversize():
    class Tokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            return "".join(m["content"] for m in messages)
        def __call__(self,text,add_special_tokens): return {"input_ids":[ord(c) for c in text]}
    row={"messages":[{"content":"prompt"},{"content":"answer"}]}
    encoded=encode_example(row,Tokenizer(),100)
    assert encoded["labels"][:6]==[-100]*6
    assert encoded["labels"][6:]==[ord(c) for c in "answer"]
    assert encode_example(row,Tokenizer(),5) is None
