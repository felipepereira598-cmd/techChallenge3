import json
import pytest
from langgraph.types import Command
from src.graph import build_graph
from src.llm import DemoLLM
from src.logger import AuditLogger
from src.safety import anonymize, violations
from src.tools import get_patient

def run(tmp_path, llm=None, patient="P001"):
    graph=build_graph(llm or DemoLLM(),AuditLogger(tmp_path/"audit.jsonl"))
    config={"configurable":{"thread_id":"test"}}
    result=graph.invoke({"request_id":"test","patient_id":patient,"question":"Resuma pendências"},config)
    return graph,config,result

def test_human_gate_and_audit(tmp_path):
    graph,config,result=run(tmp_path)
    assert "__interrupt__" in result
    assert "answer" not in result
    assert result["pending_exams"] == ["radiografia de tórax"]
    final=graph.invoke(Command(resume={"approved":True,"reviewer":"R01","note":"conferido"}),config)
    assert final["status"] == "approved"
    assert final["answer"] == final["draft"]
    events=[json.loads(s) for s in (tmp_path/"audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["load","generate","safety","human_review"]
    assert all(e["request_id"] == "test" for e in events)

def test_rejection(tmp_path):
    graph,config,_=run(tmp_path)
    result=graph.invoke(Command(resume={"approved":False,"reviewer":"R02"}),config)
    assert result["status"] == "rejected"
    assert result["answer"] != result["draft"]

def test_unsafe_output_never_reaches_approval(tmp_path):
    class Unsafe(DemoLLM):
        def generate(self,messages): return "Administre 500 mg agora"
    _,_,result=run(tmp_path,Unsafe())
    assert result["status"] == "blocked"
    assert "answer" not in result and "__interrupt__" not in result

def test_requires_reviewer(tmp_path):
    graph,config,_=run(tmp_path)
    with pytest.raises(ValueError):
        graph.invoke(Command(resume={"approved":True,"reviewer":""}),config)

def test_missing_patient_fails_and_logs(tmp_path):
    with pytest.raises(ValueError): run(tmp_path,patient="UNKNOWN")
    assert '"event": "error"' in (tmp_path/"audit.jsonl").read_text()

def test_alerts_and_completed_exams(tmp_path):
    _,_,result=run(tmp_path,patient="P003")
    assert result["alerts"]
    assert get_patient.invoke({"patient_id":"P002"})["exams"][0]["status"] == "completed"

def test_redaction():
    text=anonymize("email nome@example.com CPF 123.456.789-00 telefone (11) 99999-1234")
    assert "example.com" not in text and "123.456" not in text and "99999" not in text
    assert violations("Take 2 mg")

def test_separate_requests(tmp_path):
    graph,config,_=run(tmp_path)
    other={"configurable":{"thread_id":"other"}}
    graph.invoke({"request_id":"other","patient_id":"P002","question":"Resumo"},other)
    graph.invoke(Command(resume={"approved":False,"reviewer":"R1"}),other)
    assert graph.get_state(config).values["status"] == "pending_review"
