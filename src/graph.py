from typing import TypedDict
from time import perf_counter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from src.tools import get_patient, get_protocol
from src.safety import violations, validate_question, anonymize
from src.llm import build_chain

class State(TypedDict, total=False):
    request_id: str
    patient_id: str
    question: str
    patient: dict
    protocol: dict
    pending_exams: list
    alerts: list
    sources: list
    draft: str
    flags: list
    status: str
    reviewer: str
    review_note: str
    answer: str

def build_graph(llm, audit):
    chain = build_chain(llm)
    def observed(name, function):
        def execute(state):
            start = perf_counter()
            try:
                result = function(state)
            except Exception as exc:
                audit.log(state["request_id"], "error", node=name, error_type=type(exc).__name__)
                raise
            audit.log(state["request_id"], name, duration_ms=round((perf_counter()-start)*1000, 2),
                      model=llm.identity, prompt_version="1", output=result)
            return result
        return execute
    def load(state):
        question = validate_question(state["question"])
        patient = get_patient.invoke({"patient_id": state["patient_id"]})
        protocol = get_protocol.invoke({"protocol_id": patient["protocol_id"]})
        return {"question": question, "patient": patient, "protocol": protocol,
                "pending_exams": [e["name"] for e in patient["exams"] if e["status"] == "pending"],
                "alerts": ["Relato de falta de ar: requer avaliação humana (regra fictícia RESP-SIM-001 §3)."] if "falta de ar referida" in patient["symptoms"] else [],
                "sources": [{"id": patient["id"], "path": "data/synthetic/patients.json", "updated_at": patient["updated_at"]},
                            {"id": protocol["id"], "path": "data/synthetic/protocols.json", "version": protocol["version"]}],
                "status": "loaded"}
    def generate(state):
        return {"draft": anonymize(chain.invoke(state)), "status": "draft"}
    def safety(state):
        flags = violations(state["draft"])
        return {"flags": flags, "status": "blocked" if flags else "pending_review"}
    def review(state):
        decision = interrupt({"draft": state["draft"], "sources": state["sources"],
                              "notice": "Revisão humana obrigatória; confirme fatos e fontes."})
        if not isinstance(decision, dict) or not str(decision.get("reviewer", "")).strip():
            raise ValueError("Revisor obrigatório.")
        if type(decision.get("approved")) is not bool:
            raise ValueError("Decisão inválida.")
        accepted = decision["approved"] is True
        result = {"status": "approved" if accepted else "rejected",
                  "reviewer": anonymize(str(decision["reviewer"])),
                  "review_note": anonymize(str(decision.get("note", ""))),
                  "answer": state["draft"] if accepted else "Resposta rejeitada na revisão humana."}
        audit.log(state["request_id"], "human_review", **result)
        return result
    graph = StateGraph(State)
    for name, fn in [("load", load), ("generate", generate), ("safety", safety)]:
        graph.add_node(name, observed(name, fn))
    graph.add_node("human_review", review)
    graph.add_edge(START, "load")
    graph.add_edge("load", "generate")
    graph.add_edge("generate", "safety")
    graph.add_conditional_edges("safety", lambda s: "blocked" if s["flags"] else "review",
                                {"blocked": END, "review": "human_review"})
    graph.add_edge("human_review", END)
    return graph.compile(checkpointer=InMemorySaver())
