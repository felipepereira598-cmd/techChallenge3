import os
from uuid import uuid4
import streamlit as st
from langgraph.types import Command
from src.llm import LocalLLM, DemoLLM
from src.graph import build_graph
from src.logger import AuditLogger

st.set_page_config(page_title="Assistente hospitalar acadêmico", page_icon="🏥")
st.title("Assistente hospitalar acadêmico")
st.warning("Somente dados sintéticos. Não utilizar para atendimento real.")
mode = st.sidebar.selectbox("Modelo", ["Demonstração sem LLM", "Fine-tuned local"])
adapter = os.getenv("ADAPTER_PATH", "models/adapter")
key = (mode, adapter)
if st.session_state.get("model_key") != key:
    try:
        with st.spinner("Preparando assistente..."):
            llm = DemoLLM() if mode.startswith("Demonstração") else LocalLLM(adapter=adapter)
            st.session_state.graph = build_graph(llm, AuditLogger())
            st.session_state.model_key = key
            st.session_state.pop("config", None)
    except Exception as exc:
        st.error(str(exc)); st.stop()
with st.form("request"):
    patient_id = st.selectbox("Paciente sintético", ["P001", "P002", "P003"])
    question = st.text_area("Pergunta", "Resuma as observações e as pendências para revisão médica.")
    submit = st.form_submit_button("Analisar")
if submit:
    rid = str(uuid4())
    config = {"configurable": {"thread_id": rid}}
    try:
        st.session_state.graph.invoke({"request_id": rid, "patient_id": patient_id, "question": question}, config)
        st.session_state.config = config
    except Exception as exc:
        st.error(str(exc)); st.session_state.pop("config", None)
if "config" in st.session_state:
    graph, config = st.session_state.graph, st.session_state.config
    state = graph.get_state(config).values
    st.write("Status:", state["status"])
    for alert in state.get("alerts", []):
        st.error(alert)
    st.json({"observações": state["patient"], "exames_pendentes": state["pending_exams"], "fontes_utilizadas": state["sources"]})
    st.caption("Fontes são os registros fornecidos ao modelo; não provam que cada afirmação foi sustentada. O revisor deve conferir.")
    if state["status"] == "blocked":
        st.error("Rascunho bloqueado pelas regras: " + ", ".join(state["flags"]))
    elif state["status"] == "pending_review":
        st.subheader("Rascunho para revisão")
        st.write(state["draft"])
        with st.form("review"):
            reviewer = st.text_input("Identificador do revisor (sem dados pessoais)")
            note = st.text_area("Justificativa / observações")
            approved = st.checkbox("Conferi o conteúdo e as fontes e aprovo este rascunho")
            if st.form_submit_button("Registrar decisão"):
                if not reviewer.strip():
                    st.error("Informe o identificador do revisor.")
                else:
                    graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer, "note": note}), config)
                    st.rerun()
    else:
        st.write(state.get("answer", ""))
    st.caption("Auditoria local: logs/audit.jsonl. A revisão é uma simulação, sem autenticação de profissionais.")
