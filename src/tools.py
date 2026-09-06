import json
from pathlib import Path
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1]

@tool
def get_patient(patient_id: str) -> dict:
    """Read current synthetic patient observations by exact ID."""
    patients = json.loads((ROOT / "data/synthetic/patients.json").read_text(encoding="utf-8"))
    for patient in patients:
        if patient["id"] == patient_id and patient.get("synthetic") is True:
            return patient
    raise ValueError("Paciente sintético não encontrado.")

@tool
def get_protocol(protocol_id: str) -> dict:
    """Read one synthetic protocol by exact ID; no semantic retrieval."""
    protocols = json.loads((ROOT / "data/synthetic/protocols.json").read_text(encoding="utf-8"))
    for protocol in protocols:
        if protocol["id"] == protocol_id:
            return protocol
    raise ValueError("Protocolo não encontrado.")
