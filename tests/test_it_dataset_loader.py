import json
import pytest
from src.it_dataset_loader import DATASET_SOURCES, ITSkill, TaskKind, infer_skill_vector, normalize_record, normalize_records, write_jsonl


def test_every_skill_has_an_open_source():
    assert set(DATASET_SOURCES) == set(ITSkill)
    assert all(x.license and x.homepage.startswith("https://") for x in DATASET_SOURCES.values())


@pytest.mark.parametrize("skill", list(ITSkill))
def test_primary_skill_is_always_one(skill):
    vector = infer_skill_vector("Build a secure REST API backed by PostgreSQL", skill)
    assert len(vector) == len(ITSkill)
    assert vector[list(ITSkill).index(skill)] == 1.0
    assert all(0.0 <= score <= 1.0 for score in vector)


def test_multilabel_overlap_is_preserved():
    vector = infer_skill_vector("Deploy a secure FastAPI service with PostgreSQL using Docker and Kubernetes", ITSkill.BACKEND_API)
    scores = dict(zip(ITSkill, vector))
    assert scores[ITSkill.BACKEND_API] == 1.0
    assert scores[ITSkill.DATABASES] > 0
    assert scores[ITSkill.DEVOPS_CLOUD] > 0
    assert scores[ITSkill.SECURITY] > 0


def test_normalizes_instruction_schema():
    source = DATASET_SOURCES[ITSkill.FRONTEND]
    item = normalize_record({"instruction": "Build a responsive React navigation component.", "input": "Use accessible markup.", "output": "export function Navigation() { return null; }"}, source, 7)
    assert item is not None
    assert item.task_kind is TaskKind.CODE_GENERATION
    assert "Context:" in item.prompt
    assert item.metadata["row_index"] == 7


def test_normalizes_sql_schema():
    source = DATASET_SOURCES[ITSkill.DATABASES]
    item = normalize_record({"question": "List all users created today.", "context": "CREATE TABLE users(id INT, created_at TIMESTAMP);", "answer": "SELECT * FROM users WHERE created_at::date = CURRENT_DATE;"}, source, 0)
    assert item is not None
    assert "CREATE TABLE" in item.prompt


def test_normalizes_glaive_chat_schema():
    source = DATASET_SOURCES[ITSkill.BACKEND_API]
    item = normalize_record({"chat": "SYSTEM: Use functions.\nUSER: Call the weather API for London.\nASSISTANT: <functioncall>{\"city\": \"London\"}"}, source, 0)
    assert item is not None
    assert item.prompt == "Call the weather API for London."
    assert "functioncall" in item.reference_answer


def test_invalid_and_duplicate_rows_are_removed():
    source = DATASET_SOURCES[ITSkill.SECURITY]
    rows = [{"instruction": "Explain how to remediate SQL injection safely.", "output": "Use parameters."}, {"instruction": "Explain how to remediate SQL injection safely.", "output": "Duplicate."}, {"instruction": "short", "output": "answer"}]
    assert len(normalize_records(rows, source)) == 1


def test_jsonl_export_has_stable_schema(tmp_path):
    source = DATASET_SOURCES[ITSkill.ML_LLM]
    item = normalize_record({"prompt": "Train a scikit-learn classifier and report validation accuracy.", "solution": "model.fit(X_train, y_train)"}, source, 3)
    assert item is not None
    output = tmp_path / "tasks.jsonl"
    write_jsonl([item], output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["primary_skill"] == ITSkill.ML_LLM.value
    assert set(payload["skill_scores"]) == {skill.value for skill in ITSkill}
    assert payload["source_license"] == source.license
