from src.it_benchmark import (
    AuditThresholds, DOMAIN_TAGS, audit_benchmark, build_balanced_benchmark,
    domain_hits, normalize_stackoverflow_row, parse_tags,
)
from src.it_dataset_loader import ITSkill


def row(qid, tags, title=None, body=None, **extra):
    title = title or f"Fix implementation {qid}"
    body = body or f"A sufficiently detailed technical question {qid} with code and expected behavior."
    data = {
        "id": qid, "tags": "|".join(tags), "title": title,
        "question_body": body, "score": 10, "view_count": 1000,
        "body_len": len(body), "code_block_cnt": 1, "has_code": True,
    }
    data.update(extra)
    return data


def test_tag_taxonomy_covers_every_skill_and_excludes_ambiguous_authentication():
    assert set(DOMAIN_TAGS) == set(ITSkill)
    assert all(DOMAIN_TAGS[skill] for skill in ITSkill)
    assert "authentication" not in DOMAIN_TAGS[ITSkill.SECURITY]


def test_tags_are_ground_truth_not_prompt_keywords():
    tags = parse_tags("reactjs|postgresql|security")
    assert domain_hits(tags) == (ITSkill.FRONTEND, ITSkill.DATABASES, ITSkill.SECURITY)
    assert domain_hits(("python",)) == ()


def test_multilabel_record_preserves_all_explicit_domains():
    item = normalize_stackoverflow_row(row(1, ["reactjs", "security"]), ITSkill.FRONTEND)
    assert item is not None
    scores = dict(zip(ITSkill, item.skill_vector))
    assert scores[ITSkill.FRONTEND] == 1
    assert scores[ITSkill.SECURITY] == 1
    assert scores[ITSkill.ML_LLM] == 0
    assert item.source_url.endswith("/1")


def test_builder_is_balanced_and_deterministic():
    tags = {
        ITSkill.FRONTEND: "html", ITSkill.BACKEND_API: "django",
        ITSkill.DATABASES: "postgresql", ITSkill.ML_LLM: "machine-learning",
        ITSkill.DEVOPS_CLOUD: "docker", ITSkill.SECURITY: "cryptography",
    }
    rows = []
    qid = 1
    for skill, tag in tags.items():
        for index in range(8):
            rows.append(row(qid, [tag], body=("Detailed problem statement " * (3 + index))))
            qid += 1
    first = build_balanced_benchmark(rows, per_skill=5, seed=9)
    second = build_balanced_benchmark(rows, per_skill=5, seed=9)
    assert [x.id for x in first] == [x.id for x in second]
    counts = {skill: sum(x.primary_skill is skill for x in first) for skill in ITSkill}
    assert set(counts.values()) == {5}


def test_audit_passes_controlled_fixture():
    tags = {
        ITSkill.FRONTEND: "html", ITSkill.BACKEND_API: "django",
        ITSkill.DATABASES: "postgresql", ITSkill.ML_LLM: "machine-learning",
        ITSkill.DEVOPS_CLOUD: "docker", ITSkill.SECURITY: "cryptography",
    }
    records = []
    for offset, (skill, tag) in enumerate(tags.items()):
        for index in range(4):
            item = normalize_stackoverflow_row(row(offset * 10 + index + 1, [tag]), skill)
            assert item is not None
            records.append(item)
    report = audit_benchmark(records)
    assert report["passed"], report["failures"]
    assert report["unique_fraction"] == 1.0
    assert report["tag_purity"] == 1.0


def test_audit_rejects_duplicates_and_imbalance():
    item = normalize_stackoverflow_row(row(1, ["html"]), ITSkill.FRONTEND)
    assert item is not None
    report = audit_benchmark([item, item], AuditThresholds())
    assert not report["passed"]
    assert any("unique_fraction" in failure for failure in report["failures"])




def test_builder_never_reuses_multilabel_question_across_primary_classes():
    rows = []
    for index in range(8):
        rows.append(row(index + 1, ["cryptography", "html"]))
    for index in range(8, 16):
        rows.append(row(index + 1, ["html"]))
    for offset, tag in enumerate(("django", "postgresql", "machine-learning", "docker"), start=20):
        for index in range(8):
            rows.append(row(offset * 100 + index, [tag]))
    records = build_balanced_benchmark(rows, per_skill=5, seed=3)
    source_urls = [item.source_url for item in records]
    assert len(source_urls) == len(set(source_urls))
