import json

from server import handle_smart_task


def test_smart_task_rdkit_insights_present():
    instruction = (
        "We want to pre-shape RDKit linkers before Stage III by enforcing a minimum 91–94 span "
        "using ETKDG bounds. Recommend defaults: minSpan and optional maxSpan around the POI–E3 "
        "anchor gap (~8.55 Å). How many attempts per linker? How many conformers to keep? Also "
        "advise whether to add hydrogens for embedding or keep as-is for PROTAC flow. Consider "
        "performance for 5-linker 300s diag."
    )
    context = (
        "Stage III anchor gap is ~8.55 Å (KEAP1 case). Scripts: reembed_linkers_minspan.py and "
        "filter_linkers_by_span.py"
    )
    out = handle_smart_task({"instruction": instruction, "context": context, "dry_run": True}, None)
    data = json.loads(out)
    assert "insights" in data and data["insights"]
    rec = data["insights"]["recommendations"]
    assert 9.0 <= rec["minSpanAngstrom"] <= 9.5
    assert rec["addHydrogensBeforeEmbedding"] is True
    assert rec["keepTopConformers"] in (3,4,5)

