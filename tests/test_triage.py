from pathlib import Path

from triage import run_triage


def test_triage_ranks_sequences_without_requiring_structure(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(
        ">protease_like\n"
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWW"
        "CNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL\n"
        ">tiny\n"
        "ACDEFGHIK\n",
        encoding="utf-8",
    )

    rows = run_triage(str(fasta), ml_model_path=None)

    assert len(rows) == 2
    assert all(r["valid"] for r in rows)
    # Ranked descending by triage_score.
    assert rows[0]["triage_score"] >= rows[1]["triage_score"]


def test_triage_handles_ambiguous_residues_as_valid(tmp_path):
    # X/Z are ambiguous, not unsupported — config.py's supported +
    # ambiguous sets cover all 26 letters, so this must still process.
    fasta = tmp_path / "ambiguous.fasta"
    fasta.write_text(">ambiguous\nACDEFXXXZZZ\n", encoding="utf-8")

    rows = run_triage(str(fasta), ml_model_path=None)

    assert rows[0]["valid"] is True
