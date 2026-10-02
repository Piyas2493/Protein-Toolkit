from core.statistics import SequenceStatistics


def test_statistics_basic_sequence():
    stats = SequenceStatistics("ACDEFGHIKLMNPQRSTVWY")
    assert stats.molecular_weight() > 0
    assert -2.0 <= stats.gravy() <= 2.0


def test_statistics_handles_ambiguous_residues_without_crashing():
    # X/B/Z/J/U/O are valid per SequenceValidator but absent from
    # ProtParam's scale tables — must not KeyError.
    stats = SequenceStatistics("ACDEFXBZJUO")
    assert stats.gravy() is not None
    assert stats.instability_index() is not None
    assert stats.molecular_weight() > 0
