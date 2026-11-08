from core.catalytic import CatalyticPredictor

# Bovine trypsin (PDB 3PTB, a serine protease) and hen egg-white lysozyme (PDB 1LYZ,
# a glycosidase). The old composition-only rule called BOTH of them a serine,
# cysteine and aspartic protease and a metalloprotease.
TRYPSIN = ("IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKL"
           "KSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPV"
           "VCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN")
LYSOZYME = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASV"
            "NCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")


def names(sequence):
    return {hit["name"] for hit in CatalyticPredictor().predict(sequence)}


def test_single_residue_does_not_satisfy_duplicate_requirement():
    # One ASP and one HIS must not satisfy rules needing two of a kind
    # (Aspartic Protease needs 2x ASP, Metalloprotease needs 2x HIS).
    assert "Aspartic Protease" not in names("AAAAADAAAAAHAAAAA")
    assert "Metalloprotease" not in names("AAAAADAAAAAHAAAAA")


def test_residue_composition_alone_is_not_a_catalytic_system():
    # Two Asp, a His, a Ser and a Cys scattered through a sequence satisfy every
    # composition requirement and carry no active-site motif: nothing should fire.
    assert names("AAADAAADAAAHAAASAAACAAAHAAAEAAAHAAA") == set()


def test_conserved_motif_with_the_required_residues_fires_with_evidence():
    hits = CatalyticPredictor().predict("AAALLDTGSAAAAADAAA")      # D-T-G core + a second Asp
    (hit,) = [h for h in hits if h["name"] == "Aspartic Protease"]
    assert hit["evidence"][0]["sequence"] == "LLDTGS" and hit["evidence"][0]["start"] == 4


def test_real_serine_protease_gets_only_the_serine_rule():
    hits = CatalyticPredictor().predict(TRYPSIN)
    assert {h["name"] for h in hits} == {"Serine Protease Triad"}
    motifs = {e["motif"] for h in hits for e in h["evidence"]}
    assert any("PS00134" in m for m in motifs) and any("PS00135" in m for m in motifs)


def test_a_non_protease_gets_no_catalytic_hit():
    assert names(LYSOZYME) == set()


def test_rules_without_a_motif_list_keep_composition_behaviour():
    predictor = CatalyticPredictor()
    predictor.rules = [{"name": "Custom", "residues": ["ASP", "ASP"], "pattern": "D-D",
                        "function": "x", "enzyme_class": "Hydrolase", "confidence": 0.5}]
    (hit,) = predictor.predict("AAADAAADAAA")
    assert hit["name"] == "Custom" and hit["evidence"] == []
