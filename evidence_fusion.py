"""
Cross-evidence fusion (Sprint 12).

Combines evidence from:
    - Sequence pipeline (core/evidence.py) — motifs, catalytic, enzymes,
      functions, ligands, sequence active sites.
    - Structural pipeline (structure/evidence_structure.py) — pocket
      scores, structural active sites, predicted binding contacts.
    - Docking pipeline (docking/) — compatibility verdict, pose ranking,
      active-site overlap.

Each evidence item carries a `provenance` record so downstream tools
can re-derive every contribution. The fusion produces a final
qualitative assessment — not a probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    source: str                  # "sequence", "structure", "docking", "ml", ...
    claim: str
    weight: float = 1.0
    confidence: float = 0.5
    supports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "claim": self.claim,
            "weight": self.weight,
            "confidence": self.confidence,
            "supports": list(self.supports),
            "metadata": dict(self.metadata),
        }


@dataclass
class FusionReport:
    items: List[EvidenceItem] = field(default_factory=list)
    final_score: float = 0.0
    final_assessment: str = "UNKNOWN"
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = (
        "Cross-evidence fusion is a qualitative synthesis. It does NOT "
        "establish experimental binding or biological activity."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "final_score": round(self.final_score, 3),
            "final_assessment": self.final_assessment,
            "confidence_breakdown": dict(self.confidence_breakdown),
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
        }


class EvidenceFusionEngine:
    """Combine evidence items with explicit weights and provenance."""

    def fuse(
        self,
        sequence_items: Optional[List[EvidenceItem]] = None,
        structure_items: Optional[List[EvidenceItem]] = None,
        docking_items: Optional[List[EvidenceItem]] = None,
        ml_items: Optional[List[EvidenceItem]] = None,
    ) -> FusionReport:
        all_items = []
        for bucket in (sequence_items, structure_items, docking_items, ml_items):
            if bucket:
                all_items.extend(bucket)

        if not all_items:
            return FusionReport(
                final_assessment="INSUFFICIENT EVIDENCE",
                warnings=["No evidence was supplied."],
            )

        # Weighted score
        total_w = sum(i.weight for i in all_items)
        total_c = sum(i.weight * i.confidence for i in all_items)
        final_score = (total_c / total_w) if total_w else 0.0

        # Source-level breakdown
        by_source: Dict[str, List[EvidenceItem]] = {}
        for item in all_items:
            by_source.setdefault(item.source, []).append(item)
        breakdown = {
            source: sum(i.confidence for i in items) / len(items)
            for source, items in by_source.items()
        }

        # Verdict
        if final_score >= 0.70:
            verdict = "HIGH-PRIORITY DOCKING CANDIDATE"
        elif final_score >= 0.45:
            verdict = "MODERATE-PRIORITY CANDIDATE"
        elif final_score >= 0.25:
            verdict = "LOW-PRIORITY CANDIDATE"
        else:
            verdict = "INCOMPATIBLE / INSUFFICIENT EVIDENCE"

        warnings = []
        if len(by_source) < 2:
            warnings.append(
                "Fusion relies on a single evidence source; "
                "interpret cautiously."
            )

        return FusionReport(
            items=all_items,
            final_score=final_score,
            final_assessment=verdict,
            confidence_breakdown=breakdown,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Builders for each pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def from_sequence(
        motif_hits=None,
        catalytic_hits=None,
        enzyme_predictions=None,
        active_sites=None,
        sequence_evidence=None,
    ) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        if motif_hits:
            items.append(EvidenceItem(
                source="sequence",
                claim=f"{len(motif_hits)} motif(s) detected",
                confidence=min(1.0, 0.4 + 0.1 * len(motif_hits)),
                metadata={"motif_count": len(motif_hits)},
            ))
        if catalytic_hits:
            items.append(EvidenceItem(
                source="sequence",
                claim=f"{len(catalytic_hits)} catalytic residue prediction(s)",
                confidence=min(1.0, 0.5 + 0.1 * len(catalytic_hits)),
                metadata={"catalytic_count": len(catalytic_hits)},
            ))
        if enzyme_predictions:
            items.append(EvidenceItem(
                source="sequence",
                claim=f"{len(enzyme_predictions)} enzyme class prediction(s)",
                confidence=0.7,
                metadata={"enzyme_classes": enzyme_predictions},
            ))
        if active_sites:
            items.append(EvidenceItem(
                source="sequence",
                claim=f"{len(active_sites)} sequence-based active-site prediction(s)",
                confidence=0.55,
                metadata={"active_sites": active_sites},
            ))
        if sequence_evidence and not any(
            (motif_hits, catalytic_hits, enzyme_predictions, active_sites)
        ):
            # sequence_evidence is the dict produced by core/evidence.py
            items.append(EvidenceItem(
                source="sequence",
                claim=f"Sequence evidence score: {sequence_evidence.get('score', 'n/a')}",
                confidence=min(1.0, 0.5 + 0.005 * float(sequence_evidence.get("score", 0))),
                metadata={"raw": sequence_evidence},
            ))
        return items

    @staticmethod
    def from_structure(
        pocket_scores=None,
        structural_active_sites=None,
        structural_evidence=None,
    ) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        if pocket_scores:
            best = pocket_scores[0]
            items.append(EvidenceItem(
                source="structure",
                claim=f"Top pocket score {best.get('score', 0)}/100",
                confidence=min(1.0, best.get('score', 0) / 100),
                metadata={"pocket": best},
            ))
        if structural_active_sites:
            items.append(EvidenceItem(
                source="structure",
                claim=f"{len(structural_active_sites)} structural active-site prediction(s)",
                confidence=0.6,
                metadata={"sites": structural_active_sites},
            ))
        if structural_evidence:
            items.append(EvidenceItem(
                source="structure",
                claim=f"Structural evidence fusion with {len(structural_evidence)} entry(ies)",
                confidence=0.65,
                metadata={"raw": structural_evidence},
            ))
        return items

    @staticmethod
    def from_docking(
        compatibility_report=None,
        pose_analysis_report=None,
    ) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        if compatibility_report is not None:
            compat_map = {
                "HIGH-PRIORITY DOCKING CANDIDATE": 0.85,
                "MODERATE-PRIORITY CANDIDATE": 0.55,
                "LOW-PRIORITY CANDIDATE": 0.30,
                "INCOMPATIBLE": 0.05,
                "INSUFFICIENT EVIDENCE": 0.10,
            }
            score = compat_map.get(compatibility_report.overall_assessment, 0.4)
            items.append(EvidenceItem(
                source="docking",
                claim=(
                    f"Ligand-pocket compatibility: "
                    f"{compatibility_report.overall_assessment}"
                ),
                confidence=score,
                metadata={"best_pocket": compatibility_report.best_pocket_id},
            ))
        if pose_analysis_report is not None:
            if pose_analysis_report.ranked:
                best = pose_analysis_report.ranked[0]
                items.append(EvidenceItem(
                    source="docking",
                    claim=(
                        f"Best pose {best.pose_id} "
                        f"({best.overall}) score={best.score}"
                    ),
                    confidence={
                        "STRONG": 0.85, "MODERATE": 0.55, "WEAK": 0.25
                    }.get(best.overall, 0.4),
                    metadata={
                        "n_hbonds": best.n_hbonds,
                        "n_hydrophobic": best.n_hydrophobic,
                        "active_overlap": best.active_overlap,
                    },
                ))
        return items