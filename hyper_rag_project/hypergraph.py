"""
hypergraph.py
─────────────
Core data structures for an epistemically-scored knowledge hypergraph.

A hyperedge connects N nodes (entities/concepts) and carries:
  - a claim string  (the proposition it encodes)
  - provenance      (which passages support it)
  - an EpistemicScore across 9 dimensions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional
from enum import Enum
import math


# ─── Evidence-type taxonomy ──────────────────────────────────────────────────

class EvidenceType(Enum):
    """Ordered by methodological strength (higher = stronger)."""
    SECONDARY_COMMENTARY  = 1   # review without primary analysis, editorial
    CASE_REPORT           = 2   # single patient / single observation
    CROSS_SECTIONAL       = 3   # one-time snapshot study
    LONGITUDINAL          = 4   # repeated measures, no control
    OBSERVATIONAL         = 5   # epidemiological, controlled confounders
    CONTROLLED_EXPERIMENT = 6   # RCT or controlled lab experiment
    META_ANALYSIS         = 7   # systematic synthesis of multiple studies

    def strength(self) -> float:
        """Normalised to [0, 1]."""
        return (self.value - 1) / (max(e.value for e in EvidenceType) - 1)


# ─── Epistemic Score ──────────────────────────────────────────────────────────

@dataclass
class EpistemicScore:
    """
    9-dimensional evidence quality score, each dimension in [0, 1].

    Dimensions follow the "Not All Evidence Is Equal" framework.
    """
    source_authority:       float = 0.5   # journal impact, author credentials
    methodological_strength:float = 0.5   # EvidenceType.strength()
    directness:             float = 0.5   # does evidence address the claim directly?
    specificity:            float = 0.5   # exact conditions vs general version
    sample_adequacy:        float = 0.5   # sample size / statistical power
    independence:           float = 0.5   # are corroborating sources independent?
    consistency:            float = 0.5   # agrees (1.0) or contradicts (0.0) consensus
    replication_status:     float = 0.5   # how often reproduced
    recency:                float = 0.5   # age-adjusted relevance

    # ── Dimension weights (tunable) ──
    WEIGHTS: dict = field(default_factory=lambda: {
        "source_authority":        0.10,
        "methodological_strength": 0.20,
        "directness":              0.15,
        "specificity":             0.10,
        "sample_adequacy":         0.10,
        "independence":            0.10,
        "consistency":             0.10,
        "replication_status":      0.10,
        "recency":                 0.05,
    })

    def aggregate(self) -> float:
        """Weighted linear aggregation → scalar in [0, 1]."""
        dims = {
            "source_authority":        self.source_authority,
            "methodological_strength": self.methodological_strength,
            "directness":              self.directness,
            "specificity":             self.specificity,
            "sample_adequacy":         self.sample_adequacy,
            "independence":            self.independence,
            "consistency":             self.consistency,
            "replication_status":      self.replication_status,
            "recency":                 self.recency,
        }
        return sum(self.WEIGHTS[k] * v for k, v in dims.items())

    def vector(self) -> list[float]:
        return [
            self.source_authority, self.methodological_strength,
            self.directness, self.specificity, self.sample_adequacy,
            self.independence, self.consistency, self.replication_status,
            self.recency,
        ]

    def dimension_names(self) -> list[str]:
        return [
            "source_authority", "methodological_strength", "directness",
            "specificity", "sample_adequacy", "independence",
            "consistency", "replication_status", "recency",
        ]

    @classmethod
    def from_evidence_type(
        cls,
        etype: EvidenceType,
        source_authority: float = 0.5,
        sample_n: Optional[int] = None,
        year: Optional[int] = 2020,
        directness: float = 0.8,
        specificity: float = 0.7,
        independence: float = 0.7,
        consistency: float = 0.8,
        replication: float = 0.5,
    ) -> "EpistemicScore":
        """Convenience constructor: derive most dimensions from EvidenceType."""
        # sample adequacy: log-scaled, capped at 1.0
        if sample_n is not None:
            sample_adequacy = min(1.0, math.log10(max(sample_n, 1)) / 4.0)
        else:
            sample_adequacy = etype.strength() * 0.6

        # recency: decay over 10 years from current year
        import datetime
        current_year = datetime.date.today().year
        age = max(0, current_year - (year or current_year))
        recency = max(0.0, 1.0 - age / 10.0)

        return cls(
            source_authority=source_authority,
            methodological_strength=etype.strength(),
            directness=directness,
            specificity=specificity,
            sample_adequacy=sample_adequacy,
            independence=independence,
            consistency=consistency,
            replication_status=replication,
            recency=recency,
        )


# ─── Passage (document chunk) ─────────────────────────────────────────────────

@dataclass
class Passage:
    id: str
    text: str
    source: str                      # e.g. "Nature Neuroscience 2023"
    evidence_type: EvidenceType
    year: int
    sample_n: Optional[int] = None
    entities: List[str] = field(default_factory=list)   # pre-extracted


# ─── Nodes and Hyperedges ─────────────────────────────────────────────────────

@dataclass
class Node:
    """A concept / entity in the hypergraph."""
    id: str
    label: str
    node_type: str    # e.g. "protein", "brain_region", "behaviour", "disease"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id


@dataclass
class Hyperedge:
    """
    A hyperedge connects an arbitrary set of nodes via a claim,
    supported by one or more passages, and annotated with an
    EpistemicScore.
    """
    id: str
    nodes: FrozenSet[Node]
    claim: str
    passages: List[Passage]
    score: EpistemicScore

    def epistemic_quality(self) -> float:
        return self.score.aggregate()

    def node_labels(self) -> list[str]:
        return sorted(n.label for n in self.nodes)


# ─── Hypergraph ───────────────────────────────────────────────────────────────

class KnowledgeHypergraph:
    """
    Container for nodes and hyperedges.

    Provides:
      - add_node / add_edge
      - neighbours(node): all nodes co-occurring in any hyperedge
      - edges_for_nodes(node_ids): hyperedges containing all given nodes
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Hyperedge] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Hyperedge):
        self.edges[edge.id] = edge
        for n in edge.nodes:
            self.nodes[n.id] = n

    def neighbours(self, node_id: str) -> set[Node]:
        result = set()
        for edge in self.edges.values():
            ids = {n.id for n in edge.nodes}
            if node_id in ids:
                result.update(edge.nodes)
        result.discard(self.nodes.get(node_id))
        return result

    def edges_containing(self, node_id: str) -> list[Hyperedge]:
        return [e for e in self.edges.values()
                if any(n.id == node_id for n in e.nodes)]

    def stats(self) -> dict:
        avg_q = (sum(e.epistemic_quality() for e in self.edges.values())
                 / max(len(self.edges), 1))
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "avg_epistemic_quality": round(avg_q, 3),
        }
