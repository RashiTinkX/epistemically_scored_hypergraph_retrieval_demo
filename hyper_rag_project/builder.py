"""
builder.py
Constructs a KnowledgeHypergraph from a list of Passages.

Each passage → one Hyperedge whose:
  - nodes   = the passage's entity set
  - claim   = first sentence of the passage text (heuristic)
  - score   = EpistemicScore derived from metadata

Passages sharing entities are thus linked through shared nodes,
forming the hypergraph structure.
"""

from __future__ import annotations
from typing import List
import re

from hypergraph import (
    KnowledgeHypergraph, Node, Hyperedge,
    Passage, EpistemicScore,
)

# Node-type lookup (extend as needed)
NODE_TYPE_MAP = {
    "BDNF":                   "protein",
    "TrkB":                   "protein",
    "TREM2":                  "protein",
    "amyloid":                "protein",
    "dopamine":               "neurotransmitter",
    "LTP":                    "process",
    "synaptic_plasticity":    "process",
    "neuroinflammation":      "process",
    "reinforcement_learning": "process",
    "memory_consolidation":   "process",
    "sharp_wave_ripples":     "process",
    "reward_prediction_error":"process",
    "hippocampus":            "brain_region",
    "VTA":                    "brain_region",
    "striatum":               "brain_region",
    "spatial_memory":         "behaviour",
    "episodic_memory":        "behaviour",
    "declarative_memory":     "behaviour",
    "decision_making":        "behaviour",
    "cognitive_decline":      "behaviour",
    "aerobic_exercise":       "intervention",
    "sleep":                  "intervention",
    "microglia":              "cell_type",
    "Alzheimers_disease":     "disease",
}


def _source_authority(source: str) -> float:
    """Rough heuristic mapping journal prestige → [0, 1]."""
    tier1 = {"Nature", "Science", "Cell", "Neuron", "JAMA Neurology"}
    tier2 = {"Nature Neuroscience", "Alzheimer's & Dementia",
             "Neuroscience & Biobehavioral Reviews", "Sleep"}
    tier3 = {"Neuropsychologia", "Journal of Neuroinflammation",
             "Frontiers in Aging Neuroscience"}
    tier4 = {"Trends in Neurosciences (editorial)",
             "Scientific American (blog post)"}

    for t, score in [(tier1, 1.0), (tier2, 0.8), (tier3, 0.6), (tier4, 0.3)]:
        if any(j in source for j in t):
            return score
    return 0.5


def _independence(passage: Passage, all_passages: List[Passage]) -> float:
    """
    Estimate independence: passages sharing many entities with the current
    one are likely from the same research group / same study → lower independence.
    We reward edges that have unique entity combinations.
    """
    own = set(passage.entities)
    overlaps = []
    for p in all_passages:
        if p.id == passage.id:
            continue
        shared = len(own & set(p.entities)) / max(len(own), 1)
        overlaps.append(shared)
    if not overlaps:
        return 1.0
    avg_overlap = sum(overlaps) / len(overlaps)
    return round(1.0 - avg_overlap * 0.5, 3)   # max penalty 0.5


def _consistency(passage: Passage, all_passages: List[Passage]) -> float:
    """
    Heuristic: contradictory passages share entities but have opposing signal.
    We flag passages from the same entity cluster where evidence_type differs
    strongly; manual labels for demo clarity.
    """
    # Override for known contradictory pairs in our corpus
    contradictory_ids = {"p14"}   # p14 contradicts p02 on BDNF+exercise
    if passage.id in contradictory_ids:
        return 0.35
    # Secondary commentary — may oversimplify
    from hypergraph import EvidenceType
    if passage.evidence_type == EvidenceType.SECONDARY_COMMENTARY:
        return 0.55
    return 0.80


def _first_sentence(text: str) -> str:
    """Extract first sentence as the claim."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[0] if sentences else text[:120]


def build_hypergraph(passages: List[Passage]) -> KnowledgeHypergraph:
    """
    Main builder: one Hyperedge per Passage.
    Nodes are shared across edges when entities repeat.
    """
    hg = KnowledgeHypergraph()

    # 1. Create or retrieve nodes
    node_registry: dict[str, Node] = {}

    def get_node(entity_id: str) -> Node:
        if entity_id not in node_registry:
            node_registry[entity_id] = Node(
                id=entity_id,
                label=entity_id.replace("_", " "),
                node_type=NODE_TYPE_MAP.get(entity_id, "concept"),
            )
        return node_registry[entity_id]

    # 2. Create one Hyperedge per Passage
    for passage in passages:
        nodes = frozenset(get_node(e) for e in passage.entities)

        score = EpistemicScore.from_evidence_type(
            etype=passage.evidence_type,
            source_authority=_source_authority(passage.source),
            sample_n=passage.sample_n,
            year=passage.year,
            directness=0.9 if passage.evidence_type.value >= 5 else 0.6,
            specificity=0.85 if passage.sample_n and passage.sample_n > 50 else 0.55,
            independence=_independence(passage, passages),
            consistency=_consistency(passage, passages),
            replication=0.75 if passage.evidence_type.value >= 6 else 0.45,
        )

        edge = Hyperedge(
            id=f"e_{passage.id}",
            nodes=nodes,
            claim=_first_sentence(passage.text),
            passages=[passage],
            score=score,
        )
        hg.add_edge(edge)

    return hg
