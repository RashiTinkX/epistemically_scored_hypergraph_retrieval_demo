"""
viz_export.py
─────────────
Serialises a KnowledgeHypergraph + HyperRAGRetriever state to JSON
consumed by the HTML/D3 visualisation (viz/index.html).

Output schema:
{
  "nodes": [ { id, label, node_type, degree } ],
  "edges": [
    {
      id, claim, source, evidence_type, year, sample_n,
      epistemic_quality,          ← aggregate scalar
      score: { dim: value, … },   ← 9 raw dimensions
      weights: { dim: weight, … },
      node_ids: [ … ],
      passage_text
    }
  ],
  "meta": { nodes, edges, avg_epistemic_quality, evidence_types }
}
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from hypergraph import KnowledgeHypergraph
from retrieval import HyperRAGRetriever


def export_viz_json(
    hg: KnowledgeHypergraph,
    retriever: Optional[HyperRAGRetriever] = None,
    path: str = "viz/hypergraph_data.json",
) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_degree: dict[str, int] = {}
    for edge in hg.edges.values():
        for n in edge.nodes:
            node_degree[n.id] = node_degree.get(n.id, 0) + 1

    nodes_out = [
        {
            "id":        n.id,
            "label":     n.label,
            "node_type": n.node_type,
            "degree":    node_degree.get(n.id, 0),
        }
        for n in hg.nodes.values()
    ]

    # ── Edges ────────────────────────────────────────────────────────────────
    edges_out = []
    for edge in hg.edges.values():
        passage  = edge.passages[0]
        score    = edge.score
        dims     = score.dimension_names()
        vals     = score.vector()
        edges_out.append({
            "id":                edge.id,
            "claim":             edge.claim,
            "source":            passage.source,
            "evidence_type":     passage.evidence_type.name,
            "year":              passage.year,
            "sample_n":          passage.sample_n,
            "epistemic_quality": round(edge.epistemic_quality(), 4),
            "score":             {d: round(v, 4) for d, v in zip(dims, vals)},
            "weights":           {d: score.WEIGHTS[d] for d in dims},
            "node_ids":          [n.id for n in edge.nodes],
            "passage_text":      passage.text,
        })

    # ── Meta ─────────────────────────────────────────────────────────────────
    stats = hg.stats()
    evidence_types = list({e["evidence_type"] for e in edges_out})

    payload = {
        "nodes": nodes_out,
        "edges": edges_out,
        "meta":  {**stats, "evidence_types": sorted(evidence_types)},
    }

    out_path.write_text(json.dumps(payload, indent=2))
    return str(out_path)
