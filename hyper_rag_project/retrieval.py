"""
retrieval.py
────────────
Hyper-RAG retrieval engine — genuine hypergraph-driven retrieval.

Based on: Feng et al. (2025) "Hyper-RAG: Combating LLM Hallucinations using
Hypergraph-Driven Retrieval-Augmented Generation" (arXiv:2504.08758).

The paper's retrieval algorithm has three stages:
  1. Entity extraction  — match query tokens against known node ids/labels
  2. Hypergraph traversal:
       a. Seed edges    — all hyperedges containing ≥1 query entity (low-order)
       b. Neighbourhood — expand through co-occurring nodes to find further
                          connected hyperedges (high-order, beyond-pairwise)
     This traversal is the core of what distinguishes Hyper-RAG from standard
     RAG and Graph-RAG: a regular graph can only expand pairwise; a hyperedge
     connects N entities at once, so one hop can surface a much richer
     neighbourhood.
  3. Scoring — candidates ranked by a combination of:
       • entity_coverage  : fraction of query entities present in the edge
       • semantic_sim     : TF-IDF cosine similarity between query and edge text
       • epistemic_quality: EpistemicScore.aggregate()  (our addition)

     final_score = w_cov·coverage + w_sem·semantic + w_ep·epistemic_quality

     where w_cov, w_sem, w_ep are tunable (default 0.35 / 0.35 / 0.30).

A flat TF-IDF-only baseline (retrieve_flat) is kept for comparison so the
effect of the traversal can be directly demonstrated.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hypergraph import KnowledgeHypergraph, Hyperedge, EvidenceType


# Result type

@dataclass
class RetrievalResult:
    edge: Hyperedge
    entity_coverage:   float          # fraction of query entities in this edge
    semantic_score:    float          # TF-IDF cosine similarity
    epistemic_score:   float          # EpistemicScore.aggregate()
    combined_score:    float          # weighted sum of all three
    retrieval_path:    str            # "seed" | "1-hop" | "2-hop" | "flat"
    matched_entities:  List[str] = field(default_factory=list)
    rank: int = 0


# Retriever

class HyperRAGRetriever:
    """
    Three-stage Hyper-RAG retriever following Feng et al. (2025).

    Stage 1 — Entity extraction
    Stage 2 — Hypergraph traversal (seed → neighbourhood expansion)
    Stage 3 — Scoring: entity coverage + semantic similarity + epistemic quality
    """

    def __init__(
        self,
        hg: KnowledgeHypergraph,
        w_coverage:  float = 0.35,
        w_semantic:  float = 0.35,
        w_epistemic: float = 0.30,
        max_hops:    int   = 2,
    ):
        """
        Parameters
        ----------
        hg           : the knowledge hypergraph
        w_coverage   : weight on entity coverage score  (Σ = 1.0)
        w_semantic   : weight on TF-IDF semantic score
        w_epistemic  : weight on epistemic quality score
        max_hops     : neighbourhood expansion depth (1 or 2 recommended)
        """
        assert abs(w_coverage + w_semantic + w_epistemic - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.hg          = hg
        self.w_cov       = w_coverage
        self.w_sem       = w_semantic
        self.w_ep        = w_epistemic
        self.max_hops    = max_hops

        # Build lookup structures
        self._edges      = list(hg.edges.values())
        self._node_ids   = set(hg.nodes.keys())
        self._node_labels = {n.label.lower(): n.id for n in hg.nodes.values()}
        # Also index individual words of multi-word labels
        self._node_words: dict[str, list[str]] = {}
        for label_lc, nid in self._node_labels.items():
            for word in label_lc.split():
                self._node_words.setdefault(word, []).append(nid)

        # TF-IDF index for semantic scoring
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._build_tfidf_index()

    # TF-IDF index

    def _edge_text(self, edge: Hyperedge) -> str:
        parts = [edge.claim]
        for p in edge.passages:
            parts.append(p.text)
        parts.append(" ".join(edge.node_labels()))
        return " ".join(parts)

    def _build_tfidf_index(self):
        docs = [self._edge_text(e) for e in self._edges]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(docs)
        self._edge_index   = {e.id: i for i, e in enumerate(self._edges)}

    # Stage 1: Entity extraction

    def extract_query_entities(self, query: str) -> List[str]:
        """
        Match query tokens/phrases against known node ids and labels.
        Returns a list of matched node ids.

        Strategy:
          1. Exact match against node id (e.g. "BDNF", "hippocampus")
          2. Exact match against full node label (lowercased)
          3. Partial match: any query word that maps to a node label word
        """
        q_lower = query.lower()
        q_tokens = set(q_lower.replace("?", "").replace(",", "").split())
        matched: Set[str] = set()

        # 1. Exact node id match (case-insensitive)
        for nid in self._node_ids:
            if nid.lower() in q_lower:
                matched.add(nid)

        # 2. Exact label match
        for label_lc, nid in self._node_labels.items():
            if label_lc in q_lower:
                matched.add(nid)

        # 3. Word-level partial match
        for token in q_tokens:
            if len(token) >= 4:      # skip short stop-words
                for nid in self._node_words.get(token, []):
                    matched.add(nid)

        return list(matched)

    # Stage 2: Hypergraph traversal

    def _edges_for_node(self, node_id: str) -> List[Hyperedge]:
        return [e for e in self._edges if any(n.id == node_id for n in e.nodes)]

    def traverse(
        self,
        query_entity_ids: List[str],
        max_hops: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Hypergraph-driven traversal starting from query entities.

        Returns a dict: {edge_id → retrieval_path}
        where retrieval_path is "seed", "1-hop", or "2-hop".

        How this exploits hyperedges (vs. ordinary graph traversal):
          - In a binary graph, expanding one step from node A gives you all
            nodes directly connected to A (one pair at a time).
          - Here, a single hyperedge connects N nodes simultaneously.
            Touching one seed node in a hyperedge immediately surfaces ALL
            co-members of that edge — a much richer expansion per hop.
          - Two hops therefore spans a significantly wider neighbourhood than
            two hops on a binary graph with the same number of nodes.
        """
        hops = max_hops if max_hops is not None else self.max_hops
        found: dict[str, str] = {}   # edge_id → path label

        if not query_entity_ids:
            return found

        # Hop 0: seed edges — directly contain a query entity
        frontier_nodes: Set[str] = set(query_entity_ids)
        seed_edge_ids:  Set[str] = set()

        for nid in frontier_nodes:
            for edge in self._edges_for_node(nid):
                if edge.id not in found:
                    found[edge.id] = "seed"
                    seed_edge_ids.add(edge.id)

        if hops == 0:
            return found

        # Hop 1: all nodes co-occurring in seed edges
        # This is the key hypergraph advantage: one seed edge can span 5+ nodes,
        # so hop-1 expands much faster than in a binary graph.
        hop1_nodes: Set[str] = set()
        for eid in seed_edge_ids:
            edge = self.hg.edges[eid]
            for n in edge.nodes:
                if n.id not in frontier_nodes:
                    hop1_nodes.add(n.id)

        for nid in hop1_nodes:
            for edge in self._edges_for_node(nid):
                if edge.id not in found:
                    found[edge.id] = "1-hop"

        if hops == 1:
            return found

        # Hop 2: nodes co-occurring in hop-1 edges
        hop2_nodes: Set[str] = set()
        hop1_edge_ids = {eid for eid, path in found.items() if path == "1-hop"}
        for eid in hop1_edge_ids:
            edge = self.hg.edges[eid]
            for n in edge.nodes:
                if n.id not in frontier_nodes and n.id not in hop1_nodes:
                    hop2_nodes.add(n.id)

        for nid in hop2_nodes:
            for edge in self._edges_for_node(nid):
                if edge.id not in found:
                    found[edge.id] = "2-hop"

        return found

    # Stage 3: Scoring

    def _entity_coverage(self, edge: Hyperedge, query_entity_ids: List[str]) -> float:
        """Fraction of query entities present in this hyperedge."""
        if not query_entity_ids:
            return 0.0
        edge_node_ids = {n.id for n in edge.nodes}
        hits = sum(1 for qe in query_entity_ids if qe in edge_node_ids)
        return hits / len(query_entity_ids)

    def _semantic_score(self, query: str, edge: Hyperedge) -> float:
        """TF-IDF cosine similarity between query and edge text."""
        q_vec = self._vectorizer.transform([query])
        idx   = self._edge_index[edge.id]
        return float(cosine_similarity(q_vec, self._tfidf_matrix[idx]).flatten()[0])

    # Public retrieve interface

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_epistemic: float = 0.0,
        evidence_types: Optional[List[EvidenceType]] = None,
        max_hops: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Full Hyper-RAG retrieval: entity extraction → traversal → scoring.

        Parameters
        ----------
        query          : natural language query
        top_k          : number of results to return
        min_epistemic  : minimum epistemic quality gate [0, 1]
        evidence_types : if set, restrict to these EvidenceTypes
        max_hops       : override instance max_hops for this query
        """
        # Stage 1
        query_entities = self.extract_query_entities(query)

        # Stage 2
        traversal = self.traverse(query_entities, max_hops=max_hops)

        # If traversal found nothing (no entity matches), fall back to all edges
        if not traversal:
            traversal = {e.id: "flat" for e in self._edges}

        # Stage 3: score every candidate edge in the traversal set
        candidates = []
        for edge_id, path in traversal.items():
            edge = self.hg.edges[edge_id]
            ep   = edge.epistemic_quality()

            # Filters
            if ep < min_epistemic:
                continue
            if evidence_types:
                edge_etypes = {p.evidence_type for p in edge.passages}
                if not edge_etypes.intersection(evidence_types):
                    continue

            cov = self._entity_coverage(edge, query_entities)
            sem = self._semantic_score(query, edge)
            combined = self.w_cov * cov + self.w_sem * sem + self.w_ep * ep

            matched = [qe for qe in query_entities
                       if any(n.id == qe for n in edge.nodes)]

            candidates.append((edge, cov, sem, ep, combined, path, matched))

        candidates.sort(key=lambda x: x[4], reverse=True)

        results = []
        for rank, (edge, cov, sem, ep, comb, path, matched) in \
                enumerate(candidates[:top_k], start=1):
            results.append(RetrievalResult(
                edge=edge,
                entity_coverage=round(cov, 4),
                semantic_score=round(sem, 4),
                epistemic_score=round(ep, 4),
                combined_score=round(comb, 4),
                retrieval_path=path,
                matched_entities=matched,
                rank=rank,
            ))
        return results

    def retrieve_flat(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Flat TF-IDF-only baseline (no hypergraph traversal, no epistemic scoring).
        Equivalent to standard RAG. Used for direct comparison.
        """
        q_vec  = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

        candidates = sorted(
            zip(self._edges, scores), key=lambda x: x[1], reverse=True
        )
        results = []
        for rank, (edge, sem) in enumerate(candidates[:top_k], start=1):
            results.append(RetrievalResult(
                edge=edge,
                entity_coverage=0.0,
                semantic_score=round(float(sem), 4),
                epistemic_score=round(edge.epistemic_quality(), 4),
                combined_score=round(float(sem), 4),
                retrieval_path="flat",
                matched_entities=[],
                rank=rank,
            ))
        return results
