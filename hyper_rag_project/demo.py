"""
demo.py
───────
End-to-end Hyper-RAG demo for neuroscience knowledge hypergraph.

Demonstrates the three-stage retrieval algorithm from Feng et al. (2025):
  Stage 1 — Entity extraction from query
  Stage 2 — Hypergraph traversal (seed → 1-hop → 2-hop neighbourhood)
  Stage 3 — Scoring: entity coverage + semantic similarity + epistemic quality

Four demos:
  1. Traversal trace     — shows exactly which nodes were matched and which
                           edges were found at each hop
  2. Flat vs Hyper-RAG  — side-by-side comparison showing where results differ
  3. High-order payoff  — multi-entity query where 2-hop traversal surfaces
                           edges a flat retriever misses entirely
  4. Epistemic gate      — quality filtering on top of the traversal
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from corpus import PASSAGES
from builder import build_hypergraph
from retrieval import HyperRAGRetriever, RetrievalResult
from hypergraph import EvidenceType

console = Console(width=115)


# ─── Display helpers ──────────────────────────────────────────────────────────

ETYPE_LABEL = {
    EvidenceType.META_ANALYSIS:         "★★★ meta-analysis",
    EvidenceType.CONTROLLED_EXPERIMENT: "★★☆ RCT/experiment",
    EvidenceType.OBSERVATIONAL:         "★☆☆ observational",
    EvidenceType.LONGITUDINAL:          "○★☆ longitudinal",
    EvidenceType.CROSS_SECTIONAL:       "○○☆ cross-sectional",
    EvidenceType.CASE_REPORT:           "○○○ case report",
    EvidenceType.SECONDARY_COMMENTARY:  "·   commentary",
}

PATH_COLOR = {
    "seed":  "bold green",
    "1-hop": "cyan",
    "2-hop": "blue",
    "flat":  "dim",
}

def quality_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)

def score_color(score: float) -> str:
    if score >= 0.70: return "green"
    if score >= 0.50: return "yellow"
    return "red"

def result_table(results: list[RetrievalResult], title: str, show_path: bool = True) -> Table:
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan",
                  padding=(0, 1), expand=True)
    table.add_column("#",         width=3,  justify="right")
    table.add_column("Claim",               width=40)
    table.add_column("Evidence",            width=20)
    if show_path:
        table.add_column("Path",            width=7)
        table.add_column("Cov",  width=5,  justify="right")
    table.add_column("Sem",      width=5,  justify="right")
    table.add_column("Ep",       width=5,  justify="right")
    table.add_column("∑",        width=5,  justify="right")
    table.add_column("Quality",            width=12)

    for r in results:
        p     = r.edge.passages[0]
        etype = ETYPE_LABEL.get(p.evidence_type, "?")
        claim = (r.edge.claim[:38] + "…") if len(r.edge.claim) > 38 else r.edge.claim
        ec    = score_color(r.epistemic_score)
        cc    = score_color(r.combined_score)
        pc    = PATH_COLOR.get(r.retrieval_path, "white")

        row = [
            str(r.rank),
            claim,
            etype,
        ]
        if show_path:
            row += [
                f"[{pc}]{r.retrieval_path}[/]",
                f"[dim]{r.entity_coverage:.2f}[/]",
            ]
        row += [
            f"[dim]{r.semantic_score:.2f}[/]",
            f"[{ec}]{r.epistemic_score:.2f}[/]",
            f"[bold {cc}]{r.combined_score:.2f}[/]",
            f"[{ec}]{quality_bar(r.epistemic_score)}[/]",
        ]
        table.add_row(*row)
    return table


# ─── Demo 1: Traversal trace ──────────────────────────────────────────────────

def demo_traversal_trace(retriever: HyperRAGRetriever):
    console.rule("[bold cyan]Demo 1 · Traversal trace — how the hypergraph is walked[/]")
    query = "How does BDNF affect synaptic plasticity and memory?"
    console.print(f"[bold]Query:[/] {query}\n")

    # Stage 1
    entities = retriever.extract_query_entities(query)
    console.print(Panel(
        f"[bold]Matched entities:[/] {', '.join(entities) if entities else '(none)'}\n"
        f"[dim]These become the seed nodes for hypergraph traversal.[/]",
        title="Stage 1 — Entity extraction", border_style="green", padding=(0,1)
    ))

    # Stage 2 — show each hop
    traversal = retriever.traverse(entities, max_hops=2)
    seed_ids  = [eid for eid, p in traversal.items() if p == "seed"]
    hop1_ids  = [eid for eid, p in traversal.items() if p == "1-hop"]
    hop2_ids  = [eid for eid, p in traversal.items() if p == "2-hop"]

    hop_text = (
        f"[green bold]Seed edges[/]  ({len(seed_ids)}): "
        f"{', '.join(seed_ids)}\n"
        f"  [dim]↑ Edges directly containing a query entity[/]\n\n"
        f"[cyan bold]1-hop edges[/] ({len(hop1_ids)}): "
        f"{', '.join(hop1_ids) or '—'}\n"
        f"  [dim]↑ Found by expanding through ALL nodes co-occurring in seed edges.\n"
        f"     Each hyperedge connects N entities at once, so one hop covers\n"
        f"     far more ground than binary-graph expansion would.[/]\n\n"
        f"[blue bold]2-hop edges[/] ({len(hop2_ids)}): "
        f"{', '.join(hop2_ids) or '—'}\n"
        f"  [dim]↑ Further expansion through nodes reached at 1-hop.[/]\n\n"
        f"[bold]Total candidate pool:[/] {len(traversal)} / {len(retriever.hg.edges)} edges\n"
        f"  [dim](standard RAG scores all {len(retriever.hg.edges)}; "
        f"traversal focuses on the structurally connected subgraph)[/]"
    )
    console.print(Panel(hop_text, title="Stage 2 — Hypergraph traversal", border_style="cyan", padding=(0,1)))

    # Stage 3
    results = retriever.retrieve(query, top_k=5)
    console.print(result_table(results, "Stage 3 — Final ranked results (cov + sem + ep)"))
    console.print()


# ─── Demo 2: Flat vs Hyper-RAG ───────────────────────────────────────────────

def demo_flat_vs_hyperrag(retriever: HyperRAGRetriever):
    console.rule("[bold cyan]Demo 2 · Standard RAG vs Hyper-RAG[/]")
    query = "Does exercise increase BDNF and improve memory?"
    console.print(f"[bold]Query:[/] {query}\n")

    flat    = retriever.retrieve_flat(query, top_k=5)
    hyper   = retriever.retrieve(query, top_k=5)

    console.print(result_table(flat,  "Standard RAG (TF-IDF only, no graph traversal)", show_path=False))
    console.print()
    console.print(result_table(hyper, "Hyper-RAG (traversal + coverage + epistemic)", show_path=True))

    # Rank change summary
    flat_ids  = [r.edge.id for r in flat]
    hyper_ids = [r.edge.id for r in hyper]
    console.print("\n[bold]Rank changes (Standard RAG → Hyper-RAG):[/]")
    shown = False
    for eid in dict.fromkeys(flat_ids + hyper_ids):
        fr = flat_ids.index(eid)  + 1 if eid in flat_ids  else "—"
        hr = hyper_ids.index(eid) + 1 if eid in hyper_ids else "out"
        if fr != hr:
            shown = True
            direction = "⬆" if (isinstance(hr, int) and isinstance(fr, int) and hr < fr) else "⬇"
            console.print(f"  {direction}  #{fr} → #{hr}   {retriever.hg.edges[eid].claim[:60]}…")
    if not shown:
        console.print("  (no rank changes for this query)")
    console.print()


# ─── Demo 3: High-order payoff ────────────────────────────────────────────────

def demo_high_order(retriever: HyperRAGRetriever):
    console.rule("[bold cyan]Demo 3 · High-order traversal payoff[/]")
    console.print(
        "A multi-entity query about [bold]microglia, TREM2, and cognitive decline[/] "
        "that spans two topic clusters.\n"
        "The 1-hop expansion through hyperedges bridges them — "
        "something a flat retriever cannot do.\n"
    )
    query = "How do microglia and TREM2 influence cognitive decline in Alzheimer's?"

    entities = retriever.extract_query_entities(query)
    console.print(f"[dim]Extracted entities: {entities}[/]\n")

    traversal = retriever.traverse(entities, max_hops=2)
    paths = {}
    for eid, path in traversal.items():
        paths.setdefault(path, []).append(eid)

    for path_label, eids in paths.items():
        color = PATH_COLOR.get(path_label, "white")
        console.print(f"  [{color}]{path_label}[/]: {', '.join(eids)}")
    console.print()

    hyper = retriever.retrieve(query, top_k=6, max_hops=2)
    flat  = retriever.retrieve_flat(query, top_k=6)

    console.print(result_table(hyper, "Hyper-RAG — traversal-aware"))
    console.print()

    # Show which edges Hyper-RAG found that flat missed
    flat_ids  = {r.edge.id for r in flat}
    hyper_ids = {r.edge.id for r in hyper}
    extra = hyper_ids - flat_ids
    if extra:
        console.print(f"[bold]Edges found by Hyper-RAG traversal but NOT in flat top-6:[/]")
        for eid in extra:
            e    = retriever.hg.edges[eid]
            path = traversal.get(eid, "?")
            color = PATH_COLOR.get(path, "white")
            console.print(f"  [{color}]{path}[/]  {e.claim[:80]}…")
    console.print()


# ─── Demo 4: Epistemic gate on traversal results ─────────────────────────────

def demo_epistemic_gate(retriever: HyperRAGRetriever):
    console.rule("[bold cyan]Demo 4 · Epistemic quality gate on traversal results[/]")
    query = "What is the role of sleep in hippocampal memory consolidation?"
    console.print(f"[bold]Query:[/] {query}\n")

    no_gate   = retriever.retrieve(query, top_k=6, min_epistemic=0.0)
    with_gate = retriever.retrieve(query, top_k=6, min_epistemic=0.65)

    console.print(result_table(no_gate,   "No epistemic gate"))
    console.print()
    console.print(result_table(with_gate, "Epistemic gate ≥ 0.65 — weak evidence excluded"))
    console.print()

    removed = {r.edge.id for r in no_gate} - {r.edge.id for r in with_gate}
    if removed:
        console.print("[bold]Edges removed by the epistemic gate:[/]")
        for eid in removed:
            e  = retriever.hg.edges[eid]
            eq = e.epistemic_quality()
            console.print(f"  [{score_color(eq)}]ep={eq:.3f}[/]  {e.claim[:80]}")
    console.print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold white on blue]  Hyper-RAG Demo · Epistemically-Scored Knowledge Hypergraph  [/]\n")

    hg = build_hypergraph(PASSAGES)
    stats = hg.stats()

    console.print(Panel(
        f"[bold]Nodes:[/] {stats['nodes']}   "
        f"[bold]Hyperedges:[/] {stats['edges']}   "
        f"[bold]Avg epistemic quality:[/] [green]{stats['avg_epistemic_quality']}[/]\n\n"
        f"[dim]Retrieval: entity extraction → hypergraph traversal → "
        f"score(cov=0.35, sem=0.35, ep=0.30)\n"
        f"Paper: Feng et al. (2025) arXiv:2504.08758[/]",
        title="Hypergraph stats", border_style="blue",
    ))

    retriever = HyperRAGRetriever(hg, w_coverage=0.35, w_semantic=0.35, w_epistemic=0.30)

    demo_traversal_trace(retriever)
    demo_flat_vs_hyperrag(retriever)
    demo_high_order(retriever)
    demo_epistemic_gate(retriever)

    console.print("[dim]Demo complete.[/]\n")


if __name__ == "__main__":
    main()
