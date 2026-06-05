"""
main.py
───────
Single entry point for the Hyper-RAG pipeline.

Usage:
    python main.py                     # run all demos + export viz data
    python main.py --demo-only         # terminal demos only
    python main.py --viz-only          # export viz JSON only
    python main.py --query "your query here"   # single interactive query
"""

import argparse
from pathlib import Path

from corpus    import PASSAGES
from builder   import build_hypergraph
from retrieval import HyperRAGRetriever
from viz_export import export_viz_json


def run_query(retriever: HyperRAGRetriever, query: str, top_k: int = 5):
    """Run a single query and print results to terminal."""
    print(f'\n Query: "{query}"')
    print("─" * 80)

    entities = retriever.extract_query_entities(query)
    print(f"  Matched entities: {entities or '(none — falling back to flat scoring)'}")

    results = retriever.retrieve(query, top_k=top_k)
    for r in results:
        ep_bar = "█" * round(r.epistemic_score * 10) + "░" * (10 - round(r.epistemic_score * 10))
        print(
            f"  #{r.rank} [{r.retrieval_path:5}]  "
            f"cov={r.entity_coverage:.2f}  sem={r.semantic_score:.2f}  "
            f"ep={r.epistemic_score:.2f}  ∑={r.combined_score:.2f}  [{ep_bar}]"
        )
        print(f"      {r.edge.claim[:75]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Hyper-RAG neuroscience pipeline")
    parser.add_argument("--demo-only",  action="store_true", help="Run terminal demos only")
    parser.add_argument("--viz-only",   action="store_true", help="Export viz JSON only")
    parser.add_argument("--query",      type=str,            help="Run a single query")
    parser.add_argument("--top-k",      type=int, default=5, help="Number of results")
    parser.add_argument("--out",        type=str, default="viz/hypergraph_data.json",
                        help="Output path for viz JSON")
    args = parser.parse_args()

    # ── Build pipeline ───────────────────────────────────────────────────────
    print("\n[1/3] Building knowledge hypergraph from corpus…")
    hg    = build_hypergraph(PASSAGES)
    stats = hg.stats()
    print(f"      ✓ {stats['nodes']} nodes  |  {stats['edges']} hyperedges  "
          f"|  avg epistemic quality: {stats['avg_epistemic_quality']}")

    print("[2/3] Initialising Hyper-RAG retriever…")
    retriever = HyperRAGRetriever(hg, w_coverage=0.35, w_semantic=0.35, w_epistemic=0.30)
    print(f"      ✓ TF-IDF index built  |  weights: cov=0.35  sem=0.35  ep=0.30")

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if args.query:
        print("[3/3] Running query…")
        run_query(retriever, args.query, top_k=args.top_k)

    elif args.viz_only:
        print("[3/3] Exporting visualisation JSON…")
        out_path = export_viz_json(hg, retriever, path=args.out)
        print(f"      ✓ Written to {out_path}")
        print(f"\n  Open viz/index.html in your browser.\n")

    elif args.demo_only:
        print("[3/3] Running terminal demos…\n")
        from demo import main as demo_main
        demo_main()

    else:
        print("[3/3] Running terminal demos + exporting visualisation…\n")
        from demo import main as demo_main
        demo_main()

        print("\nExporting visualisation data…")
        out_path = export_viz_json(hg, retriever, path=args.out)
        print(f"  ✓ Written to {out_path}")
        print(f"\n  Open viz/index.html in your browser.\n")


if __name__ == "__main__":
    main()
