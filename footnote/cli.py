"""Footnote CLI. Subcommands land with their milestones; stubs fail loudly."""

from __future__ import annotations

import typer

app = typer.Typer(help="Ask the GDPR and the EU AI Act a question.", no_args_is_help=True)


@app.command()
def ingest(
    corpus: str = typer.Option("gdpr,ai_act", help="Comma-separated corpus ids"),
    strategy: str = typer.Option("provision", help="provision|article|window"),
    dry_run: bool = typer.Option(False, help="Report counts and quota projection only"),
) -> None:
    """Fetch, parse, chunk, embed, and index a corpus. (M1-M2)"""
    from collections import Counter

    from footnote.corpus.chunkers import estimate_tokens, select_chunks
    from footnote.corpus.fetch import fetch_instrument
    from footnote.corpus.parser import parse_instrument

    for iid in corpus.split(","):
        iid = iid.strip()
        path = fetch_instrument(iid)
        provisions = parse_instrument(iid, path)
        chunks = select_chunks(provisions, strategy)
        counts = dict(Counter(p.type for p in provisions))
        typer.echo(
            f"{iid}: {len(provisions)} provisions {counts} -> "
            f"{len(chunks)} chunks [{strategy}], ~{estimate_tokens(chunks):,} tokens to embed"
        )
    if dry_run:
        typer.echo("dry run - nothing embedded or indexed.")
        return

    from footnote.embed.jina import get_provider
    from footnote.store.qdrant_store import QdrantStore, collection_name

    provider = get_provider("jina")
    store = QdrantStore()
    name = collection_name(provider.model_id, strategy)
    store.ensure_collection(name, provider.dimensions)

    all_chunks = []
    for iid in corpus.split(","):
        iid = iid.strip()
        provisions = parse_instrument(iid, fetch_instrument(iid))
        all_chunks.extend(select_chunks(provisions, strategy))

    vectors = provider.embed_documents([c.text for c in all_chunks])
    store.upsert(name, all_chunks, vectors)
    typer.echo(
        f"indexed {len(all_chunks)} chunks into '{name}' ({store.mode}), "
        f"total points={store.count(name)}, API tokens this run={provider.tokens_used:,}"
    )


@app.command()
def search(
    query: str,
    instrument: str = typer.Option(None),
    type: str = typer.Option(None, help="article|recital|annex|definition"),
    no_rerank: bool = typer.Option(False),
    no_sparse: bool = typer.Option(False),
    no_dense: bool = typer.Option(False),
) -> None:
    """Hybrid search over indexed provisions. (M3)"""
    from footnote.config import RunConfig
    from footnote.retrieve.pipeline import Retriever

    cfg = RunConfig(
        rerank_enabled=not no_rerank,
        sparse_enabled=not no_sparse,
        dense_enabled=not no_dense,
    )
    ret = Retriever(cfg).search(query, instrument=instrument, type=type)
    typer.echo(f"confidence={ret.confidence:.3f} via={ret.results[0].via if ret.results else '-'}")
    for r in ret.results:
        parts = [
            f"d={r.dense_score:.3f}" if r.dense_score is not None else None,
            f"s={r.sparse_score:.1f}" if r.sparse_score is not None else None,
            f"rr={r.rerank_score:.3f}" if r.rerank_score is not None else None,
        ]
        detail = " ".join(x for x in parts if x)
        typer.echo(f"{r.score:6.3f}  {r.provision.citation_label:<28} {detail}")
        typer.echo(f"        {r.provision.text[:100]}")


@app.command()
def ask(
    question: str,
    instrument: str = typer.Option(None),
    agent: bool = typer.Option(False, help="Use the multi-hop research loop"),
    max_hops: int = typer.Option(4),
    model: str = typer.Option(None, help="Override generation model"),
) -> None:
    """Answer a question with citations, or refuse. (M4/M5)"""
    from footnote.answer.generate import Answerer
    from footnote.config import RunConfig

    cfg = RunConfig()
    if model:
        cfg.generation_model = model
    if agent:
        raise typer.Exit(_todo("--agent", "M5"))

    r = Answerer(cfg).ask(question, instrument=instrument)
    a = r.answer
    typer.echo(f"[{a.verdict}] confidence={a.confidence:.3f}"
               + (f" gate={r.gate_fired}" if r.gate_fired else ""))
    if a.text:
        typer.echo(a.text)
    for c in a.citations:
        typer.echo(f"  - {c.citation_label}: \"{c.quote[:110]}\"")
    if a.refusal_reason:
        typer.echo(f"  reason: {a.refusal_reason}")
    if r.llm:
        typer.echo(f"  [{r.llm.model} {r.llm.prompt_tokens}+{r.llm.completion_tokens}tok "
                   f"{r.llm.latency_ms}ms ${r.llm.cost_usd:.4f}]")
    typer.echo(f"  {a.disclaimer}")


@app.command()
def eval(
    suite: str = typer.Option("all"),
    config: str = typer.Option("configs/baseline.yaml"),
    compare: str = typer.Option(None, help="Baseline results file for regression check"),
) -> None:
    """Run the eval suite. (M6)"""
    raise typer.Exit(_todo("eval", "M6"))


@app.command()
def traces(limit: int = typer.Option(20)) -> None:
    """Recent queries: cost totals, latency, refusal rate. (M8)"""
    raise typer.Exit(_todo("traces", "M8"))


def _todo(cmd: str, milestone: str) -> int:
    typer.echo(f"'{cmd}' lands in {milestone} — not implemented yet.", err=True)
    return 1


if __name__ == "__main__":
    app()
