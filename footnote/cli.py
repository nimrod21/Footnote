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
) -> None:
    """Hybrid search over indexed provisions. (M3)"""
    raise typer.Exit(_todo("search", "M3"))


@app.command()
def ask(
    question: str,
    agent: bool = typer.Option(False, help="Use the multi-hop research loop"),
    max_hops: int = typer.Option(4),
) -> None:
    """Answer a question with citations, or refuse. (M4/M5)"""
    raise typer.Exit(_todo("ask", "M4"))


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
