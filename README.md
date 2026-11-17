# chatrag

A local, verifiable retrieval backend for RAG over chat exports. It finds what one
chosen person (the *subject*) actually wrote in a large conversation archive, proves
it against the original files, and hands an LLM agent the quotes together with their
surroundings, dates and authorship status. Generation is left to the calling agent;
nothing is sent to an external model by the retrieval commands.

Python, SQLite FTS5, [USearch](https://github.com/unum-cloud/USearch) HNSW,
[FastEmbed](https://github.com/qdrant/fastembed) (multilingual MiniLM embeddings and
an mMARCO cross-encoder reranker), an MCP server over stdio.

## Why it exists

Answering "what does X think about Y?" from years of chats is a trust problem, not
just a search problem: neighbours' words, reply previews, forwarded posts and deleted
accounts all look like the subject's text. chatrag treats every hit as evidence that
has to be re-verified:

- every result carries the file, line range, date/time and the **SHA-256 of the
  source**; the text is compared with the original on every query, and a changed file
  makes the search fail closed until the index is rebuilt;
- every message has an **authorship status**: `confirmed`, `probable` (an account the
  user explicitly marked), `other`, or `forwarded`. Only `confirmed` is returned by
  default; the rest requires an explicit opt-in and is labelled in the output and in
  the prompt;
- reply previews (`→Bob: "..."`) are stored in the header and never indexed as the
  subject's words;
- the prompt produced by `context` tells the answering model to treat chat text as
  data, to cite `[S#]` plus date, to separate quotes from paraphrase and not to read
  a missing hit as a missing opinion.

## How a query is answered

```
query ──► normalise (stemming, transliteration, alias dictionary, guarded typo fix)
      ├─► FTS5 over message text            (BM25)
      ├─► FTS5 over conversation context    (±2 neighbours, same day, ≤15 min)
      └─► USearch HNSW over embeddings      (cosine; confirmed-only and full indexes)
             │
             ▼  reciprocal rank fusion
        ≤256 candidates ─► cross-encoder rerank on the window around the match
             │
             ▼
        verify against the original file (SHA-256 + exact text) ─► quote + context
```

Details that matter in practice:

- **Explainable normalisation** (`terms.py`): Snowball stemming for Russian,
  `Unidecode` transliteration, and `aliases.json` mapping slang, abbreviations and
  transliterations to canonical concepts (`питон`, `python`, `piton` → one concept).
  Typo correction is limited to known aliases (≥5 chars, similarity ≥90, and never
  when two concepts are equally close). Every substitution is returned in
  `expansion`, so a result can be audited.
- **Single-concept queries** return actual mentions of the concept; neighbouring
  replies stay as context instead of becoming hits.
- **Approximate quotes** (`quote`): for phrases remembered loosely or spread over
  several messages. Candidate windows of 1–3 messages are scored by IDF-weighted stem
  recall, precision, ordered stem pairs and RapidFuzz `partial_ratio`; the best
  non-overlapping windows come back with ±N messages of context.
- **Context budget**: `context` packs sources into a *character* budget; long sources
  are included as a verbatim fragment around the match with the offsets, and the
  relevant neighbours are kept.
- **Atomic, fail-loud storage**: the SQLite index is built in a temporary file and
  swapped in; vector generations are tied to the corpus hashes and the schema, are
  published only after a full build, and are checked (count, dimension, SHA-256) on
  load. A damaged index is rebuilt from the embedding cache. A failing backend adds
  a warning and lexical results still come back.
- **Telegram import** (`telegram.py`): archives the original `result.json`
  byte-for-byte under its SHA-256, writes a JSONL projection and re-verifies every
  record against the archive. The projection cannot promote a deleted or forwarded
  sender to `confirmed`; the subject name is checked against the one stored in the
  index, not the one in the record.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # or: pip install -r requirements.lock
.venv/bin/chatrag prepare-models    # the only command that downloads anything
```

Data lives under `$CHATRAG_HOME` (default `./data`): `index.sqlite`, vector indexes,
embedding cache, models and Telegram imports. Keep it out of version control.

## Use

Chats are plain text files named `chat_*.txt`:

```
== 2024-03-02 ==
18:03 Alice Example (→Bob: "what do you think about ..."): the board lives on the server
18:04 Alice Example: otherwise every reconnect becomes a merge problem
```

```bash
chatrag --chats examples/chats --subject "Alice Example" update   # index + vectors
chatrag search "what should an editor edit?" --json
chatrag context "where should the board state live?" --budget 24000
chatrag quote "a neural net never gives a reliable solution" -k 3 --context 15
chatrag search "питон" --from 2024 --until 2024-06 --order oldest
chatrag verify      # full SHA-256 check of every source
chatrag doctor      # sources, vector counts, embedder and reranker
```

`--lexical` disables vectors and the reranker (no models needed); `--no-rerank` keeps
hybrid retrieval without the reranker. Global flags go before the command.

Telegram Desktop exports:

```bash
chatrag --subject "Alice Example" import-telegram ~/Downloads/ChatExport --probable-account user123456
chatrag --chats ./chats --subject "Alice Example" update
chatrag search "some topic" --include-uncertain
```

## MCP server

```json
{
  "mcpServers": {
    "chatrag": {
      "command": "/path/to/.venv/bin/chatrag",
      "args": ["--db", "/path/to/index.sqlite", "serve"]
    }
  }
}
```

Read-only tools: `chat_search`, `chat_context`, `chat_source(id)` and `chat_quote`.
Transport is MCP over stdio (JSON-RPC); the process is started by the client.
Connecting it does not make it acceptable to send private conversations to a model
provider, and that decision stays with the user.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v          # 60 tests
chatrag --chats examples/chats --subject "Alice Example" --lexical build
chatrag-eval --lexical                            # regression set in eval/cases.json
```

Tests use small synthetic corpora (some in Russian on purpose, to exercise
morphology), fake embedding vectors, and subprocess-level MCP checks. `eval/cases.json`
holds paraphrase, transliteration, typo and negative queries over `examples/chats`;
it reports direct hit@10, evidence hit@10 (the verified quote appears in the returned
context) and MRR. It is a regression set, not proof of quality on a real corpus.

## Layout

```
src/chatrag/
  knowledge.py   index build, search, context rendering, MCP server, CLI
  terms.py       normalisation, alias dictionary, query expansion
  aliases.json   slang / transliteration / abbreviation dictionary
  vectors.py     USearch indexes, embedding cache, reranker with manifest check
  quote.py       approximate-quote search
  telegram.py    verifiable Telegram import
  evaluate.py    retrieval evaluation
tests/  examples/chats/  eval/cases.json
```

## Limits

Search results are sources, not proof of completeness. The reranker score is a
logit, not a probability. The alias dictionary is a set of retrieval hints, not a
claim about anybody's opinions, and it does not resolve all homonyms.

## License

MIT
