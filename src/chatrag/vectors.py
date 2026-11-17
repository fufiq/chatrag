"""Local USearch index, embedding cache and a multilingual FastEmbed reranker."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

HOME = Path(os.environ.get('CHATRAG_HOME', 'data'))
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANK_REPO = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
RERANK_REVISION = '1427fd652930e4ba29e8149678df786c240d8825'
RERANK_FILE = 'onnx/model_quint8_avx2.onnx'
MODEL_DIR = HOME / 'models/mmarco'


def fingerprint(con):
    sources = [tuple(r) for r in con.execute('SELECT path,sha256 FROM sources ORDER BY path')]
    version = con.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0]
    return hashlib.sha256(json.dumps([version, sources, MODEL], ensure_ascii=False).encode()).hexdigest()


def index_path(db, con):
    return Path(str(db) + '.vectors') / (fingerprint(con) + '.usearch')


def confirmed_path(path):
    return path.with_name(path.stem + '-confirmed.usearch')


@lru_cache(maxsize=2)
def embedder(models_dir):
    from fastembed import TextEmbedding
    # Only an already downloaded local model; a query never touches the network.
    return TextEmbedding(MODEL, cache_dir=str(models_dir),
                         threads=4, local_files_only=True)


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from fastembed.common.model_description import ModelSource
    if not (MODEL_DIR / RERANK_FILE).exists():
        raise ValueError('Reranker is not downloaded. Run prepare-models.')
    provenance = json.loads((MODEL_DIR / 'provenance.json').read_text())
    if provenance['revision'] != RERANK_REVISION or provenance['repository'] != RERANK_REPO:
        raise ValueError('Wrong reranker revision; run prepare-models')
    if RERANK_FILE not in provenance['files'] or 'tokenizer.json' not in provenance['files']:
        raise ValueError('Reranker manifest is incomplete')
    for name, expected in provenance['files'].items():
        asset = (MODEL_DIR / name).resolve()
        if not asset.is_relative_to(MODEL_DIR.resolve()) or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            raise ValueError('Reranker file is corrupted: ' + name)
    name = 'chatrag/mmarco-quantized'
    if not any(m['model'] == name for m in TextCrossEncoder.list_supported_models()):
        TextCrossEncoder.add_custom_model(name, sources=ModelSource(hf=RERANK_REPO),
            model_file=RERANK_FILE, license='apache-2.0', size_in_gb=0.12)
    return TextCrossEncoder(name, specific_model_path=str(MODEL_DIR), threads=4,
                            local_files_only=True)


def prepare_models(models_dir=None):
    """The only command allowed to download public models."""
    from huggingface_hub import snapshot_download
    from fastembed import TextEmbedding
    if models_dir is not None:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        TextEmbedding(MODEL, cache_dir=str(models_dir))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(RERANK_REPO, revision=RERANK_REVISION, local_dir=str(MODEL_DIR),
        allow_patterns=[RERANK_FILE, 'config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json', 'README.md'])
    assets = {str(p.relative_to(MODEL_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in MODEL_DIR.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name != 'provenance.json'}
    (MODEL_DIR / 'provenance.json').write_text(json.dumps(dict(repository=RERANK_REPO,
        revision=RERANK_REVISION, license='apache-2.0', files=assets), indent=2))
    return dict(model=RERANK_REPO, revision=RERANK_REVISION, files=len(assets))


def embedding_text(row):
    # Short replies are embedded together with the message they answer.
    import re
    preview = re.search(r'\(→.*?: "(.*?)"\): $', row['header'])
    text = row['body']
    if preview:
        text += ' [reply to: ' + preview[1] + ']'
    if len(row['body']) < 25 and row['context_text']:
        text += '\nContext: ' + row['context_text'][:800]
    return text[:8000]


def build_vectors(con, db, models_dir):
    import fcntl
    lock = open(str(db) + '.vectors.lock', 'a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('The vector index is already being updated by another process') from None
        return _build_vectors(con, db, models_dir)
    finally:
        lock.close()


def _build_vectors(con, db, models_dir):
    import numpy as np
    from usearch.index import Index
    target = index_path(db, con)
    target.parent.mkdir(parents=True, exist_ok=True)
    confirmed_target = confirmed_path(target)
    if all(p.exists() and p.with_suffix('.json').exists() for p in (target, confirmed_target)):
        try:
            return dict(index=str(target), reused=True, vectors=len(load_index(str(target))),
                        confirmed_vectors=len(load_index(str(confirmed_target))))
        except Exception as exc:
            print('Rebuilding a damaged index: ' + str(exc), file=sys.stderr)
    cache = sqlite3.connect(str(db) + '.embeddings.sqlite')
    cache.execute('CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY, vector BLOB)')
    key = lambda text: hashlib.sha256((MODEL + '\0' + text).encode()).hexdigest()
    rows = con.execute('SELECT * FROM messages WHERE subject=1 ORDER BY id').fetchall()
    confirmed_ids = {r[0] for r in con.execute("SELECT m.id FROM messages m JOIN provenance p ON p.message_id=m.id WHERE m.subject=1 AND p.authorship='confirmed'")}
    index = Index(ndim=384, metric='cos', dtype='f32', connectivity=16,
                  expansion_add=128, expansion_search=128)
    confirmed = Index(ndim=384, metric='cos', dtype='f32', connectivity=16,
                      expansion_add=128, expansion_search=128)
    def add(keys, vecs):
        keys = np.asarray(keys, dtype='uint64')
        vecs = np.asarray(vecs, dtype='float32')
        index.add(keys, vecs, threads=4)
        mask = np.array([int(k) in confirmed_ids for k in keys], dtype=bool)
        if mask.any():
            confirmed.add(keys[mask], vecs[mask], threads=4)
    missing = []
    generated = 0
    def flush(batch):
        nonlocal generated
        if not batch:
            return
        texts = [text for _, text in batch]
        vecs = np.array(list(embedder(str(models_dir)).embed(texts, batch_size=64)), dtype='float32')
        cache.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?)',
                          [(key(t), v.tobytes()) for t, v in zip(texts, vecs)])
        cache.commit()
        add([rid for rid, _ in batch], vecs)
        generated += len(batch)
        print(f'vectors: {len(index)}/{len(rows)}, generated={generated}', file=sys.stderr, flush=True)
    ids, vectors = [], []
    for row in rows:
        text = embedding_text(row)
        hit = cache.execute('SELECT vector FROM vectors WHERE key=?', (key(text),)).fetchone()
        if hit and (len(hit[0]) != 384*4 or not np.isfinite(np.frombuffer(hit[0], dtype='float32')).all()):
            hit = None
        if hit:
            ids.append(row['id'])
            vectors.append(np.frombuffer(hit[0], dtype='float32'))
            if len(ids) >= 1024:
                add(ids, vectors)
                ids, vectors = [], []
        else:
            missing.append((row['id'], text))
            if len(missing) >= 256:
                flush(missing)
                missing = []
    if ids:
        add(ids, vectors)
    flush(missing)
    cache.close()
    # The confirmed subset is published first; the full index completes the generation.
    for built, path in ((confirmed, confirmed_target), (index, target)):
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix='.building-')
        os.close(fd)
        try:
            built.save(temporary)
            metadata = dict(model=MODEL, fingerprint=path.stem, vectors=len(built),
                            sha256=hashlib.sha256(Path(temporary).read_bytes()).hexdigest())
            staging_manifest = Path(temporary + '.json')
            staging_manifest.write_text(json.dumps(metadata, indent=2))
            os.replace(temporary, path)
            os.replace(staging_manifest, path.with_suffix('.json'))
        finally:
            Path(temporary).unlink(missing_ok=True)
    return dict(index=str(target), vectors=len(index), generated=generated,
                confirmed_vectors=len(confirmed), model=MODEL, fingerprint=fingerprint(con))


@lru_cache(maxsize=4)
def load_index(path):
    from usearch.index import Index
    p = Path(path)
    meta = json.loads(p.with_suffix('.json').read_text())
    if meta['model'] != MODEL or meta['fingerprint'] != p.stem:
        raise ValueError('Vector index manifest does not match the model/corpus')
    if hashlib.sha256(p.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Vector index SHA-256 mismatch')
    index = Index.restore(path, view=True)
    if len(index) != meta['vectors'] or index.ndim != 384:
        raise ValueError('Vector index is incomplete')
    return index


def candidates(con, db, query, models_dir, limit=120, include_uncertain=False):
    import numpy as np
    from chatrag.terms import semantic_text
    if type(include_uncertain) is not bool:
        raise ValueError('include_uncertain must be boolean')
    path = index_path(db, con)
    if not path.exists():
        raise ValueError('The vector index does not match the corpus. Run build-vectors.')
    if not include_uncertain:
        path = confirmed_path(path)
    texts = list(dict.fromkeys([query, semantic_text(query)]))
    vecs = list(embedder(str(models_dir)).embed(texts))
    ranked = {}
    index = load_index(str(path))
    for vector in vecs:
        matches = index.search(np.asarray(vector, dtype='float32'), count=limit)
        for rank, hit in enumerate(matches, 1):
            rid = int(hit.key)
            ranked[rid] = max(ranked.get(rid, 0), 1/(60+rank))
    return sorted(ranked, key=ranked.get, reverse=True)


def rerank(query, passages):
    from chatrag.terms import semantic_text
    return list(reranker().rerank(semantic_text(query), [semantic_text(p, fuzzy=False) for p in passages], batch_size=16))
