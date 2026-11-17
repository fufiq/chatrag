"""Approximate-quote search over the whole corpus (text chats and Telegram imports), with surroundings.

Needed when there is no verbatim match: the phrase is remembered loosely, in other word forms,
or spread over neighbouring messages. Local data only, no external models:
  1. stems of the query words (Snowball, Russian + passthrough for other scripts); stop words dropped;
  2. candidates from FTS5 over the normalised tokens, ranked by bm25 (rare words weigh more);
  3. windows of 1-3 consecutive messages from one source (a phrase is often split across messages);
  4. window score: weighted stem recall + precision + ordered stem pairs + rapidfuzz partial_ratio;
  5. output: best non-overlapping windows, the authorship status of every message, +-N messages of
     context (all authors) and the same source verification as ordinary search.
Words of other participants and forwarded messages never become the subject's words:
the authorship status is shown on every message.
"""
import json
import re

import snowballstemmer
from rapidfuzz import fuzz

_RU = snowballstemmer.stemmer('russian')
_WORD = re.compile(r'[a-zа-я0-9_]+')
_STOP = set('и в во не на но а как что это то же бы ли по из за от до для при так вот ну да нет или если когда он она они мы вы я ты '
            'его её их мой твой наш ваш быть есть был была были будет уже еще ещё тоже только очень даже там тут здесь тогда потом '
            'the and for are but not you with this that was were have has had its from they them what when where then than'.split())


def tokens(text):
    t = text.lower().replace('ё', 'е')
    return [w for w in _WORD.findall(t) if len(w) >= 3 and w not in _STOP]


def stems(text):
    return [_RU.stemWord(w) if re.search('[а-я]', w) else w for w in tokens(text)]


def _idf(con, phrase):
    """Weight of a query stem: rare words (by FTS message count) weigh more; index normalisation comes from terms."""
    import math
    from chatrag import terms
    total = con.execute('SELECT count(*) FROM messages').fetchone()[0] or 1
    weights = {}
    for w in tokens(phrase):
        st = _RU.stemWord(w) if re.search('[а-я]', w) else w
        norm = terms.normalize(w, fuzzy=True)
        if not norm:
            continue
        df = con.execute('SELECT count(*) FROM search WHERE search MATCH ?', ('"' + norm[0].replace('"', '') + '"',)).fetchone()[0]
        weights[st] = max(weights.get(st, 0.0), math.log((total + 1) / (df + 1)))
    return weights


def _score(query_stems, query_text, window_text, weights):
    ws = stems(window_text)
    if not ws or not query_stems:
        return 0.0, 0.0, 0.0
    wset = set(ws)
    qset = set(query_stems)
    total = sum(weights.get(x, 1.0) for x in qset) or 1.0
    recall = sum(weights.get(x, 1.0) for x in qset & wset) / total  # share of the query stems' weight found in the window
    qp = _pairs(query_stems)
    pair = (len(qp & _pairs(ws)) / len(qp)) if qp else recall
    norm_q = ' '.join(tokens(query_text))
    norm_w = ' '.join(tokens(window_text))
    fuzzy = fuzz.partial_ratio(norm_q, norm_w) / 100 if len(norm_w) >= 8 else 0.0
    precision = len(qset & wset) / len(wset)  # long random windows must not win: precision matters too
    score = 0.40 * recall + 0.20 * min(1.0, precision * 3) + 0.15 * pair + 0.25 * fuzzy
    return score, recall, pair


def _pairs(xs):
    return {(xs[i], xs[i + 1]) for i in range(len(xs) - 1)}


def _candidates(con, phrase, limit=4000):
    """FTS candidates: the index holds normalised tokens (stems + transliteration); the query goes through the same terms.normalize."""
    from chatrag import terms
    terms = list(dict.fromkeys(terms.normalize(phrase, fuzzy=True)))
    expr = ' OR '.join('"' + t.replace('"', '') + '"' for t in terms if t)
    if not expr:
        return []
    return [r[0] for r in con.execute(
        'SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search) LIMIT ?', (expr, limit))]


def _neighbors(con, row, before, after):
    b = con.execute('SELECT * FROM messages WHERE source=? AND line<? ORDER BY line DESC LIMIT ?',
                    (row['source'], row['line'], before)).fetchall()[::-1]
    a = con.execute('SELECT * FROM messages WHERE source=? AND line>? ORDER BY line LIMIT ?',
                    (row['source'], row['line'], after)).fetchall()
    return b, a


def _auth(con, mid):
    r = con.execute('SELECT authorship FROM provenance WHERE message_id=?', (mid,)).fetchone()
    return r[0] if r else ''


def _msg(con, m):
    return {'id': f"{m['source'].rsplit('/', 1)[-1]}:{m['line']}", 'date': m['date'], 'time': m['time'], 'author': m['author'],
            'authorship': _auth(con, m['id']), 'body': m['body']}


def find_quote(con, phrase, context=15, limit=5, subject_only=False, include_uncertain=True):
    qs = stems(phrase)
    if len(qs) < 2:
        return {'phrase': phrase, 'warnings': ['the query has fewer than two meaningful words'], 'results': []}
    weights = _idf(con, phrase)
    cand = _candidates(con, phrase)
    rows = {}
    for cid in cand:
        row = con.execute('SELECT * FROM messages WHERE id=?', (cid,)).fetchone()
        if row is not None:
            rows[cid] = row
    # pre-selection by the share of query stems in the message itself
    pre = []
    for cid, row in rows.items():
        s = set(stems(row['body']))
        hit = sum(weights.get(x, 1.0) for x in set(qs) & s)
        if hit:
            pre.append((hit, cid))
    pre.sort(reverse=True)
    scored = []
    seen_windows = set()
    for _, cid in pre[:400]:
        row = rows[cid]
        b, a = _neighbors(con, row, 2, 2)
        seq = b + [row] + a
        k = len(b)
        for start in range(max(0, k - 2), k + 1):
            for width in (1, 2, 3):
                part = seq[start:start + width]
                if len(part) < width or row['id'] not in [m['id'] for m in part]:
                    continue
                key = (part[0]['source'], part[0]['line'], width)
                if key in seen_windows:
                    continue
                seen_windows.add(key)
                if subject_only and not any(m['subject'] for m in part):
                    continue
                text = ' '.join(m['body'] for m in part)
                sc, rec, pair = _score(qs, phrase, text, weights)
                scored.append((sc, rec, pair, part))
    scored.sort(key=lambda x: -x[0])
    picked, used = [], set()
    for sc, rec, pair, part in scored:
        ids = {m['id'] for m in part}
        if ids & used:
            continue
        qset = set(qs)
        while len(part) > 1 and not (set(stems(part[0]['body'])) & qset):  # edge messages without a single query word are not part of the window
            part = part[1:]
        while len(part) > 1 and not (set(stems(part[-1]['body'])) & qset):
            part = part[:-1]
        ids = {m['id'] for m in part}
        if ids & used:
            continue
        auths = [_auth(con, m['id']) for m in part]
        if not include_uncertain and any(x == 'probable' for x in auths):
            continue
        used |= ids
        first, last = part[0], part[-1]
        b, _ = _neighbors(con, first, context, 0)
        _, a = _neighbors(con, last, 0, context)
        picked.append({'score': round(sc, 3), 'recall': round(rec, 3), 'pairs': round(pair, 3),
                       'window': [_msg(con, m) for m in part],
                       'context_before': [_msg(con, m) for m in b], 'context_after': [_msg(con, m) for m in a],
                       'source': first['source'], 'line': first['line'], 'end_line': last['line'],
                       'date': first['date'], 'dates': sorted({m['date'] for m in part if m['date']})})
        if len(picked) >= limit:
            break
    return {'phrase': phrase, 'query_stems': qs, 'candidates': len(cand), 'results': picked,
            'note': 'A fuzzy match is not a verbatim quote. Check the window and the context; do not attribute other participants\' words to the subject.'}


def render(result, width=260):
    out = [f"Query: {result['phrase']}", f"stems: {' '.join(result.get('query_stems', []))}; candidates: {result.get('candidates', 0)}"]
    for w in result.get('warnings', []):
        out.append('! ' + w)
    for i, r in enumerate(result['results'], 1):
        out.append(f"\n=== {i}. score {r['score']} (stems {r['recall']}, pairs {r['pairs']}) {r['source'].rsplit('/', 1)[-1]}:{r['line']}–{r['end_line']} dates {', '.join(r['dates'])}")
        for m in r['context_before']:
            out.append(f"   {m['time']} {m['author']} [{m['authorship']}]: {m['body'][:width]}")
        for m in r['window']:
            out.append(f" > {m['time']} {m['author']} [{m['authorship']}]: {m['body'][:width * 2]}")
        for m in r['context_after']:
            out.append(f"   {m['time']} {m['author']} [{m['authorship']}]: {m['body'][:width]}")
    if not result['results']:
        out.append('No matches.')
    if result.get('note'):
        out.append('\n' + result['note'])
    return '\n'.join(out)


def run(con, args):
    res = find_quote(con, args.query, context=args.context, limit=args.k, subject_only=args.subject_only,
                     include_uncertain=args.include_uncertain)
    print(json.dumps(res, ensure_ascii=False, indent=1) if args.json else render(res))
