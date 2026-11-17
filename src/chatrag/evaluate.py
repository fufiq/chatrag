"""Reproducible retrieval evaluation; no model grades its own answers."""
import argparse
import json
from pathlib import Path
import statistics
import time

from chatrag import knowledge as rk
from chatrag.terms import normalize


def evaluate(db, models_dir, cases, lexical=False):
    rows = []
    with rk.connect(db) as con:
        for case in cases:
            started = time.monotonic()
            bundle = rk.search(con, case['query'], models_dir, limit=10, semantic=not lexical)
            hits = bundle['results']
            direct = None
            contextual = None
            if 'expected' in case:
                source, line = case['expected'].split(':')
                line = int(line)
                for i, hit in enumerate(hits, 1):
                    if hit['id'] == case['expected'] and direct is None:
                        direct = i
                    if Path(hit['source']).name == source and (
                        hit['line'] <= line <= hit['end_line'] or
                        any(m['line'] == line for m in hit['context_before'] + hit['context_after'])):
                        if contextual is None:
                            contextual = i
                passed = contextual is not None
            elif 'concept' in case:
                token = 'zconcept' + case['concept']
                passed = bool(hits) and all(token in normalize(hit['quote']) for hit in hits[:3])
            else:
                passed = not hits
            row = dict(query=case['query'], passed=passed, direct_rank=direct,
                       evidence_rank=contextual, ids=[h['id'] for h in hits],
                       latency_ms=round(1000*(time.monotonic()-started), 1),
                       warnings=bundle['warnings'], stages=bundle['stages'])
            row['split'] = case.get('split', 'regression')
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    known = [r for r, c in zip(rows, cases) if 'expected' in c]
    return dict(cases=len(rows), passed=sum(r['passed'] for r in rows),
                regression_passed=sum(r['passed'] for r in rows if r['split']=='regression'),
                control_passed=sum(r['passed'] for r in rows if r['split']=='control'),
                direct_hit_at_10=sum(r['direct_rank'] is not None for r in known)/len(known),
                evidence_hit_at_10=sum(r['evidence_rank'] is not None for r in known)/len(known),
                evidence_mrr=sum(1/r['evidence_rank'] if r['evidence_rank'] else 0 for r in known)/len(known),
                median_ms=statistics.median(r['latency_ms'] for r in rows), results=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', type=Path, default=rk.DEFAULT_DB)
    ap.add_argument('--models-dir', type=Path, default=rk.DEFAULT_MODELS)
    ap.add_argument('--cases', type=Path, default=Path(__file__).parents[2] / 'eval' / 'cases.json')
    ap.add_argument('--output', type=Path, default=Path('eval-report.json'))
    ap.add_argument('--lexical', action='store_true')
    args = ap.parse_args()
    report = evaluate(args.db, args.models_dir, json.loads(args.cases.read_text())['cases'], args.lexical)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}))


if __name__ == '__main__':
    main()
