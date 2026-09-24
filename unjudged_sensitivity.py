# -*- coding: utf-8 -*-
"""Unjudged-item sensitivity behind the Section 6.2 comparisons.

Pooling bounds every ranking metric: an entity outside the 847-pair annotation pool
carries no vote at all. The manuscript reports the pooled-label result and, in
Section 6.2, two re-scoring treatments that bound how much of the observed gap that
choice explains. This script recomputes them from the frozen inputs.

Treatments (unified candidate space, `conditional_93`, each formulation separately):

  control      as published — full list, gold = `gold_ids`, unjudged treated
               irrelevant. Re-derived here and diffed against the frozen
               `paired_bootstrap.csv`; a mismatch means this script and the
               manuscript have drifted apart.
  condensed    unjudged entities dropped from the returned list before scoring.
  promoted@5   unjudged entities inside the Top-5 window of any compared strategy
               counted relevant, with one relevance set per question so all three
               systems are scored against the same gold. This is the reading quoted
               in the manuscript.
  promoted     the same promotion over each system's whole list. Reported as a
               caution, not a result: it inflates the ideal DCG of whichever system
               returns more unjudged material, and it flips the sign of the RRF
               minus BM25 NDCG@5 difference.

Usage:  python unjudged_sensitivity.py
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fair_candidate_space as fcs  # noqa: E402

BENCH = fcs.BENCH_ROOT / '20260916-221246-rag-fair-candidate-space'
OUT = HERE / 'results' / 'unjudged_sensitivity.txt'
STRATEGIES = ('rrf_k60', 'bm25', 'dense')
COMPARISONS = (('rrf_k60', 'bm25'), ('rrf_k60', 'dense'))
TREATMENTS = ('control', 'condensed', 'promoted@5', 'promoted')


def load_case():
    questions = {row['qid']: row for row in fcs.read_jsonl(fcs.QA_PATH)}
    qids = sorted(q for q in questions if q not in fcs.NO_GOLD)
    gold = {q: set(questions[q]['gold_ids']) for q in qids}
    pool = {q: set(questions[q]['gold_votes']) for q in qids}
    rankings = {}
    for form in fcs.FORMS:
        raw = fcs.load_rankings('unified', form)
        rankings[form] = {
            q: {'rrf_k60': fcs.fuse_keys(raw[q]['bm25'], raw[q]['dense'], k=60,
                                         space_of=fcs.space_by_channel),
                'bm25': list(raw[q]['bm25']),
                'dense': list(raw[q]['dense'])}
            for q in qids}
    return qids, gold, pool, rankings


def score(treatment, ranking, gold, judged, window=None):
    if treatment == 'control':
        return ranking, gold
    if treatment == 'condensed':
        return [k for k in ranking if k in judged], gold
    if treatment == 'promoted@5':
        return ranking, gold | {k for k in window if k not in judged}
    if treatment == 'promoted':
        return ranking, gold | {k for k in ranking if k not in judged}
    raise ValueError(treatment)


def rows_for(treatment, form, qids, gold, pool, rankings, strategy):
    rows = []
    for q in qids:
        lists = rankings[form][q]
        window = (lists['rrf_k60'][:fcs.CUTOFF] + lists['bm25'][:fcs.CUTOFF]
                  + lists['dense'][:fcs.CUTOFF])
        ranking, g = score(treatment, lists[strategy], gold[q], pool[q], window)
        rows.append({'qid': q, **fcs.metrics(ranking, g)})
    return rows


def run(qids, gold, pool, rankings):
    out = {}
    for treatment in TREATMENTS:
        out[treatment] = {}
        for form in fcs.FORMS:
            by_strategy = {s: rows_for(treatment, form, qids, gold, pool, rankings, s)
                           for s in STRATEGIES}
            out[treatment][form] = {
                '%s - %s' % (left, right): fcs.cluster_bootstrap(by_strategy[left],
                                                                 by_strategy[right])
                for left, right in COMPARISONS}
    return out


def published_cells():
    """The control treatment must reproduce the frozen bootstrap archive, cell by cell."""
    want = {}
    for r in csv.DictReader(open(BENCH / 'paired_bootstrap.csv', encoding='utf-8-sig')):
        if 'oracle' in r['comparison'] or not r['comparison'].startswith('rrf_k60_unified - '):
            continue
        key = (r['queryForm'],
               r['comparison'].replace('rrf_k60_unified - bm25_unified', 'rrf_k60 - bm25')
                .replace('rrf_k60_unified - dense_unified', 'rrf_k60 - dense'),
               r['metric'])
        want[key] = (float(r['delta']), float(r['ciLow']), float(r['ciHigh']))
    return want


def main():
    qids, gold, pool, rankings = load_case()
    results = run(qids, gold, pool, rankings)
    lines = ['unjudged-item sensitivity, unified space, conditional_93 (n=%d), '
             '2,000 resamples, seed %d' % (len(qids), fcs.SEED + 17), '']

    drift = []
    want = published_cells()
    for (form, comparison, metric), (d, lo, hi) in sorted(want.items()):
        got = results['control'][form][comparison][metric]
        if (got['delta'], got['ci'][0], got['ci'][1]) != (d, lo, hi):
            drift.append('%s/%s/%s: archive %s, recomputed %s'
                         % (form, comparison, metric, (d, lo, hi),
                            (got['delta'], got['ci'])))
    lines.append('control vs frozen archive: %s'
                 % ('bit-exact on all %d cells' % len(want) if not drift
                    else 'DRIFT\n  ' + '\n  '.join(drift)))
    lines.append('')

    for treatment in TREATMENTS:
        lines.append('== %s ==' % treatment)
        for form in fcs.FORMS:
            for comparison, metrics in sorted(results[treatment][form].items()):
                for metric in ('ndcg@5', 'mrr'):
                    cell = metrics[metric]
                    lo, hi = cell['ci']
                    lines.append('  %-14s %-9s %-8s %+0.4f [%+0.4f,%+0.4f]%s'
                                 % (comparison, form, metric, cell['delta'], lo, hi,
                                    '   <- crosses or touches 0' if lo <= 0 <= hi else ''))
        lines.append('')
    lines.append('The manuscript quotes `condensed` and `promoted@5`. `promoted` is kept here '
                 'to show why the window matters: scoring whole lists against a gold set that '
                 'grows with the unjudged material a system returns rewards the systems that '
                 'return less of it.')

    text = '\n'.join(lines) + '\n'
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding='utf-8')
    print(text)
    if drift:
        raise SystemExit('control no longer reproduces the frozen archive')


if __name__ == '__main__':
    main()
