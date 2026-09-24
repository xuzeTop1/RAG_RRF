# -*- coding: utf-8 -*-
"""How much of each returned Top-5 was actually judged, and how well the frozen
runs rebuild the annotation pool.

Two questions a reviewer keeps asking, answered from the published assets:

1. **Judged@5** — the share of the slots a system returns that carry a relevance
   vote. Unjudged slots are counted rather than dropped, so the number bounds the
   pooling depth instead of pretending it away. Reported under both denominators
   (slots actually returned, and a fixed 5) because the two diverge sharply for BM25
   on the unified index, where most questions return fewer than five hits.
2. **Pool reconstruction** — the manuscript describes the pool as the union of the
   Top-5 of BM25, Dense and Hybrid under both formulations. The frozen ranking files
   carry those six runs, but they are the post-annotation rebuild, so the union does
   not reproduce the pool. The gap is printed per question set, not glossed.

Nothing here feeds a manuscript number; it documents the boundary of what the pool
can support. Usage:  python judged_coverage_check.py
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fair_candidate_space as fcs  # noqa: E402

WORK = HERE / 'work'
OUT = HERE / 'results' / 'judged_coverage.txt'
DEPLOY = 'hybrid_raw_a20'
CONFIGS = ('bm25', 'dense', 'hybrid')


def read_jsonl(path):
    return [json.loads(line) for line in io.open(path, encoding='utf-8') if line.strip()]


def load_deployment(form):
    out = {}
    for row in read_jsonl(WORK / ('%s_%s.jsonl' % (DEPLOY, form))):
        hits = [h['key'] for h in sorted(row['hits'], key=lambda item: item['rank'])]
        out.setdefault(row['qid'], {})[row['config']] = hits
    return out


def pool_of(questions):
    return {qid: set(row['gold_votes']) for qid, row in questions.items()}


def union_top5(runs, depth=5):
    """Union of the first `depth` slots of every configuration and formulation."""
    out = defaultdict(set)
    for per_form in runs:
        for qid, by_config in per_form.items():
            for config in CONFIGS:
                out[qid] |= set(by_config.get(config, [])[:depth])
    return out


def judge_counts(ranking, judged):
    top = ranking[:5]
    hit = len([key for key in top if key in judged])
    return hit, len(top)


def main():
    questions = {row['qid']: row for row in read_jsonl(WORK / 'qa_100_human_v3.jsonl')}
    pool = pool_of(questions)
    covered = sorted(q for q in questions if q not in fcs.NO_GOLD)
    non_empty = sorted(q for q in questions if questions[q]['gold_ids'])
    lines = []
    add = lines.append

    add('annotation pool: %d pairs over %d questions (keys of `gold_votes`)'
        % (sum(len(p) for p in pool.values()), len(pool)))
    add('conditional_93 = the frozen run set minus NO_GOLD (%d qids)' % len(covered))
    if covered == non_empty:
        add('gate: the 93 named by NO_GOLD and the 93 with a non-empty gold_ids coincide -> ok')
    else:
        add('gate: the two ways of naming the 93 disagree: %s'
            % sorted(set(covered) ^ set(non_empty)))

    # ---- 1. can the six frozen runs rebuild the pool?
    add('')
    add('== pool reconstruction from the frozen Top-5 slots (the six runs the manuscript '
        'describes) ==')
    sets = {
        'deployment path (%s_* x2 forms x3 channels)' % DEPLOY:
            [load_deployment(f) for f in fcs.FORMS],
        'unified space (fair_rank_unified_* x2 forms x3 channels)':
            [fcs.load_rankings('unified', f) for f in fcs.FORMS],
        'baseline copy (fair_rank_base_* x2 forms x3 channels)':
            [fcs.load_rankings('base', f) for f in fcs.FORMS],
    }
    for label, runs in sets.items():
        union = union_top5(runs)
        exact = [q for q in covered if union.get(q, set()) == pool[q]]
        unjudged = sum(len(pool[q] - union.get(q, set())) for q in covered)
        extra = sum(len(union.get(q, set()) - pool[q]) for q in covered)
        add('  %-62s exact per-question match %2d/93 | pool pairs outside the union %3d '
            '| union items never judged %d' % (label, len(exact), unjudged, extra))
        if label.startswith('deployment'):
            fails = [q for q in covered if q not in exact]
            add('     questions whose pool the frozen runs do not reproduce (%d): %s'
                % (len(fails), ' '.join(fails)))

    # ---- 2. Judged@5
    add('')
    add('== Judged@5 over conditional_93 (share of returned Top-5 slots that carry a vote) ==')
    add('   denominator A = slots actually returned; B = a fixed 5 (short lists are '
        'under-coverage, not exempt)')
    for form in fcs.FORMS:
        unified = fcs.load_rankings('unified', form)
        deployed = load_deployment(form)
        strategies = [
            ('BM25, unified index', lambda q: unified[q]['bm25']),
            ('Dense, unified index', lambda q: unified[q]['dense']),
            ('RRF k=60 (frozen hybrid list, unified)', lambda q: unified[q]['hybrid']),
            ('RRF k=60 (recomputed from the two channel lists)',
             lambda q: fcs.fuse_keys(unified[q]['bm25'], unified[q]['dense'], k=60,
                                     space_of=fcs.space_by_channel)),
            ('RRF k=1 (recomputed)',
             lambda q: fcs.fuse_keys(unified[q]['bm25'], unified[q]['dense'], k=1,
                                     space_of=fcs.space_by_channel)),
            ('Interleave (recomputed, BM25 first)',
             lambda q: fcs.interleave(unified[q]['bm25'], unified[q]['dense'])),
            ('BM25, deployment path (159 private chunks)', lambda q: deployed[q]['bm25']),
            ('RRF k=60 (frozen hybrid list, deployment)', lambda q: deployed[q]['hybrid']),
        ]
        add('  --- formulation: %s ---' % form)
        for label, getter in strategies:
            num_a = den_a = num_b = 0
            short = 0
            for q in covered:
                ranking = getter(q)
                hit, returned = judge_counts(ranking, pool[q])
                num_a += hit
                den_a += returned
                num_b += hit
                short += 1 if len(ranking) < 5 else 0
            add('    %-48s A=%.4f  B=%.4f   questions returning <5 slots: %d/93'
                % (label, num_a / den_a, num_b / (len(covered) * 5.0), short))

    add('')
    add('reading: Dense is fully judged by construction (its Top-5 built the pool); the '
        'unified BM25 row is the pooling limit, not a ranking result.')
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\nwrote %s' % OUT.relative_to(HERE))


if __name__ == '__main__':
    main()
