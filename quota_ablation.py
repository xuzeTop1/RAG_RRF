# -*- coding: utf-8 -*-
"""Source-quota on/off ablation on the frozen deployment-path rankings (Section 6.1).

The deployment fusion is replayed offline, exactly as implemented in
`src-tauri/src/rag/hybrid.rs`:

  * `fuse_rrf`      — union of the two channel lists, score 1/(k + rank) summed;
  * `tie_break_key` — (dual, offset rank, source space, chunk id);
  * `apply_source_quota` — per-source floor, then fill in fusion order.

Three steps:
  1. Fidelity: the replayed quota-on list at depth 20 must reproduce the frozen
     `hybrid_raw_a20_*` hybrid list key for key and score for score.  If it does
     not, everything below is meaningless.
  2. The quota at depth 20 (the depth the frozen evaluation used) changes nothing.
  3. At the 5 slots the deployment actually returns, quota off vs quota on, with a
     question-clustered paired bootstrap over conditional_93.

Frozen inputs expected in --benchmark-dir (pseudonymised, checked into `work/`):
  qa_100_human_v3.jsonl, hybrid_raw_a20_template.jsonl, hybrid_raw_a20_bare.jsonl

Usage:
  python quota_ablation.py                 # defaults to ./work
  python quota_ablation.py --benchmark-dir <dir with the frozen files>
"""
import argparse
import io
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numerics import csum  # noqa: E402

K_RRF = 60          # hybrid.rs DEFAULT_K_RRF
FLOOR = 2           # hybrid.rs PER_SOURCE_FLOOR
BOOT_SEED = 20260916 + 13   # same stream as the paper's other bootstrap runs
N_BOOT = 2000
NO_GOLD = ['q073', 'q074', 'q087', 'q092', 'q095', 'q099', 'q100']  # conditional_94 -> 93


def load_jsonl(path):
    return [json.loads(line) for line in io.open(path, encoding='utf-8') if line.strip()]


def load_raw(path):
    out = {}
    for r in load_jsonl(path):
        out.setdefault(r['qid'], {})[r['config']] = sorted(r['hits'], key=lambda h: h['rank'])
    return out


def source_space(key):
    """hybrid.rs `source_space`: 0 = PrivateChunk, 1 = the rest (incl. Dense)."""
    return 0 if key.startswith('private_chunk:') else 1


def fuse_rrf(bm25_keys, dense_keys, k=K_RRF):
    """fuse_rrf + tie_break_key, as in hybrid.rs."""
    order = []
    for key in list(bm25_keys) + list(dense_keys):
        if key not in order:
            order.append(key)
    scored = []
    for key in order:
        b = bm25_keys.index(key) + 1 if key in bm25_keys else None
        d = dense_keys.index(key) + 1 if key in dense_keys else None
        scored.append({'key': key,
                       'score': (1.0 / (k + b) if b else 0.0) + (1.0 / (k + d) if d else 0.0),
                       'b': b, 'd': d})

    def tie_break(e):
        dual = e['b'] is not None and e['d'] is not None
        rank = min(r for r in (e['b'], e['d']) if r is not None)
        space = source_space(e['key'])
        private_first = (rank % 2) == 1        # private source first at odd ranks
        offset = 0 if (space == 0) == private_first else 1
        return (0 if dual else 1, (rank - 1) * 2 + offset, space, e['key'])

    scored.sort(key=lambda e: (-e['score'], tie_break(e)))
    return scored


def apply_source_quota(sorted_list, top_k, floor=FLOOR):
    """apply_source_quota, as in hybrid.rs: keep `floor` per space, then fill."""
    if len(sorted_list) <= top_k:
        return list(sorted_list)
    spaces = {source_space(e['key']) for e in sorted_list}
    if spaces != {0, 1}:
        return sorted_list[:top_k]
    selected = [False] * len(sorted_list)
    chosen = 0
    for space in (0, 1):
        taken = 0
        for i, entry in enumerate(sorted_list):
            if taken >= floor or chosen >= top_k:
                break
            if not selected[i] and source_space(entry['key']) == space:
                selected[i] = True
                taken += 1
                chosen += 1
    for i in range(len(selected)):
        if chosen >= top_k:
            break
        if not selected[i]:
            selected[i] = True
            chosen += 1
    return [e for e, keep in zip(sorted_list, selected) if keep]


def metrics(ranking, gold):
    hit1 = 1.0 if any(k in gold for k in ranking[:1]) else 0.0
    hit3 = 1.0 if any(k in gold for k in ranking[:3]) else 0.0
    hit5 = 1.0 if any(k in gold for k in ranking[:5]) else 0.0
    mrr = 0.0
    for i, k in enumerate(ranking[:5]):
        if k in gold:
            mrr = 1.0 / (i + 1)
            break
    dcg = csum(1.0 / math.log2(i + 2) for i, k in enumerate(ranking[:5]) if k in gold)
    idcg = csum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    return hit1, hit3, hit5, mrr, (dcg / idcg if idcg else 0.0)


def agg(rows):
    n = len(rows)
    return {k: csum(m[i] for _, m in rows) / n
            for i, k in enumerate(('hit@1', 'hit@3', 'hit@5', 'mrr', 'ndcg@5'))}


METRIC_KEYS = ('hit@1', 'hit@3', 'hit@5', 'mrr', 'ndcg@5')
METRIC_SLOT = {'hit@1': 0, 'hit@3': 1, 'hit@5': 2, 'mrr': 3, 'ndcg@5': 4}


def boot(a, b, n_boot=N_BOOT, metrics=('ndcg@5', 'mrr')):
    """Question-clustered paired bootstrap over (qid, form) pairs: a - b.

    The resampling stream is independent of `metrics`, so asking for extra
    metrics never moves the values already reported for NDCG@5 / MRR@5.
    """
    ma, mb = dict(a), dict(b)
    by_q = defaultdict(list)
    for key in sorted(set(ma) & set(mb)):
        by_q[key[0]].append((ma[key], mb[key]))
    qs = sorted(by_q)
    rng = random.Random(BOOT_SEED)
    idx = [METRIC_SLOT[m] for m in metrics]
    draws = {m: [] for m in metrics}
    observed = {m: 0.0 for m in metrics}
    n_pairs = 0
    for q in qs:
        for va, vb in by_q[q]:
            n_pairs += 1
            for m, i in zip(metrics, idx):
                observed[m] += va[i] - vb[i]
    for _ in range(n_boot):
        sums = [0.0] * len(idx)
        c = 0
        for q in [qs[rng.randrange(len(qs))] for _ in qs]:
            for va, vb in by_q[q]:
                for k, i in enumerate(idx):
                    sums[k] += va[i] - vb[i]
                c += 1
        for m, s in zip(metrics, sums):
            draws[m].append(s / c)
    out = {}
    for m in metrics:
        values = sorted(draws[m])
        out[m] = {'delta': observed[m] / n_pairs,               # 观测配对均值差 = 点估计
                  'boot_mean': csum(values) / n_boot,           # 仅作重抽样中心，不对外报
                  'ci': [values[int(.025 * n_boot)], values[int(.975 * n_boot) - 1]]}
    return out


def resolve_benchmark_dir(explicit=None):
    """Frozen files: explicit flag, then the pseudonymised `work/` next to this script."""
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parent / 'work'


def compute(benchmark_dir=None):
    """Replay the frozen fusion and return every number the ablation reports."""
    work = resolve_benchmark_dir(benchmark_dir)
    gold = {}
    for r in load_jsonl(work / 'qa_100_human_v3.jsonl'):
        gold[r['qid']] = set(r.get('gold_ids') or [])
    conditional = sorted(q for q in gold if q not in NO_GOLD)
    raw = {form: load_raw(work / ('hybrid_raw_a20_%s.jsonl' % form)) for form in ('template', 'bare')}

    var = defaultdict(list)
    mismatched, checked, changed_at_5, unchanged_at_20 = [], 0, [], 0
    for form, rows in raw.items():
        for qid in sorted(rows):
            bm = [h['key'] for h in rows[qid].get('bm25', [])]
            dn = [h['key'] for h in rows[qid].get('dense', [])]
            fused = fuse_rrf(bm, dn)
            frozen = rows[qid].get('hybrid', [])

            replayed, reference = apply_source_quota(fused, 20), frozen
            checked += 1
            if [e['key'] for e in replayed] == [e['key'] for e in fused[:20]]:
                unchanged_at_20 += 1
            if [e['key'] for e in replayed] != [h['key'] for h in reference] or \
               any(abs(a['score'] - b['rrfScore']) > 1e-12 for a, b in zip(replayed, reference)):
                mismatched.append((qid, form))

            no_quota, quota5 = fused[:5], apply_source_quota(fused, 5)
            if [e['key'] for e in no_quota] != [e['key'] for e in quota5]:
                changed_at_5.append((qid, form))

            g = gold[qid] if qid in gold else set()
            var['no_quota'].append(((qid, form), metrics([e['key'] for e in no_quota], g)))
            var['quota@5'].append(((qid, form), metrics([e['key'] for e in quota5], g)))
            var['prod_quota@20'].append(((qid, form), metrics([h['key'] for h in frozen[:5]], g)))

    aggregates = {name: agg(var[name]) for name in ('no_quota', 'quota@5', 'prod_quota@20')}
    comparisons = {'no_quota_vs_quota@5': boot(var['no_quota'], var['quota@5'],
                                              metrics=('ndcg@5', 'mrr', 'hit@5', 'hit@1')),
                   'no_quota_vs_prod@20': boot(var['no_quota'], var['prod_quota@20'],
                                               metrics=('ndcg@5', 'mrr', 'hit@5', 'hit@1'))}
    return {'benchmark_dir': str(work), 'n_pairs': checked,
            'n_questions': len(conditional), 'n_ranking_questions': len(raw['template']),
            'n_form_pairs_replayed': checked - len(mismatched),
            'mismatched': mismatched, 'unchanged_at_20': unchanged_at_20,
            'changed_at_5': changed_at_5, 'aggregates': aggregates,
            'comparisons': comparisons, 'boot_seed': BOOT_SEED, 'n_boot': N_BOOT,
            'k_rrf': K_RRF, 'floor': FLOOR,
            'only_questions': sorted(raw['template']) == conditional}


def render(data):
    """Human-readable report of compute()'s numbers."""
    checked, changed = data['n_pairs'], len(data['changed_at_5'])
    report = ['fidelity: replayed quota@20 == frozen production list for %d/%d (qid, form) pairs'
              % (data['n_form_pairs_replayed'], checked)]
    if data['mismatched']:
        report.append('   MISMATCHES: %s' % data['mismatched'][:8])
    report.append('quota@20 vs no quota: identical 20-slot lists for %d/%d pairs (the depth the frozen '
                  'evaluation used)' % (data['unchanged_at_20'], checked))
    report.append('quota@5 changes the 5-slot list for %d/%d pairs' % (changed, checked))

    report.append('')
    report.append('== metrics over conditional_93 (both forms pooled, n=%d pairs) ==' % checked)
    report.append('%-22s %8s %8s %8s %8s' % ('variant', 'hit@1', 'hit@5', 'mrr', 'ndcg@5'))
    for name in ('no_quota', 'quota@5', 'prod_quota@20'):
        m = data['aggregates'][name]
        report.append('%-22s %8.4f %8.4f %8.4f %8.4f'
                      % (name, m['hit@1'], m['hit@5'], m['mrr'], m['ndcg@5']))

    report.append('')
    report.append('== paired bootstrap (2,000 resamples, seed %d, metrics in the order given) =='
                  % data['boot_seed'])
    for label, key in [('no quota  - quota@5', 'no_quota_vs_quota@5'),
                       ('no quota  - frozen production (A20 slice)', 'no_quota_vs_prod@20')]:
        parts = ['d%-9s=%+.4f CI=[%+.4f,%+.4f]' % (m, c['delta'], c['ci'][0], c['ci'][1])
                 for m, c in data['comparisons'][key].items()]
        report.append('%s : %s' % (label, '  '.join(parts)))

    report.append('')
    report.append('question set: rankings=%d, conditional_93=%d, identical=%s'
                  % (data['n_ranking_questions'], data['n_questions'], data['only_questions']))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benchmark-dir', default=None,
                    help='directory holding the frozen qa/ranking files')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent
                                         / 'results' / 'quota_ablation.txt'))
    args = ap.parse_args()

    data = compute(args.benchmark_dir)
    txt = '\n'.join(render(data))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt, encoding='utf-8')
    print(txt)
    print('\nwritten: %s' % args.out)


if __name__ == '__main__':
    main()
