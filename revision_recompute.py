# -*- coding: utf-8 -*-
"""Revision-round recomputation on the frozen artifacts (no re-retrieval, no model calls).

Produces the four numbers the review asked for:
  1. both-form central effects with qid-clustered paired bootstrap (the abstract
     reports both-form point estimates but only template/bare intervals);
  2. a CORRECT post-hoc single-channel oracle = per query, per metric max(BM25,
     Dense), replacing the archived `oracle_best` (first-hit position only, any
     hit -> ndcg@5 = 1);
  3. private-source exposure and per-target-source breakdown for RQ3;
  4. whether a parent-document grouping is derivable from the frozen keys (P1-4).

Everything is read from the frozen files through `fair_candidate_space`'s own
functions, so the seed (SEED + 17 = 20260933) and the qid clustering match §5.3.
"""
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fair_candidate_space as fcs  # noqa: E402

OUT = HERE / 'results' / 'revision_recompute.json'
METRICS = ('ndcg@5', 'mrr', 'hit@5', 'hit@1')


def quota_exposure():
    """P1-7: private-source exposure and private-target recall under quota off vs on.

    Replays the frozen A20 rankings through quota_ablation's own fuse/quota code so
    the numbers match Section 6.1 exactly; exposure is the share of the five returned
    slots taken from the private space.
    """
    import statistics as st
    import quota_ablation as qa

    work = qa.resolve_benchmark_dir()
    gold = {r['qid']: set(r.get('gold_ids') or []) for r in qa.load_jsonl(work / 'qa_100_human_v3.jsonl')}
    target = {r['qid']: r.get('target_source') for r in qa.load_jsonl(work / 'qa_100_human_v3.jsonl')}
    share, priv_hit, priv_nd = {'off': [], 'on': [], 'prod': []}, {'off': [], 'on': []}, {'off': [], 'on': []}
    for form in ('template', 'bare'):
        rows = qa.load_raw(work / ('hybrid_raw_a20_%s.jsonl' % form))
        for qid in sorted(rows):
            if qid in qa.NO_GOLD:
                continue
            bm = [h['key'] for h in rows[qid].get('bm25', [])]
            dn = [h['key'] for h in rows[qid].get('dense', [])]
            fused = qa.fuse_rrf(bm, dn)
            off = [e['key'] for e in fused[:5]]
            on = [e['key'] for e in qa.apply_source_quota(fused, 5)]
            prod = [h['key'] for h in rows[qid].get('hybrid', [])][:5]
            for name, lst in (('off', off), ('on', on), ('prod', prod)):
                share[name].append(sum(1 for k in lst if qa.source_space(k) == 0) / 5.0)
            if target.get(qid) == 'private_chunk':
                g = gold[qid]
                for name, lst in (('off', off), ('on', on)):
                    m = qa.metrics(lst, g)
                    priv_hit[name].append(m[qa.METRIC_SLOT['hit@5']])
                    priv_nd[name].append(m[qa.METRIC_SLOT['ndcg@5']])
    out = {'pairs': len(share['off']),
           'private_share_top5': {k: st.fmean(v) for k, v in share.items()},
           'private_target': {'n': len(priv_hit['off']),
                              'hit@5': {k: st.fmean(v) for k, v in priv_hit.items()},
                              'ndcg@5': {k: st.fmean(v) for k, v in priv_nd.items()}}}
    print('\n== 配额开/关下的私有来源曝光（A20 重放，conditional_93，两形态）==')
    print('  题次=%d  私有占 Top-5 槽位：关配额 %.4f / 开配额 %.4f / 冻结生产 %.4f'
          % (out['pairs'], out['private_share_top5']['off'], out['private_share_top5']['on'],
             out['private_share_top5']['prod']))
    print('  目标为私有切片的题次 n=%d：Hit@5 关 %.4f 开 %.4f；NDCG@5 关 %.4f 开 %.4f'
          % (out['private_target']['n'], out['private_target']['hit@5']['off'],
             out['private_target']['hit@5']['on'], out['private_target']['ndcg@5']['off'],
             out['private_target']['ndcg@5']['on']))
    return out


def main():
    qa = {r['qid']: r for r in fcs.read_jsonl(fcs.QA_PATH)}
    gold = {q: set(r['gold_ids']) for q, r in qa.items()}
    rows = {}
    for form in fcs.FORMS:
        rk = fcs.load_rankings('unified', form)
        per = {}
        for qid, cfg in sorted(rk.items()):
            g = gold.get(qid, set())
            if not g:
                continue
            bm, dn = cfg['bm25'], cfg['dense']
            rrf = fcs.fuse_keys(bm, dn, k=60, space_of=fcs.space_by_channel)
            m = {'bm25': fcs.metrics(bm, g), 'dense': fcs.metrics(dn, g), 'rrf': fcs.metrics(rrf, g)}
            top5 = rrf[:5]
            m['oracle'] = {k: max(m['bm25'][k], m['dense'][k]) for k in METRICS}
            m['_target'] = qa[qid]['target_source']
            per[qid] = m
        rows[form] = per
        print('%-9s 有金标且有排名的题=%d' % (form, len(per)))

    def as_list(left, right):
        qids = sorted(set(rows[fcs.FORMS[0]]) & set(rows[fcs.FORMS[1]]))
        a, b = [], []
        for form in fcs.FORMS:
            for q in qids:
                a.append({'qid': q, 'ndcg@5': rows[form][q][left]['ndcg@5'],
                          'mrr': rows[form][q][left]['mrr']})
                b.append({'qid': q, 'ndcg@5': rows[form][q][right]['ndcg@5'],
                          'mrr': rows[form][q][right]['mrr']})
        return a, b

    report = {'seed_actual': fcs.SEED + 17, 'n_pairs_per_form': len(rows['template']), 'both': {},
              'per_form': {}, 'levels_both': {}, 'exposure': {}, 'by_target': {}, 'key_shape': {}}

    for left, right, label in (('rrf', 'dense', 'RRF-Dense'), ('rrf', 'bm25', 'RRF-BM25'),
                               ('rrf', 'oracle', 'RRF-Oracle(正确的逐题逐指标上界)')):
        a, b = as_list(left, right)
        ci = fcs.cluster_bootstrap(a, b)
        report['both'][label] = ci
        print('both  %-34s ΔNDCG=%+.5f CI=[%+.5f,%+.5f]  ΔMRR=%+.5f CI=[%+.5f,%+.5f]'
              % (label, ci['ndcg@5']['delta'], *ci['ndcg@5']['ci'],
                 ci['mrr']['delta'], *ci['mrr']['ci']))
        for form in fcs.FORMS:
            qids = sorted(rows[form])
            fa = [{'qid': q, 'ndcg@5': rows[form][q][left]['ndcg@5'], 'mrr': rows[form][q][left]['mrr']}
                  for q in qids]
            fb = [{'qid': q, 'ndcg@5': rows[form][q][right]['ndcg@5'], 'mrr': rows[form][q][right]['mrr']}
                  for q in qids]
            report['per_form'].setdefault(label, {})[form] = fcs.cluster_bootstrap(fa, fb)

    for strat in ('rrf', 'dense', 'bm25', 'oracle'):
        report['levels_both'][strat] = {
            m: statistics.fmean([rows[f][q][strat][m] for f in fcs.FORMS for q in rows[f]])
            for m in METRICS}
    print()
    for strat, lv in report['levels_both'].items():
        print('  both 水平 %-7s NDCG@5=%.4f MRR@5=%.4f Hit@5=%.4f Hit@1=%.4f'
              % (strat, lv['ndcg@5'], lv['mrr'], lv['hit@5'], lv['hit@1']))

    for strat in ('rrf', 'dense', 'bm25'):
        share = []
        for form in fcs.FORMS:
            rk = fcs.load_rankings('unified', form)
            for q in sorted(rows[form]):
                lst = rk[q]['bm25'] if strat == 'bm25' else (
                    rk[q]['dense'] if strat == 'dense' else
                    fcs.fuse_keys(rk[q]['bm25'], rk[q]['dense'], k=60, space_of=fcs.space_by_channel))
                share.append(sum(1 for key in lst[:5] if key.startswith('private_chunk:')) / 5.0)
        report['exposure'][strat] = statistics.fmean(share)
    print()
    for strat, v in report['exposure'].items():
        print('  私有切片占 Top-5 槽位比例  %-6s = %.4f' % (strat, v))

    targets = sorted({rows[f][q]['_target'] for f in fcs.FORMS for q in rows[f]})
    for t in targets:
        sub = {strat: statistics.fmean([rows[f][q][strat]['ndcg@5'] for f in fcs.FORMS
                                        for q in rows[f] if rows[f][q]['_target'] == t])
               for strat in ('rrf', 'dense', 'bm25')}
        n = sum(1 for f in fcs.FORMS for q in rows[f] if rows[f][q]['_target'] == t)
        report['by_target'][t] = {'n_pairs': n, **sub}
        print('  目标来源 %-14s n=%3d  RRF=%.4f Dense=%.4f BM25=%.4f'
              % (t, n, sub['rrf'], sub['dense'], sub['bm25']))

    report['quota_exposure'] = quota_exposure()

    report['key_shape']['samples'] = sorted({k for q in list(gold)[:3] for k in gold[q]})[:6]
    report['key_shape']['note'] = ('ids are opaque after desensitize.py '
                                   '(knowledge_node:kn-*, question:qs-*, private_chunk:pc-*) '
                                   'and carry no parent-document field, so document-level '
                                   'deduplication needs a corpus-side mapping the frozen '
                                   'rankings cannot supply')
    print('\n  key samples:', report['key_shape']['samples'])

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written:', OUT.name)


main()
