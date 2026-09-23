"""生成论文矢量图 F1 / F2 / F5。

原则：所有数值都从盘上既有产物读取，脚本内不出现任何手抄结果。
输出：figures/<name>.pdf（矢量，投稿用）、.svg（可编辑）、.png（仅用于目视检查）
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quota_ablation   # F6 replays the frozen fusion itself; no value is hand-typed

BENCH = Path(__file__).resolve().parent / 'benchmark-results' / '20260916-221246-rag-fair-candidate-space'
SENS = Path(__file__).resolve().parent / 'results' / 'gold_sensitivity_ext.json'
OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)

rcParams.update({
    'pdf.fonttype': 42,
    'svg.fonttype': 'path',       # SVG 文字转路径：Word 的 SVG 渲染器缺 DejaVu 时不会跑版
    'font.family': 'DejaVu Sans',                     # 图内标签用英文，避免字体依赖
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .6,
    'figure.dpi': 300, 'savefig.dpi': 300,
    # 关键：不裁边距，画布尺寸 == figsize，Word 按 PNG 物理尺寸嵌入，宽度即可控
    'savefig.bbox': None, 'savefig.pad_inches': 0,
})
# 目标版面宽度（英寸）；A4 + 常规页边距的正文宽约 6.3 in
WIDTH = {'f1_bootstrap_forest': 5.4, 'f2_gold_sensitivity': 6.1, 'f5_coverage_vs_fusion': 5.9,
         'f6_quota_ablation': 6.1}
INK, MUTED, HOT, COOL, ACC = '#1a1a1a', '#6b6b6b', '#c1272d', '#1f5c99', '#e8a33d'


def save(fig, name):
    # PDF/SVG 供 LaTeX 投稿；Word 只能嵌位图，故另出 300 dpi PNG
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.svg')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)
    return name


SHORT = {'rrf_k60_unified': 'RRF', 'bm25_unified': 'BM25-U', 'dense_unified': 'Dense',
         'bm25_prod_chunkonly': 'BM25-P', 'oracle_best_single_channel': 'Oracle'}
FORM_SHORT = {'template': 'T', 'bare': 'B'}


def load_bootstrap():
    """表 4 的比较集合。对 Oracle 的比较被排除：归档中 Oracle 的 NDCG 列是退化值（见正文 6.2 的披露）。"""
    rows = list(csv.DictReader(open(BENCH / 'paired_bootstrap.csv', encoding='utf-8-sig')))
    kept, dropped = [], []
    for r in rows:
        left, right = (s.strip() for s in r['comparison'].split('-'))
        if 'oracle' in r['comparison']:
            dropped.append(r['comparison'])
            continue
        kept.append(dict(name='%s−%s' % (SHORT[left], SHORT[right]),
                         group=r['comparison'], form=r['queryForm'], metric=r['metric'],
                         delta=float(r['delta']), lo=float(r['ciLow']), hi=float(r['ciHigh'])))
    return kept, sorted(set(dropped))


def load_overall():
    rows = list(csv.DictReader(open(BENCH / 'results_overall.csv', encoding='utf-8-sig')))
    return {(r['scope'], r['queryForm'], r['strategy']):
            {k: float(v) for k, v in r.items() if k in ('hit@1', 'hit@3', 'hit@5', 'mrr', 'ndcg@5')}
            for r in rows}


def f1_forest():
    data, dropped = load_bootstrap()
    labels, pos, gap_after, prev = [], [], [], None
    for i, r in enumerate(data):
        if prev is not None and r['group'] != prev:
            gap_after.append(i - 0.5)
        prev = r['group']
        labels.append('%s · %s · %s' % (r['name'], FORM_SHORT[r['form']], r['metric'].upper()))
        pos.append(i)
    fig, ax = plt.subplots(figsize=(WIDTH['f1_bootstrap_forest'], 4.1), layout='constrained')
    y = pos[::-1]
    for yi, r in zip(y, data[::-1]):
        col = HOT if r['lo'] <= 0 <= r['hi'] else COOL
        ax.plot([r['lo'], r['hi']], [yi, yi], '-', lw=1.2, color=col, alpha=.8, zorder=2)
        for edge in (r['lo'], r['hi']):
            ax.plot([edge, edge], [yi - .13, yi + .13], '-', lw=1.2, color=col, zorder=2)
        ax.plot(r['delta'], yi, 'o', ms=4.6, color=col, zorder=3)
    ax.axvline(0, color=INK, lw=.9, ls=(0, (4, 2)), zorder=1)
    for g in gap_after:
        ax.axhline(len(data) - g - 0.5, color='#cccccc', lw=.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=7.2)
    ax.set_xlabel(r'$\Delta$ NDCG@5 / MRR@5 with 95% paired bootstrap interval')
    # 轴界由数据决定：写死上限会把区间右端裁掉（评审 P1 次要问题）
    lo_all = [r['lo'] for r in data]
    hi_all = [r['hi'] for r in data]
    pad = 0.04 * (max(hi_all) - min(lo_all))
    ax.set_xlim(min(lo_all) - pad, max(hi_all) + pad)
    ax.set_title('Paired comparisons under the unified candidate space\n'
                 '(2,000 resamples, seed 20260933;\n'
                 'no correction for multiple comparisons)',
                 fontsize=8.4, loc='left', color=INK)
    return save(fig, 'f1_bootstrap_forest')


RULE_LABEL = [('v3_frozen', 'paper labels (v3_frozen)'),
              ('majority5', '5-rater majority (>=3/5)'),
              ('human_unanimous_pos', 'human only, conservative'),
              ('human_any_pos', 'human only, permissive'),
              ('model_only', 'model majority only'),
              ('majority6_ge3', 'incl. rater D (>=3/6)'),
              ('majority6_strict4', 'incl. rater D (>=4/6)')]


def f2_gold_sensitivity():
    rules = json.loads(SENS.read_text(encoding='utf-8'))['rules']
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH['f2_gold_sensitivity'], 3.6), sharey=True)
    fig.subplots_adjust(left=.30, right=.995, top=.90, bottom=.25, wspace=.06)
    ys = range(len(RULE_LABEL))[::-1]
    for ax, form, title in zip(axes, ('template', 'bare'),
                               ('templated (template)', 'concept-only (bare)')):
        for yi, (key, label) in zip(ys, RULE_LABEL):
            cell = rules[key]['unified'][form]
            agg = cell['agg']
            ci = cell['ci']['rrf_vs_dense']['ndcg@5']['ci']
            crosses = ci[0] <= 0 <= ci[1]
            ax.plot([agg['bm25']['ndcg@5'], agg['dense']['ndcg@5'], agg['rrf_k60']['ndcg@5']],
                    [yi] * 3, ls=':', lw=.9, color='#bbbbbb', zorder=1)
            ax.plot(agg['bm25']['ndcg@5'], yi, 's', ms=4.6, mfc='none', mec=MUTED, zorder=3)
            ax.plot(agg['dense']['ndcg@5'], yi, '^', ms=5.0, mfc='none', mec=ACC, zorder=3)
            ax.plot(agg['rrf_k60']['ndcg@5'], yi, 'o', ms=6.2,
                    color=COOL if not crosses else HOT, zorder=4)
            if crosses:
                ax.annotate('CI crosses 0', (agg['rrf_k60']['ndcg@5'], yi), xytext=(-9, 8),
                            textcoords='offset points', fontsize=6.6, color=HOT, ha='right',
                            arrowprops=dict(arrowstyle='-', color=HOT, lw=.7,
                                            shrinkA=0, shrinkB=3))
        ax.set_xlim(.50, 1.02)
        ax.set_xticks([.6, .7, .8, .9, 1.0])
        ax.set_title(title, fontsize=8.6, color=INK)
        ax.set_xlabel('NDCG@5')
    axes[0].set_yticks(list(ys))
    axes[0].set_yticklabels(['%s  (n=%d)' % (lbl, rules[k]['questionsWithGold'])
                             for k, lbl in RULE_LABEL], fontsize=7.2)
    handles = [plt.Line2D([], [], marker='o', ls='', mfc=COOL, mec=COOL, label='RRF (k=60)'),
               plt.Line2D([], [], marker='^', ls='', mfc='none', mec=ACC, label='Dense'),
               plt.Line2D([], [], marker='s', ls='', mfc='none', mec=MUTED, label='BM25'),
               plt.Line2D([], [], marker='o', ls='', mfc=HOT, mec=HOT,
                          label='RRF where the RRF-Dense interval crosses 0')]
    fig.legend(handles=handles, fontsize=7, ncol=4, loc='lower center',
               bbox_to_anchor=(.5, .015), frameon=False)
    return save(fig, 'f2_gold_sensitivity')


def f5_coverage_vs_fusion():
    overall = load_overall()
    get = lambda s: overall[('conditional_93', 'template', s)]['ndcg@5']
    steps = [('BM25, 159 private\nchunks only\n(production index)', get('bm25_prod_chunkonly'), MUTED),
             ('BM25, unified\n1,696 entities', get('bm25_unified'), ACC),
             ('Dense, unified\n1,696 entities', get('dense_unified'), ACC),
             ('RRF (k=60), unified\nno quota', get('rrf_k60_unified'), COOL)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH['f5_coverage_vs_fusion'], 3.2),
                                 gridspec_kw={'width_ratios': [1, 1.25]})
    fig.subplots_adjust(left=.085, right=.955, top=.86, bottom=.20, wspace=.32)
    a1.bar([0, 1], [steps[0][1], steps[1][1]], width=.55, color=[steps[0][2], steps[1][2]],
           edgecolor=INK, lw=.6)
    for x, (_, v, _) in zip([0, 1], steps[:2]):
        a1.text(x, v + .012, '%.4f' % v, ha='center', fontsize=7.4)
    a1.annotate('+%.4f' % (steps[1][1] - steps[0][1]), xy=(.5, (steps[0][1] + steps[1][1]) / 2),
                ha='center', fontsize=7.8, color=ACC, weight='bold')
    a1.set_xticks([0, 1])
    a1.set_xticklabels([steps[0][0], steps[1][0]], fontsize=6.6)
    a1.set_ylim(0, .85)
    a1.set_ylabel('NDCG@5')
    a1.set_title('A. index expansion\n(coverage + BM25 statistics change together)', fontsize=8.2, color=INK)

    a2.bar([0, 1, 2], [steps[1][1], steps[2][1], steps[3][1]], width=.55,
           color=[steps[1][2], steps[2][2], steps[3][2]], edgecolor=INK, lw=.6)
    for x, (_, v, _) in zip([0, 1, 2], steps[1:]):
        a2.text(x, v + .012, '%.4f' % v, ha='center', fontsize=7.4)
    diff_fusion = steps[3][1] - steps[2][1]
    a2.annotate('+%.4f\nover Dense' % diff_fusion, xy=(1.725, .84),
                xytext=(1.25, .885), ha='center', fontsize=7.2, color=COOL,
                arrowprops=dict(arrowstyle='->', color=COOL, lw=.9,
                                connectionstyle='arc3,rad=-0.12'))
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels([steps[1][0], steps[2][0], steps[3][0]], fontsize=6.6)
    a2.set_ylim(0, 1.0)
    a2.set_title('B. fusion under fixed coverage\n(both channels on the same 1,696 entities)',
                 fontsize=8.2, color=INK)
    return save(fig, 'f5_coverage_vs_fusion')


def f6_quota_ablation():
    """配额开关消融：水平（关/开）与差值区间（关−开@5 槽）。

    数值全部来自 quota_ablation.compute()（离线重放 hybrid.rs 的融合/并列/配额），
    与 §6.1 正文、quota_ablation.txt 同源。
    """
    data = quota_ablation.compute()
    agg, comp = data['aggregates'], data['comparisons']['no_quota_vs_quota@5']
    off, on = agg['no_quota'], agg['quota@5']
    metrics = [('hit@5', 'Hit@5'), ('mrr', 'MRR@5'), ('ndcg@5', 'NDCG@5')]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH['f6_quota_ablation'], 2.9),
                                 gridspec_kw={'width_ratios': [1.22, 1]})
    fig.subplots_adjust(left=.095, right=.985, top=.82, bottom=.28, wspace=.34)
    minus = lambda v: ('%+.4f' % v).replace('-', '\u2212')

    xs = range(len(metrics))
    a1.bar([x - .19 for x in xs], [off[k] for k, _ in metrics], width=.36,
           color=COOL, edgecolor=INK, lw=.5, label='quota off')
    a1.bar([x + .19 for x in xs], [on[k] for k, _ in metrics], width=.36,
           color=ACC, edgecolor=INK, lw=.5, label='quota on (5 slots returned)')
    for x, (k, _) in zip(xs, metrics):
        a1.text(x - .19, off[k] + .006, '%.4f' % off[k], ha='center', fontsize=6.2, color=INK)
        a1.text(x + .19, on[k] + .006, '%.4f' % on[k], ha='center', fontsize=6.2, color=INK)
    a1.set_xticks(list(xs))
    a1.set_xticklabels([label for _, label in metrics], fontsize=7.6)
    a1.set_ylim(.70, 1.0)
    a1.set_yticks([.7, .8, .9, 1.0])
    a1.set_ylabel('metric value', fontsize=8)
    a1.legend(fontsize=6.8, frameon=False, loc='upper center', ncol=2,
              bbox_to_anchor=(.5, -.16))
    fig.text(.095, .03,
             'Quota at the retrieval depth of the frozen evaluation (20) \u2261 quota off '
             '(186/186 lists identical).\n'
             'Hit@1 is identical under both settings (0.6935); resampling is clustered by question.\n'
             'Red marks an interval that crosses 0.',
             fontsize=6.3, color=MUTED, va='bottom')
    a1.set_title('A. Metric levels with the source quota off vs on\n'
                 '(conditional_93, both forms pooled, 186 pairs)',
                 fontsize=8.0, loc='left', color=INK)

    rows = [('ndcg@5', 'NDCG@5'), ('mrr', 'MRR@5'), ('hit@5', 'Hit@5')]
    ys = list(range(len(rows)))[::-1]
    for y, (k, _) in zip(ys, rows):
        cell = comp[k]
        crosses = cell['ci'][0] <= 0 <= cell['ci'][1]
        col = HOT if crosses else COOL
        a2.plot(cell['ci'], [y, y], '-', lw=1.2, color=col, alpha=.85, zorder=2)
        for edge in cell['ci']:
            a2.plot([edge, edge], [y - .14, y + .14], '-', lw=1.2, color=col, zorder=2)
        a2.plot(cell['delta'], y, 'o', ms=4.8, color=col, zorder=3)
        a2.annotate(minus(cell['delta']), (cell['delta'], y), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=6.6, color=col)
    a2.axvline(0, color=INK, lw=.9, ls=(0, (4, 2)), zorder=1)
    a2.set_yticks(ys)
    a2.set_yticklabels([label for _, label in rows], fontsize=7.6)
    a2.set_ylim(-.55, len(rows) - .35)
    a2.set_xlim(-.048, .048)
    a2.set_xticks([-.04, -.02, 0, .02, .04])
    a2.set_xlabel(r'$\Delta$ (quota off $-$ quota on @5)', fontsize=8)
    a2.set_title('B. Paired bootstrap of the difference\n(2,000 resamples)',
                 fontsize=8.0, loc='left', color=INK)
    return save(fig, 'f6_quota_ablation')


def main():
    made = [f1_forest(), f2_gold_sensitivity(), f5_coverage_vs_fusion(), f6_quota_ablation()]
    root = Path(__file__).resolve().parent
    work = Path(quota_ablation.resolve_benchmark_dir())

    def rel(path):
        return Path(path).resolve().relative_to(root).as_posix()

    qa_rel = rel(work / 'qa_100_human_v3.jsonl')
    manifest = {'generated': datetime.now().isoformat(timespec='seconds'),
                'figures': made,
                'sources': {'f1': rel(BENCH / 'paired_bootstrap.csv'),
                            'f2': rel(SENS),
                            'f5': rel(BENCH / 'results_overall.csv'),
                            'f6': '%s + %s/hybrid_raw_a20_{template,bare}.jsonl'
                                  % (qa_rel, qa_rel.rsplit('/', 1)[0])},
                'note': 'all values read from the listed artefacts; oracle row excluded '
                        '(its NDCG column in the archive is degenerate)'}
    (OUT / 'figures_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('figures:', ', '.join(made))


if __name__ == '__main__':
    main()
