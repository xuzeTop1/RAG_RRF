# -*- coding: utf-8 -*-
"""Re-run every published stage and check it reproduces the checked-in artefacts.

The frozen ranking and label files under `work/` are the inputs; everything under
`results/`, `figures/` and `benchmark-results/<timestamp>-…/` is derived.

Figures are checked through `figures/figures_manifest.json`, which records the
values each figure plots; the image bytes themselves are reported but not
required to match, because the PNG/SVG/PDF encoders stamp the matplotlib
version and resolve fonts through whatever the host has installed. Float
accumulation in the pipeline goes through `numerics.csum`, so the numbers do not
depend on the interpreter's own `sum()` behaviour (which changed in CPython 3.12).

Usage:  python reproduce.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'benchmark-results' / '20260916-221246-rag-fair-candidate-space'
REPORT_CSVS = ('results_overall.csv', 'paired_bootstrap.csv', 'per_query_metrics.csv')

STAGES = [
    ('quota ablation (Sec. 6.1)', ['quota_ablation.py'],
     ['results/quota_ablation.txt']),
    ('gold-label sensitivity (Sec. 6.6)', ['gold_sensitivity_ext.py'],
     ['results/gold_sensitivity_ext.json', 'results/gold_sensitivity_ext.txt']),
    ('revision recompute (RQ2, exposure)', ['revision_recompute.py'],
     ['results/revision_recompute.json']),
    ('unified-space report (Tables 3-4)', ['fair_candidate_space.py', 'report'], None),
    ('pooling coverage (Sec. 5.1 limits)', ['judged_coverage_check.py'],
     ['results/judged_coverage.txt']),
    ('unjudged-item sensitivity (Sec. 6.2)', ['unjudged_sensitivity.py'],
     ['results/unjudged_sensitivity.txt']),
    ('figures 2-5', ['make_figures.py'], ['figures/figures_manifest.json']),
]

# Reported, never fatal: raster/vector bytes move with the plotting environment.
INFORMATIVE = ['figures/f1_bootstrap_forest.png', 'figures/f2_gold_sensitivity.png',
               'figures/f5_coverage_vs_fusion.png', 'figures/f6_quota_ablation.png']


def run(command):
    proc = subprocess.run([sys.executable] + command, cwd=HERE,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        print((proc.stdout or '')[-1200:])
        print((proc.stderr or '')[-1200:])
    return proc.returncode


def main():
    if not (HERE / 'work' / 'qa_100_human_v3.jsonl').exists():
        raise SystemExit('work/ is missing: repopulate it with desensitize.py from the '
                         'private checkout, or unpack a published release')
    print('interpreter: %s' % ' '.join(sys.version.split()[:2]))
    images_before = {a: ((HERE / a).read_bytes() if (HERE / a).exists() else None)
                     for a in INFORMATIVE}
    failures = []
    for label, command, artefacts in STAGES:
        marked = len(failures)
        before = {a: ((HERE / a).read_bytes() if (HERE / a).exists() else None)
                  for a in (artefacts or [])}
        if run(command) != 0:
            failures.append('%s: stage exited non-zero' % label)
        elif artefacts is None:
            fresh = sorted(d for d in (HERE / 'benchmark-results').iterdir()
                           if d.is_dir() and d != FROZEN)
            if not fresh:
                failures.append('%s: report wrote no archive directory' % label)
            for directory in fresh:
                for name in REPORT_CSVS:
                    produced, frozen = directory / name, FROZEN / name
                    # `report` writes through csv on Windows, the published copies are
                    # LF-only; line endings are the only legitimate difference.
                    got = produced.read_bytes().replace(b'\r\n', b'\n') if produced.exists() else None
                    want = frozen.read_bytes().replace(b'\r\n', b'\n')
                    if got is None:
                        failures.append('%s: %s not produced' % (label, name))
                    elif got != want:
                        failures.append('%s: %s differs from the frozen copy' % (label, name))
                shutil.rmtree(directory)
        else:
            for artefact in artefacts:
                path = HERE / artefact
                if before[artefact] is None or not path.exists():
                    failures.append('%s: %s missing' % (label, artefact))
                elif before[artefact] != path.read_bytes():
                    failures.append('%s: %s changed on rerun' % (label, artefact))
        print('%-36s %s' % (label, 'FAIL' if len(failures) > marked else 'ok'))

    for artefact in INFORMATIVE:
        before = images_before[artefact]
        path = HERE / artefact
        if before is not None and path.exists() and before != path.read_bytes():
            print('note: %s bytes differ from the checked-in copy; the plotted values in '
                  'figures/figures_manifest.json matched, and image bytes depend on the '
                  'host matplotlib and fonts.' % artefact)

    if failures:
        print('\nFAIL:')
        for line in failures:
            print('  -', line)
        return 1
    print('\nAll stages reproduce the checked-in artefacts.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
