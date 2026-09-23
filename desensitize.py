# -*- coding: utf-8 -*-
"""Rewrite the frozen evaluation assets into the pseudonymised release layout.

Entity keys in the source project carry corpus-derived slugs (dictionary head
words, private-document fingerprints), so every key becomes an opaque id while
ranks, scores, votes and labels are left untouched.

Two properties the replay depends on, both preserved here:
  * the `private_chunk:` / `knowledge_node:` / `question:` prefixes, which
    `source_space()` and the fusion tie-break branch on;
  * lexicographic key order, the last component of the tie-break, so ids are
    numbered in sorted order of the original key within each prefix.

The reverse mapping is deliberately not written anywhere.

Usage:  python desensitize.py --source-root <checkout with work/> [--dry-run]
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX_GROUP = {'knowledge_node': 'kn', 'question': 'qs', 'private_chunk': 'pc'}
REDACTED = '[redacted: corpus-derived text]'
KEY_RE = re.compile(r'(?P<prefix>%s):(?P<slug>[^"\s,\]\}]+)' % '|'.join(PREFIX_GROUP))

# `fair_candidate_space.py` mirrors every entity into the private-chunk space so
# the BM25 index covers it, and `remap()` folds those mirror ids back afterwards.
# The fold has to happen before pseudonymisation: mirrored and original ids denote
# one entity, and rewriting them separately would give an entity two ids and break
# `remap()` for good.
MIRROR_PREFIX = {'private_chunk:evalmirror-kn-': 'knowledge_node:',
                 'private_chunk:evalmirror-q-': 'question:'}


def unmirror(text):
    for mirror, real in MIRROR_PREFIX.items():
        text = text.replace('"%s' % mirror, '"%s' % real)
    return text


def rel(source_root, *parts):
    return source_root.joinpath(*parts)


def input_files(source_root):
    """(source path, release path) for every frozen asset that is published.

    `source_root` is the retrieval project checkout: the frozen assets live under
    its `experiments/hybrid_retrieval/work/` and `benchmark-results/` directories.
    """
    w = rel(source_root, 'experiments', 'hybrid_retrieval', 'work')
    a = w / 'annotation'
    b = rel(source_root, 'benchmark-results', '20260916-221246-rag-fair-candidate-space')
    files = [(w / 'qa_100_human_v3.jsonl', 'work/qa_100_human_v3.jsonl')]
    for name in ('hybrid_raw_a20_template.jsonl', 'hybrid_raw_a20_bare.jsonl'):
        files.append((w / name, 'work/' + name))
    for tag in ('base', 'unified'):
        for form in ('template', 'bare'):
            # frozen Rust-harness rankings; `report` replays these, never the corpus
            files.append((w / ('fair_rank_%s_%s.jsonl' % (tag, form)),
                          'work/fair_rank_%s_%s.jsonl' % (tag, form)))
    for name in ('multi_kappa_report.json', 'majority_votes.json'):
        files.append((a / name, 'work/annotation/' + name))
    # per-rater exports use two namings (`annotations_X.json` and `X_annotations.json`);
    # gold_annotation_sensitivity.py keys them by the `annotator` field, not the filename.
    # Anything carrying candidate passages is blocked outright: `pool.json` and
    # `_extracted_data.json` embed the corpus text the annotation pages were built from.
    blocked = {'pool.json', '_extracted_data.json', '_state.json', 'kappa_report.json',
               'gap_analysis.json', 'disagreements_a1_a3.json',
               'multi_kappa_report.json', 'majority_votes.json'}
    digits = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6'}
    for path in sorted(a.glob('*.json')):
        if path.name in blocked or path.name.endswith('.bak'):
            continue
        # ASCII entry names: CJK zip members break on some tools, and the two
        # source namings collapse to one. The in-file `annotator` label stays
        # 评审一..六, which is what gold_annotation_sensitivity.py keys on.
        cjk = [c for c in path.name if c in digits]
        if len(cjk) != 1:
            raise SystemExit('cannot name rater file unambiguously: %s' % path.name)
        files.append((path, 'work/annotation/annotations_rater%s.json' % digits[cjk[0]]))
    for name in ('results_overall.csv', 'paired_bootstrap.csv', 'per_query_metrics.csv'):
        files.append((b / name, 'benchmark-results/20260916-221246-rag-fair-candidate-space/' + name))
    return [(s, t) for s, t in files if s.exists()], [s for s, _ in files if not s.exists()]


def collect_keys(files):
    keys = set()
    for src, _ in files:
        if src.suffix == '.jsonl' or src.name.endswith('.json'):
            body = unmirror(src.read_text(encoding='utf-8'))
            keys.update(m.group(0) for m in KEY_RE.finditer(body))
    return keys


def build_mapping(keys):
    by_group = {}
    for key in keys:
        prefix = key.split(':', 1)[0]
        by_group.setdefault(prefix, []).append(key)
    mapping, unknown = {}, []
    for prefix, group in by_group.items():
        tag = PREFIX_GROUP.get(prefix)
        if tag is None:
            unknown.append(prefix)
            continue
        for index, key in enumerate(sorted(group), start=1):
            mapping[key] = '%s:%s-%05d' % (prefix, tag, index)
    if unknown:
        raise SystemExit('unexpected key prefixes, extend PREFIX_GROUP: %s' % unknown)
    return mapping


def rewrite(text, mapping):
    # longest-first plus a "not followed by a key character" guard, so
    # `question:...-q8` is never rewritten inside `...-q87`.
    for key in sorted(mapping, key=len, reverse=True):
        text = re.sub(re.escape(key) + r'(?![A-Za-z0-9_\-])', mapping[key], text)
    return text


def redact_free_text(target):
    """Drop the fields whose wording is corpus-derived, keep the audit structure."""
    if target.name != 'qa_100_human_v3.jsonl' and target.name != 'multi_kappa_report.json':
        return
    if target.suffix == '.jsonl':
        rows = [json.loads(line) for line in target.read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in rows:
            for field in ('query', 'target_term'):
                if field in row:
                    row[field] = None
            for field in ('ai_review_correction', 'ai_review_supplement'):
                value = row.get(field)
                # both shapes occur: a single dict for one correction, a list for several
                for entry in (value if isinstance(value, list) else [value] if value else []):
                    if isinstance(entry, dict) and 'note' in entry:
                        entry['note'] = REDACTED
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                          encoding='utf-8', newline='\n')
        return
    report = json.loads(target.read_text(encoding='utf-8'))
    for entry in report.get('changed_vs_machine') or []:
        if 'query' in entry:
            entry['query'] = None
    target.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--source-root', required=True,
                    help='checkout holding work/ and benchmark-results/ (local dev only)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    files, missing = input_files(Path(args.source_root))
    mapping = build_mapping(collect_keys(files))
    print('entity keys pseudonymised: %d' % len(mapping))
    for prefix in sorted(PREFIX_GROUP):
        n = sum(1 for k in mapping if k.startswith(prefix + ':'))
        print('  %-16s %d' % (prefix + ':', n))
    if args.dry_run:
        return 0

    for src, arc in files:
        target = HERE / arc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewrite(unmirror(src.read_text(encoding='utf-8')), mapping),
                          encoding='utf-8', newline='\n')
        redact_free_text(target)
        print('wrote %-58s %8d bytes' % (arc, target.stat().st_size))
    if missing:
        print('\nmissing source assets (%d):' % len(missing))
        for path in missing:
            print('  ', path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
