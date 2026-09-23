# -*- coding: utf-8 -*-
"""Release gate: assert the published tree carries no non-pseudonymised material.

Runs against the checkout as it stands, with no access to the private source
project, so every check is structural. It is a backstop for `desensitize.py`,
not a replacement for reading it.

Two scopes. The data under `work/` and `benchmark-results/` must be free of
pre-anonymisation identifiers. The whole tree, code and Rust reference included,
must be free of machine-local paths and of references to unrelated projects.

Usage:  python check_release.py [path/to/release]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
DATA_DIRS = ('work', 'benchmark-results')
KEY_RE = re.compile(r'(knowledge_node|question|private_chunk):([^"\s,\]\}]+)')
OPAQUE = re.compile(r'^(knowledge_node:kn|question:qs|private_chunk:pc)-\d{5}$')
# A mirror id left un-folded would still name the entity it mirrors.
DATA_TRACES = re.compile(r'(?i)(evalmirror|privdoc|privchk)')
TREE_TRACES = re.compile(r'(?i)(proxy_norm|teacher[_-]?agent|\b[a-z]:[\\/]|'
                         r'api[_-]?key\s*=|bearer\s+[a-z0-9]|/users/[a-z0-9._-]+/)')
BANNED = re.compile(r'(?i)^(pool|_extracted_data|_state)\.json$|\.sqlite3$|^\.env$|\.html$')
TEXT_SUFFIXES = {'.json', '.jsonl', '.py', '.md', '.csv', '.txt', '.rs', '.cfg', '.toml'}


def is_data(path):
    parts = path.relative_to(ROOT).parts
    return bool(parts) and parts[0] in DATA_DIRS


def main():
    failures = []
    scanned = 0
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name == Path(__file__).name:
            continue
        if BANNED.search(path.name):
            failures.append('%s: file name is on the banned list' % path.relative_to(ROOT))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        body = path.read_text(encoding='utf-8', errors='replace')
        rel = path.relative_to(ROOT).as_posix()
        data = is_data(path)
        rules = [(TREE_TRACES, 'machine-local path or unrelated-project reference')]
        if data:
            rules.append((DATA_TRACES, 'pre-anonymisation identifier'))
        for pattern, why in rules:
            for hit in sorted(set(m.group(0) for m in pattern.finditer(body))):
                failures.append('%s: %s %r' % (rel, why, hit))
        if data:
            for parts in sorted(set(KEY_RE.findall(body))):
                key = '%s:%s' % parts
                if not OPAQUE.match(key):
                    failures.append('%s: non-opaque entity key %r' % (rel, key))
            for number, line in enumerate(body.splitlines(), start=1):
                if path.suffix != '.jsonl' or not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    failures.append('%s:%d unparsable (%s)' % (rel, number, exc))
                    continue
                for field in ('query', 'target_term'):
                    if row.get(field) is not None:
                        failures.append('%s: leaks %s' % (rel, field))

    print('scanned %d text files under %s' % (scanned, ROOT))
    if failures:
        print('\nFAIL (%d):' % len(failures))
        for line in failures[:40]:
            print('  -', line)
        return 1
    print('PASS: entity ids opaque, no machine-local paths, no corpus payload file')
    return 0


if __name__ == '__main__':
    sys.exit(main())
