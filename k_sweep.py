"""RRF smoothing-constant sweep on the frozen unified-space rankings.

Reviewer comment 1 asked for intermediate k values. The two channel lists are
frozen in `work/fair_rank_unified_*.jsonl`, so k can be varied without touching
the corpus, the model, or the index: the fusion is re-run with the same
production tie-break that `fair_candidate_space.fuse_rrf` ports.

Gate first, numbers second: the script recomputes the published k=60 and k=1
cells from `results_overall.csv` and refuses to print anything new unless every
control cell matches to the published four decimals.
"""
import csv
import json
from pathlib import Path

from fair_candidate_space import (NO_GOLD, QA_PATH, aggregate, cluster_bootstrap,
                                  FORMS, load_rankings, metrics, read_jsonl,
                                  space_by_channel, fuse_keys)

KS = [1, 10, 20, 40, 60]
RESULTS = Path("benchmark-results/20260916-221246-rag-fair-candidate-space")
PUBLISHED = {"rrf_k60_unified": 60, "rrf_k1_unified": 1}


def published_cells():
    out = {}
    with (RESULTS / "results_overall.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["strategy"] in PUBLISHED and row["scope"] == "conditional_93":
                out[(row["strategy"], row["queryForm"])] = float(row["ndcg@5"])
    return out


def main():
    questions = {row["qid"]: row for row in read_jsonl(QA_PATH)}
    covered = sorted(q for q in questions if q not in NO_GOLD)
    rankings = {form: load_rankings("unified", form) for form in FORMS}

    # per (form, qid): gold + the two channel lists
    cells = {}
    for form in FORMS:
        for qid in covered:
            gold = set(questions[qid]["gold_ids"])
            cells[form] = cells.get(form, [])
            cells[form].append((qid, gold, rankings[form][qid]["bm25"],
                                rankings[form][qid]["dense"]))

    # ---- gate: reproduce the published k=60 / k=1 cells -------------------
    print("gate: recomputed vs published NDCG@5 (conditional_93)")
    pub = published_cells()
    failures = []
    for strategy, k in PUBLISHED.items():
        for form in (*FORMS, "both"):
            rows = []
            for f in ([form] if form != "both" else FORMS):
                for qid, gold, bm25, dense in cells[f]:
                    ranking = fuse_keys(bm25, dense, k=k, space_of=space_by_channel)
                    rows.append(metrics(ranking, gold))
            got = aggregate(rows)["ndcg@5"]
            want = pub[(strategy, form)]
            ok = abs(got - want) < 5e-5
            print("  %-18s %-8s %.4f vs %.4f  %s" % (strategy, form, got, want,
                                                     "OK" if ok else "MISMATCH"))
            if not ok:
                failures.append((strategy, form, got, want))
    if failures:
        raise SystemExit("gate failed on %d cells -- refusing to report new k"
                         % len(failures))

    # ---- sweep ------------------------------------------------------------
    table = {}
    per_query = {}
    for k in KS:
        for form in (*FORMS, "both"):
            rows = []
            for f in ([form] if form != "both" else FORMS):
                for qid, gold, bm25, dense in cells[f]:
                    ranking = fuse_keys(bm25, dense, k=k, space_of=space_by_channel)
                    m = metrics(ranking, gold)
                    m["qid"] = qid
                    rows.append(m)
            table[(k, form)] = aggregate(rows)
            per_query[(k, form)] = rows

    print("\nRRF smoothing-constant sweep, conditional_93 (186 query instances "
          "when form=both)")
    print("  k     form      NDCG@5   MRR@5    Hit@5")
    for k in KS:
        for form in (*FORMS, "both"):
            a = table[(k, form)]
            print("  %-5d %-9s %.4f   %.4f   %.4f" % (
                k, form, a["ndcg@5"], a["mrr"], a["hit@5"]))

    # ---- RRF vs Dense at each k, same question-clustered bootstrap --------
    print("\nDelta NDCG@5 (RRF at k minus Dense), question-clustered percentile "
          "bootstrap, 2000 draws")
    dense_rows = {}
    for form in (*FORMS, "both"):
        rows = []
        for f in ([form] if form != "both" else FORMS):
            for qid, gold, bm25, dense in cells[f]:
                m = metrics(dense, gold)
                m["qid"] = qid
                rows.append(m)
        dense_rows[form] = rows
    out = {}
    for k in KS:
        line = {"k": k}
        for form in (*FORMS, "both"):
            bs = cluster_bootstrap(per_query[(k, form)], dense_rows[form])
            line[form] = {"delta": round(bs["ndcg@5"]["delta"], 4),
                          "ci": [round(v, 4) for v in bs["ndcg@5"]["ci"]]}
            print("  k=%-3d %-9s %+0.4f [%+0.4f, %+0.4f]" % (
                k, form, line[form]["delta"], line[form]["ci"][0],
                line[form]["ci"][1]))
        out["k=%d" % k] = line

    dest = Path("results/k_sweep.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({
        "scope": "conditional_93",
        "note": "frozen unified-space channel lists re-fused at each k with the "
                "production tie-break; controls for k=60 and k=1 reproduced from "
                "results_overall.csv before any new value was computed",
        "levels_ndcg5": {"%d/%s" % (k, f): round(table[(k, f)]["ndcg@5"], 4)
                         for k in KS for f in (*FORMS, "both")},
        "levels_mrr5": {"%d/%s" % (k, f): round(table[(k, f)]["mrr"], 4)
                        for k in KS for f in (*FORMS, "both")},
        "levels_hit5": {"%d/%s" % (k, f): round(table[(k, f)]["hit@5"], 4)
                        for k in KS for f in (*FORMS, "both")},
        "delta_vs_dense": out,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
