"""金标替换敏感性实验（回应批注 15/17/18/19 与评审 M4）。

复用既有冻结资产，不重新检索、不调用任何模型：
  排名  work/fair_rank_unified_{template,bare}.jsonl（Rust harness 产出，Top-20）
  投票  work/annotation/annotations_评审{一,二,三,四,五,六}.json
  标签  work/qa_100_human_v3.jsonl
指标与 Bootstrap 全部走 fair_candidate_space 的原函数（同一 seed、同一 qid 聚类口径）。
"""
import ast
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("gas", HERE / "gold_annotation_sensitivity.py")
gas = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(gas)

fcs = gas.fair_candidate_space if hasattr(gas, "fair_candidate_space") else None
import fair_candidate_space as fcs  # noqa: E402

HUMAN, MODEL, EXCL = gas.HUMAN, gas.MODEL, gas.EXCLUDED
FULL6 = (*HUMAN, *MODEL, EXCL)


def decide(annotations, qid, key, rule):
    human = gas.votes_for(annotations, qid, key, HUMAN)
    model = gas.votes_for(annotations, qid, key, MODEL)
    five = gas.votes_for(annotations, qid, key, gas.GOLD_JUDGES)
    six = gas.votes_for(annotations, qid, key, FULL6)
    if rule == "majority5":
        return int(sum(five) >= gas.THRESHOLD)
    if rule == "human_unanimous_pos":
        return int(human[0] == 1 and human[1] == 1)
    if rule == "human_any_pos":
        return int(human[0] == 1 or human[1] == 1)
    if rule == "model_only":
        return gas.majority(model)
    if rule == "majority6_ge3":
        return int(sum(six) >= 3)
    if rule == "majority6_strict4":
        return int(sum(six) >= 4)
    raise ValueError(rule)


def gold_from(annotations, qids, rule):
    out = {}
    for qid in qids:
        keys = set()
        for judge in FULL6:
            keys |= set(annotations[judge].get(qid, {}))
        out[qid] = {k for k in keys if decide(annotations, qid, k, rule)}
    return out


def as_set(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return set(value)


def evaluate(rankings, gold, qids):
    rows = []
    for qid in qids:
        cfg = rankings.get(qid)
        if not cfg:
            continue
        g = gold[qid]
        bm25, dense = cfg["bm25"], cfg["dense"]
        rrf = fcs.fuse_keys(bm25, dense, k=60, space_of=fcs.space_by_channel)
        m = {"bm25": fcs.metrics(bm25, g), "dense": fcs.metrics(dense, g), "rrf_k60": fcs.metrics(rrf, g)}
        rows.append({"qid": qid, **{k: m[k] for k in m}})
    return rows


def summarize(rows):
    agg = {}
    for strat in ("bm25", "dense", "rrf_k60"):
        agg[strat] = {
            "ndcg@5": statistics.fmean(r[strat]["ndcg@5"] for r in rows),
            "mrr": statistics.fmean(r[strat]["mrr"] for r in rows),
        }
    return agg


def main():
    annotations = gas.load_annotations()
    qids = sorted({q for j in FULL6 for q in annotations[j]})
    covered_base = [q for q in qids if q not in gas.NO_GOLD]
    rules = ("majority5", "human_unanimous_pos", "human_any_pos", "model_only",
             "majority6_ge3", "majority6_strict4")
    golds = {r: gold_from(annotations, qids, r) for r in rules}
    v3 = {row["qid"]: as_set(row["gold_ids"]) for row in fcs.read_jsonl(fcs.QA_PATH)}
    golds["v3_frozen"] = v3

    report = {"protocol": {
        "rankings": "fair_rank_unified_{template,bare}.jsonl (Top-20, Rust harness)",
        "voting": "847 池化查询—实体对；人类=评审一/二；模型=评审三/五/六；被剔除=评审四",
        "bootstrap": "qid 聚类配对，2000 次，seed 同 fair_candidate_space（SEED+17）",
        "conditionalSet": "各金标规则下'池内相关集合非空'的题集（口径随规则变化，故同时给出固定 93 题结果）"},
        "rules": {}}

    lines = ["== 金标定义与规模 =="]
    for name in (*rules, "v3_frozen"):
        sizes = [len(golds[name].get(q, ())) for q in covered_base]
        report["rules"][name] = {"questionsWithGold": sum(1 for s in sizes if s > 0),
                                "goldItems": sum(sizes)}
        lines.append("  %-20s 有金标题数=%3d  金标条目=%4d" % (name, report["rules"][name]["questionsWithGold"], sum(sizes)))

    for name in (*rules, "v3_frozen"):
        base_delta = sum(1 for q in covered_base if golds[name].get(q) != golds["majority5"].get(q))
        report["rules"][name]["diffVsMajority5"] = base_delta
        lines.append("  %-20s 与 majority5 不同的题数=%d" % (name, base_delta))

    lines.append("")
    lines.append("== 统一空间指标（both = 两形态等权）")
    for name in (*rules, "v3_frozen"):
        per_form = {}
        for form in fcs.FORMS:
            rankings = fcs.load_rankings("unified", form)
            qset = [q for q in covered_base if golds[name].get(q)]
            rows = evaluate(rankings, golds[name], qset)
            agg = summarize(rows)
            rr = [{"qid": r["qid"], "ndcg@5": r["rrf_k60"]["ndcg@5"], "mrr": r["rrf_k60"]["mrr"]} for r in rows]
            ci = {
                "rrf_vs_dense": fcs.cluster_bootstrap(rr, [{"qid": r["qid"], "ndcg@5": r["dense"]["ndcg@5"], "mrr": r["dense"]["mrr"]} for r in rows]),
                "rrf_vs_bm25": fcs.cluster_bootstrap(rr, [{"qid": r["qid"], "ndcg@5": r["bm25"]["ndcg@5"], "mrr": r["bm25"]["mrr"]} for r in rows]),
            }
            per_form[form] = {"n": len(qset), "agg": agg, "ci": ci}
        entry = report["rules"][name]["unified"] = per_form
        both_nd = statistics.fmean([entry[f]["agg"]["rrf_k60"]["ndcg@5"] for f in fcs.FORMS])
        den_nd = statistics.fmean([entry[f]["agg"]["dense"]["ndcg@5"] for f in fcs.FORMS])
        bm_nd = statistics.fmean([entry[f]["agg"]["bm25"]["ndcg@5"] for f in fcs.FORMS])
        flips = [f for f in fcs.FORMS
                 if entry[f]["ci"]["rrf_vs_dense"]["ndcg@5"]["ci"][0] <= 0
                 or entry[f]["ci"]["rrf_vs_bm25"]["ndcg@5"]["ci"][0] <= 0]
        lines.append("  %-20s n(每形态)=%3d/%3d  RRF=%.4f Dense=%.4f BM25=%.4f  Δ(RRF-Dense)=%+.4f Δ(RRF-BM25)=%+.4f  CI下界≤0的形态=%s"
                     % (name, entry["template"]["n"], entry["bare"]["n"], both_nd, den_nd, bm_nd,
                        both_nd - den_nd, both_nd - bm_nd, flips or "无"))

    out_json = HERE / "results" / "gold_sensitivity_ext.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_json.parent / "gold_sensitivity_ext.txt").write_text("\n".join(lines), encoding="utf-8")
    print("OK json+txt written; rules=%d" % len(report["rules"]))


if __name__ == "__main__":
    main()
