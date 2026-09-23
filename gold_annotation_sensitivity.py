#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p1-4：RAG gold 标注的构成核验与 gold 口径敏感性分析。

背景
----
论文的检索 gold 由 **2 名真人评审 + 3 个 AI 评审等权多数决**产生，因此不是纯人工金标准。
本脚本把这件事从「口头声明」变成可核算的数字，并检查结论是否依赖 AI 票：

1. **判断者构成**：`work/annotation/` 下实际有 6 份标注（评审一~六）。归档的 kappa 报告
   （`multi_kappa_report.json`）只用了 5 份（一、二、三、五、六）——即**评审四被剔除**。
   本脚本复算「评审四与各保留评审的一致率」，核对剔除理由。
2. **人机分歧**：人类一致票（评审一、二都给 1 或都给 0）与 5 人多数票、3 个模型多数票的
   冲突计数——即论文所称「人类一致票与 LLM 多数票冲突仅 3/847」。
3. **gold 口径敏感性**：在 4 种 gold 变体下重算检索指标，看 p1-1 的结论
   （覆盖收益 vs RRF 排序收益）是否随 gold 口径改变。

用法
----
    python gold_annotation_sensitivity.py            # 全部三步
    python gold_annotation_sensitivity.py --report   # 只出核验与分歧统计
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
ANNOTATION = WORK / "annotation"
BENCH_ROOT = HERE / "benchmark-results"

sys.path.insert(0, str(HERE))
from fair_candidate_space import (  # noqa: E402
    FORMS, aggregate, fuse_keys, load_rankings, metrics, read_jsonl, space_by_channel,
)

HUMAN = ("评审一", "评审二")
MODEL = ("评审三", "评审五", "评审六")
EXCLUDED = "评审四"
GOLD_JUDGES = (*HUMAN, *MODEL)          # 与 multi_kappa_report.json 的 annotators 一致
THRESHOLD = 3                            # 5 人多数票阈值（>=3 记为相关）
NO_GOLD = ["q073", "q074", "q087", "q092", "q095", "q099", "q100"]
VARIANTS = ("majority5", "human_first", "human_unanimous_pos", "model_only")


def load_annotations() -> dict:
    """读取全部标注文件（兼容两种命名：`annotations_X.json` 与 `X_annotations.json`）。"""
    table = {}
    for path in sorted(ANNOTATION.glob("*.json")):
        if "annotation" not in path.name.lower() or path.name.startswith("_"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        annotator = data.get("annotator")
        if not annotator or not isinstance(data.get("judgments"), dict):
            continue
        table[annotator] = data["judgments"]
    return table


def votes_for(annotations: dict, qid: str, key: str, judges) -> list:
    return [int(annotations[judge].get(qid, {}).get(key, {}).get("v", 0)) for judge in judges]


def majority(values) -> int:
    return 1 if sum(values) * 2 > len(values) else 0


def gold_from(annotations: dict, qids, judges, rule) -> dict:
    """按给定规则产出 {qid: set(gold keys)}。"""
    out = {}
    for qid in qids:
        keys = set()
        for judge in judges:
            keys |= set(annotations[judge].get(qid, {}))
        gold = set()
        for key in keys:
            human = votes_for(annotations, qid, key, HUMAN)
            model = votes_for(annotations, qid, key, MODEL)
            five = votes_for(annotations, qid, key, GOLD_JUDGES)
            if rule == "majority5":
                decision = 1 if sum(five) >= THRESHOLD else 0
            elif rule == "model_only":
                decision = majority(model)
            elif rule == "human_first":
                # 两名真人一致即定；分歧时回落到 3 个模型的多数票（无第三名真人可用）
                decision = human[0] if human[0] == human[1] else majority(model)
            elif rule == "human_unanimous_pos":
                decision = 1 if human[0] == 1 and human[1] == 1 else 0
            else:
                raise ValueError(rule)
            if decision:
                gold.add(key)
        out[qid] = gold
    return out


def raw_agreement(left: dict, right: dict, qids) -> tuple:
    same = total = 0
    for qid in qids:
        keys = set(left.get(qid, {})) | set(right.get(qid, {}))
        for key in keys:
            a = int(left.get(qid, {}).get(key, {}).get("v", 0))
            b = int(right.get(qid, {}).get(key, {}).get("v", 0))
            same += a == b
            total += 1
    return same, total


def main() -> None:
    parser = argparse.ArgumentParser(description="p1-4 标注构成核验与 gold 口径敏感性")
    parser.add_argument("--report", action="store_true", help="只出核验与分歧统计")
    args = parser.parse_args()

    annotations = load_annotations()
    print("磁盘上的标注文件：")
    for name, judgments in annotations.items():
        print(f"  {name:<6} qids={len(judgments):>4} judgments={sum(len(v) for v in judgments.values()):>4}")
    report = json.loads((ANNOTATION / "multi_kappa_report.json").read_text(encoding="utf-8"))
    print(f"\nkappa 报告采用的判断者：{report['annotators']}（n_items={report['n_items']}，"
          f"Fleiss κ={report['fleiss_kappa']:.4f}）")
    print(f"→ 磁盘上多出一份 {EXCLUDED}，未被纳入 gold 计算")
    missing = [judge for judge in GOLD_JUDGES if judge not in annotations]
    if missing:
        raise SystemExit(f"缺少 gold 判断者标注：{missing}")

    qids = sorted({qid for judge in GOLD_JUDGES for qid in annotations[judge]})
    print(f"\n题目数 {len(qids)}，候选判定总数 {sum(len(annotations[j].get(q, {})) for j in GOLD_JUDGES for q in qids)}")

    # ─ 1. 评审四 的剔除理由核验 ──────────────────────────────────────
    print(f"\n== 1. {EXCLUDED}（被剔除）与各保留评审的原始一致率 ==")
    exclusion = {}
    for judge in GOLD_JUDGES:
        same, total = raw_agreement(annotations[EXCLUDED], annotations[judge], qids)
        rate = same / total if total else float("nan")
        exclusion[judge] = {"rawAgreement": rate, "items": total}
        print(f"  {EXCLUDED} vs {judge}: {rate:.4f}（{same}/{total}）")

    # ── 2. 人机分歧计数 ───────────────────────────────────────────────
    print("\n== 2. 人类一致票 vs 多数票的冲突计数 ==")
    counts = Counter()
    conflicts = {"vs_majority5": [], "vs_model_only": []}
    for qid in qids:
        keys = set()
        for judge in GOLD_JUDGES:
            keys |= set(annotations[judge].get(qid, {}))
        for key in keys:
            human = votes_for(annotations, qid, key, HUMAN)
            model = votes_for(annotations, qid, key, MODEL)
            five = votes_for(annotations, qid, key, GOLD_JUDGES)
            counts["items"] += 1
            if human[0] != human[1]:
                counts["human_split"] += 1
                continue
            counts["human_unanimous"] += 1
            human_verdict = human[0]
            if human_verdict != (1 if sum(five) >= THRESHOLD else 0):
                counts["conflict_with_majority5"] += 1
                conflicts["vs_majority5"].append((qid, key, human, five))
            if human_verdict != majority(model):
                counts["conflict_with_model_only"] += 1
                conflicts["vs_model_only"].append((qid, key, human, model))
    for name, value in counts.items():
        print(f"  {name:<26} {value}")
    if counts["human_unanimous"]:
        print(f"  → 人类一致票与 5 人多数票冲突率："
              f"{counts['conflict_with_majority5']}/{counts['human_unanimous']} "
              f"= {counts['conflict_with_majority5'] / counts['human_unanimous']:.4f}")
        print(f"  → 人类一致票与模型多数票冲突率："
              f"{counts['conflict_with_model_only']}/{counts['human_unanimous']} "
              f"= {counts['conflict_with_model_only'] / counts['human_unanimous']:.4f}")
    print("  冲突项明细（最多 5 条）：")
    for qid, key, human, five in conflicts["vs_majority5"][:5]:
        print(f"    {qid} {key}  真人={human} 五人={five}")

    # 与既有 majority_votes.json 交叉核对
    stored = json.loads((ANNOTATION / "majority_votes.json").read_text(encoding="utf-8"))
    mismatch = 0
    for qid, entry in stored.items():
        for key, value in entry.items():
            mine = votes_for(annotations, qid, key, GOLD_JUDGES)
            if value.get("votes") and list(value["votes"]) != mine:
                mismatch += 1
    print(f"  与 majority_votes.json 的 votes 数组逐项核对：不一致 {mismatch} 项")

    if args.report:
        return

    # ── 3. gold 口径敏感性 ────────────────────────────────────────────
    covered = [qid for qid in qids if qid not in NO_GOLD]
    golds = {rule: gold_from(annotations, qids, GOLD_JUDGES, rule) for rule in VARIANTS}
    golds["v3_frozen"] = {
        row["qid"]: set(row["gold_ids"]) for row in read_jsonl(WORK / "qa_100_human_v3.jsonl")
    }

    print("\n== 3. gold 变体规模与差异 ==")
    table = {}
    for name, gold in golds.items():
        sizes = [len(gold.get(qid, ())) for qid in covered]
        table[name] = {"questionsWithGold": sum(1 for s in sizes if s > 0),
                       "goldItems": sum(sizes), "emptyQuestions": sum(1 for s in sizes if s == 0)}
        print(f"  {name:<22} 有 gold 题数={table[name]['questionsWithGold']:>3} "
              f"gold 条目={table[name]['goldItems']:>4} 空 gold={table[name]['emptyQuestions']:>2}")
    for name in VARIANTS:
        if name == "majority5":
            continue
        changed = sum(1 for qid in covered if golds[name][qid] != golds["majority5"][qid])
        added = sum(len(golds[name][qid] - golds["majority5"][qid]) for qid in covered)
        removed = sum(len(golds["majority5"][qid] - golds[name][qid]) for qid in covered)
        print(f"  vs majority5：{name:<22} 题级不同 {changed:>3} 题，新增 {added:>3} 条，移除 {removed:>3} 条")

    # 检索指标：用 p1-1 已缓存的排名（同候选集与部署口径各一套）
    metrics_table = {}
    for tag, label in (("base", "部署口径（BM25 只覆盖私有切片）"),
                       ("unified", "同候选集（1696 条）")):
        for form in FORMS:
            rankings = load_rankings(tag, form)
            for variant, gold in golds.items():
                rows = {}
                for qid in covered:
                    cfg = rankings.get(qid)
                    if not cfg:
                        continue
                    bm25, dense = cfg["bm25"], cfg["dense"]
                    strategies = {
                        "bm25": metrics(bm25, gold[qid]),
                        "dense": metrics(dense, gold[qid]),
                        "rrf_k60": metrics(fuse_keys(bm25, dense, k=60, space_of=space_by_channel), gold[qid]),
                    }
                    if tag == "base":
                        strategies["prod_hybrid"] = metrics(cfg["hybrid"], gold[qid])
                    for strategy, value in strategies.items():
                        rows.setdefault(strategy, []).append(value)
                metrics_table[f"{label}|{form}|{variant}"] = {
                    strategy: aggregate(values) for strategy, values in rows.items()
                }

    print("\n== 3b. 各 gold 变体下的 NDCG@5（条件 93 题） ==")
    header = f"  {'gold 变体':<22}{'口径':<10}{'形态':<10}{'bm25':>9}{'dense':>9}{'rrf_k60':>9}{'prod':>9}"
    print(header)
    for variant in (*VARIANTS, "v3_frozen"):
        for tag, label in (("base", "部署"), ("unified", "同候选集")):
            for form in FORMS:
                entry = metrics_table[f"{'部署口径（BM25 只覆盖私有切片）' if tag == 'base' else '同候选集（1696 条）'}|{form}|{variant}"]
                prod = entry.get("prod_hybrid", {}).get("ndcg@5")
                print(f"  {variant:<22}{label:<10}{form:<10}"
                      f"{entry['bm25']['ndcg@5']:>9.4f}{entry['dense']['ndcg@5']:>9.4f}"
                      f"{entry['rrf_k60']['ndcg@5']:>9.4f}"
                      f"{(f'{prod:.4f}' if prod is not None else '-'):>9}")

    # 结论稳定性：同候选集上 rrf 是否仍优于两个单通道
    stable = 0
    total = 0
    for variant in (*VARIANTS, "v3_frozen"):
        for form in FORMS:
            entry = metrics_table[f"同候选集（1696 条）|{form}|{variant}"]
            total += 1
            if (entry["rrf_k60"]["ndcg@5"] > entry["bm25"]["ndcg@5"]
                    and entry["rrf_k60"]["ndcg@5"] > entry["dense"]["ndcg@5"]):
                stable += 1
    print(f"\n  同候选集上「RRF 的 NDCG@5 同时高于 BM25 与 Dense」在 {stable}/{total} 个"
          f"（gold 变体 × 形态）组合中成立")

    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = BENCH_ROOT / f"{timestamp}-gold-annotation-sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "annotationFiles": {name: len(judgments) for name, judgments in annotations.items()},
        "goldJudges": list(GOLD_JUDGES),
        "humanJudges": list(HUMAN),
        "modelJudges": list(MODEL),
        "excludedJudge": EXCLUDED,
        "kappaReport": {"annotators": report["annotators"], "items": report["n_items"],
                        "fleissKappa": report["fleiss_kappa"]},
        "exclusionCheck": exclusion,
        "humanVsMajority": dict(counts),
        "conflictsVsMajority5": [{"qid": q, "key": k, "humanVotes": h, "fiveVotes": f}
                                 for q, k, h, f in conflicts["vs_majority5"]],
        "conflictsVsModelOnly": [{"qid": q, "key": k, "humanVotes": h, "modelVotes": m}
                                 for q, k, h, m in conflicts["vs_model_only"]],
        "goldVariants": table,
        "metrics": metrics_table,
        "rrfBeatsBothChannelsOnCommonSet": {"satisfied": stable, "total": total},
    }
    (out_dir / "gold_annotation_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n归档： {out_dir}")


if __name__ == "__main__":
    sys.exit(main())