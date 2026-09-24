#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p1-1：同候选空间（common candidate set）公平对照。

为什么需要这个脚本
------------------
生产链路的两个通道候选空间**不相等**：

* 倒排通道 `bm25_top_k` 只查 `private_document_chunks_fts`，因此只能召回
  私有切片（评测副本里 159 条）；
* 稠密通道遍历 `DENSE_ENTITY_TYPES`，覆盖知识节点 773 + 题库 764 + 私有切片 159。

于是「hybrid 优于 BM25」这类比较同时混合了两件事：**融合排序是否更好** 与
**候选覆盖是否更大**。本脚本在评测副本上把两通道拉到同一个候选集合
（knowledge_node + question + private_chunk，共 1696 条），再比较
BM25 / Dense / RRF 三种策略，从而把「排序收益」与「覆盖收益」分开报告。

做法（不改生产 Rust）
----------------------
把每个「已审核且已有 bge-m3 向量」的知识节点与题目，按**与生产嵌入完全相同的
文本配方**（`seedEmbeddingService.ts`：节点 `"{title}: {summary}"`、
题目 `"{title}: {content}" + " 答案: {answer}"`，截断 8000 字符）镜像成
`private_document_chunks` 的一行，id 前缀 `evalmirror-kn-` / `evalmirror-q-`。
外部内容表 `private_document_chunks_fts` 的触发器会把镜像行写进倒排索引，
于是**生产 BM25 代码路径**（`retrieve_context(query_embedding=None)`）自然就在
统一语料上排名——Python 侧不重写 BM25、余弦或 RRF。

镜像只写 `work/eval_corpus_fair*.sqlite3`（守卫见 `guard_target`），线上库零写入。

复现
----
    python fair_candidate_space.py prepare      # 建 baseline 与 unified 两个评测副本
    python fair_candidate_space.py rank         # 调 Rust harness 出两副本的排名
    python fair_candidate_space.py report       # 指标 + 配对 Bootstrap + 归档
    python fair_candidate_space.py all          # 以上三步
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from numerics import csum

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
WORK = HERE / "work"
BENCH_ROOT = HERE / "benchmark-results"
TAURI_ROOT = HERE / "src-tauri"

SOURCE_DB = WORK / "eval_corpus.sqlite3"
BASELINE_DB = WORK / "eval_corpus_fairbase.sqlite3"
UNIFIED_DB = WORK / "eval_corpus_fairunified.sqlite3"

QA_PATH = WORK / "qa_100_human_v3.jsonl"
NO_GOLD = ["q073", "q074", "q087", "q092", "q095", "q099", "q100"]

# The live database `prepare` must never write into. Only the pseudonymised frozen
# assets under work/ are published, so on a public checkout this is a placeholder.
PRODUCTION_DB = os.environ.get("HYBRID_SOURCE_DB", str(WORK / "source_live.sqlite3"))

MODEL = "bge-m3"
TOP_K = 20          # 与既有 a20 归档一致：每通道取 Top-20
CUTOFF = 5          # 论文口径：指标取 Top-5
FORMS = ("template", "bare")
SEED = 20260916
N_BOOT = 2000

MIRROR_DOC_ID = "evalmirror-unified-source"
KN_PREFIX = "evalmirror-kn-"
Q_PREFIX = "evalmirror-q-"
MAX_TEXT_LENGTH = 8000


# ── 守卫 ───────────────────────────────────────────────────────────────
def guard_target(path: Path) -> Path:
    """只允许在评测副本上动手。"""
    target = path.resolve()
    if str(target) == str(Path(PRODUCTION_DB).resolve()):
        raise SystemExit(f"拒绝执行：目标是生产库 {target}")
    if not target.name.startswith("eval_corpus"):
        raise SystemExit(f"拒绝执行：文件名必须以 eval_corpus 开头（当前 {target.name}）")
    return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── prepare ────────────────────────────────────────────────────────────
def embedding_text(entity_type: str, row: dict) -> str:
    """生产嵌入的文本配方（src/services/knowledge/seedEmbeddingService.ts）。"""
    if entity_type == "knowledge_node":
        text = f"{row.get('title') or ''}: {row.get('summary') or ''}"
    else:
        text = f"{row.get('title') or ''}: {row.get('content') or ''}"
        answer = row.get("answer") or ""
        if answer:
            text += f" 答案: {answer}"
    return text[:MAX_TEXT_LENGTH]


def mirror_rows(connection: sqlite3.Connection):
    """只镜像「已有 bge-m3 向量」的实体——没有向量的实体稠密通道永远召不回，
    放进倒排侧会给 BM25 造成不公平的覆盖优势。"""
    vectors = {
        (entity_type, entity_id)
        for entity_type, entity_id in connection.execute(
            "SELECT entity_type, entity_id FROM vector_embeddings WHERE embedding_model = ?", (MODEL,)
        )
    }
    rows = []
    node_columns = {row[1] for row in connection.execute("PRAGMA table_info(knowledge_nodes)")}
    question_columns = {row[1] for row in connection.execute("PRAGMA table_info(questions)")}
    for entity_type, table in (("knowledge_node", "knowledge_nodes"), ("question", "questions")):
        columns = sorted(
            ({c for c in ("id", "title", "summary", "content", "answer")} & node_columns)
            if table == "knowledge_nodes"
            else ({c for c in ("id", "title", "summary", "content", "answer")} & question_columns)
        )
        select = ", ".join(columns)
        for record in connection.execute(f"SELECT {select} FROM {table} ORDER BY id"):
            row = dict(zip(columns, record))
            if (entity_type, row["id"]) not in vectors:
                continue
            prefix = KN_PREFIX if entity_type == "knowledge_node" else Q_PREFIX
            rows.append((f"{prefix}{row['id']}", embedding_text(entity_type, row)))
    return rows


def build_copy(source: Path, target: Path) -> None:
    guard_target(target)
    if target.exists():
        target.unlink()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, sqlite3.connect(target) as dst:
        src.backup(dst)


def command_prepare(args) -> None:
    if not SOURCE_DB.exists():
        raise SystemExit(f"评测副本不存在：{SOURCE_DB}\n  请先运行： python run_eval.py prepare")

    build_copy(SOURCE_DB, BASELINE_DB)
    print(f"baseline 副本： {BASELINE_DB}")

    build_copy(SOURCE_DB, UNIFIED_DB)
    connection = sqlite3.connect(UNIFIED_DB)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    rows = mirror_rows(connection)
    connection.execute(
        """INSERT INTO private_documents
               (id, student_id, subject_code, file_name, file_type, title,
                source_type, status, content_hash, chunk_count, created_at, updated_at, deleted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
        (MIRROR_DOC_ID, "eval-unified", "eval-unified", "eval-unified-mirror", "text/markdown",
         "p1-1 统一候选空间镜像（评测专用）", "eval_unified_mirror", "approved", None,
         len(rows), stamp, stamp),
    )
    connection.executemany(
        """INSERT INTO private_document_chunks (id, document_id, chunk_index, heading, text, token_estimate, created_at)
           VALUES (?, ?, ?, NULL, ?, NULL, ?)""",
        [(chunk_id, MIRROR_DOC_ID, index, text, stamp) for index, (chunk_id, text) in enumerate(rows)],
    )
    connection.execute("UPDATE private_documents SET chunk_count = ? WHERE id = ?", (len(rows), MIRROR_DOC_ID))
    connection.commit()

    counts = dict(connection.execute(
        "SELECT entity_type, COUNT(DISTINCT entity_id) FROM vector_embeddings"
        " WHERE embedding_model = ? GROUP BY entity_type", (MODEL,)).fetchall())
    chunks = connection.execute("SELECT COUNT(*) FROM private_document_chunks").fetchone()[0]
    connection.close()

    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "workItem": "p1-1 common-candidate-set retrieval comparison",
        "sourceCopy": str(SOURCE_DB),
        "sourceCopySha256": sha256_of(SOURCE_DB),
        "baselineCopy": {"path": str(BASELINE_DB), "sha256": sha256_of(BASELINE_DB),
                         "note": "与源副本逐字节一致，用于复现既有部署口径排名"},
        "unifiedCopy": {"path": str(UNIFIED_DB), "sha256": sha256_of(UNIFIED_DB),
                        "mirroredEntities": len(rows),
                        "mirrorChunksTotal": chunks,
                        "textRecipe": "knowledge_node \"{title}: {summary}\"；question \"{title}: {content}\"+ \" 答案: {answer}\"；截断 8000 字符",
                        "scope": "EVAL-ONLY；镜像行只写评测副本，生产库零写入"},
        "embeddableEntities": counts,
        "candidateSpace": {
            "unifiedSize": sum(counts.values()),
            "vectors": counts,
            "excluded": "5 个 draft 状态知识节点无 bge-m3 向量，稠密通道不可召回，故不镜像",
        },
    }
    (WORK / "fair_candidate_space_prepare.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"unified 副本： {UNIFIED_DB}（镜像 {len(rows)} 条实体，切片总数 {chunks}）")


# ── rank ───────────────────────────────────────────────────────────────
def run_harness(db: Path, tag: str, form: str, profile: str) -> Path:
    vectors = WORK / f"query_vectors_{form}.jsonl"
    if not vectors.exists():
        raise SystemExit(f"缺少查询向量：{vectors}")
    out_path = WORK / f"fair_rank_{tag}_{form}.jsonl"
    environment = dict(os.environ)
    environment.update({
        "HYBRID_EVAL_DB": str(db.resolve()),
        "HYBRID_EVAL_IN": str(vectors.resolve()),
        "HYBRID_EVAL_OUT": str(out_path.resolve()),
        "HYBRID_EVAL_TOP": str(TOP_K),
        "HYBRID_EVAL_MODEL": MODEL,
        "HYBRID_EVAL_REPS": "1",
    })
    command = ["cargo", "test", "--lib", "hybrid_retrieval_eval", "--", "--ignored", "--nocapture"]
    if profile == "release":
        command.insert(2, "--release")
    print(f"  rank[{tag}/{form}]:", " ".join(command))
    started = time.perf_counter()
    subprocess.run(command, cwd=TAURI_ROOT, env=environment, check=True)
    print(f"  rank[{tag}/{form}]: {time.perf_counter() - started:.1f}s -> {out_path.name}")
    return out_path


def command_rank(args) -> None:
    for db, tag in ((BASELINE_DB, "base"), (UNIFIED_DB, "unified")):
        if not db.exists():
            raise SystemExit(f"缺少副本 {db}：请先运行 prepare")
        for form in FORMS:
            run_harness(db, tag, form, args.profile)


# ── 指标 ───────────────────────────────────────────────────────────────
def remap(key: str) -> str:
    """把镜像切片 id 还原为原始实体键。"""
    source, _, entity_id = key.partition(":")
    if source != "private_chunk":
        return key
    if entity_id.startswith(KN_PREFIX):
        return f"knowledge_node:{entity_id[len(KN_PREFIX):]}"
    if entity_id.startswith(Q_PREFIX):
        return f"question:{entity_id[len(Q_PREFIX):]}"
    return key


def load_rankings(tag: str, form: str, remap_keys: bool = True):
    path = WORK / f"fair_rank_{tag}_{form}.jsonl"
    if not path.exists():
        raise SystemExit(f"缺少排名文件 {path}：请先运行 rank")
    out = {}
    for row in read_jsonl(path):
        hits = [hit["key"] for hit in sorted(row["hits"], key=lambda item: item["rank"])]
        out.setdefault(row["qid"], {})[row["config"]] = [remap(key) if remap_keys else key for key in hits]
    return out


def space_by_prefix(key: str) -> int:
    """生产 `source_space`：以 `private_chunk:` 前缀区分来源空间。"""
    return 0 if key.startswith("private_chunk:") else 1


def space_by_channel(key: str, bm25_set: set, dense_set: set) -> int:
    """统一语料下的来源空间：按**哪个通道召回**区分（0=倒排，1=稠密）。

    生产实现用 id 前缀代替通道，因为在部署配置里两通道候选空间不相交
    （倒排只出 private_chunk、稠密只出 knowledge_node/question），前缀与通道等价。
    统一语料后该等价不再成立，故这里改为按通道归属判定；其余排序语义与
    `fuse_rrf` 逐字一致。
    """
    return 0 if key in bm25_set else 1


def fuse_rrf(bm25: list[str], dense: list[str], k: int = 60, space_of=space_by_prefix):
    """`fuse_rrf`（src-tauri/src/rag/hybrid.rs）的逐语义移植：标准 RRF + 生产并列破局键。"""
    bm25_rank = {key: index for index, key in enumerate(bm25, start=1)}
    dense_rank = {key: index for index, key in enumerate(dense, start=1)}
    bm25_set, dense_set = {*bm25}, {*dense}
    entries = []
    for key in dict.fromkeys([*bm25, *dense]):
        left, right = bm25_rank.get(key), dense_rank.get(key)
        score = (1.0 / (k + left) if left else 0.0) + (1.0 / (k + right) if right else 0.0)
        dual = left is not None and right is not None
        rank = min(value for value in (left, right) if value is not None)
        space = space_of(key, bm25_set, dense_set) if space_of is space_by_channel else space_of(key)
        # 奇数名次倒排优先、偶数名次稠密优先
        offset = 0 if (space == 0) == (rank % 2 == 1) else 1
        entries.append({
            "key": key,
            "score": score,
            "space": space,
            "tie": (0 if dual else 1, (rank - 1) * 2 + offset, space, key),
        })
    entries.sort(key=lambda entry: (-entry["score"], entry["tie"]))
    return entries


def apply_source_quota(entries: list[dict], top_k: int, floor: int) -> list[dict]:
    """`apply_source_quota` 的逐语义移植：两侧各保底 floor 个槽位，其余按融合顺序补足。"""
    if len(entries) <= top_k:
        return entries
    if {entry["space"] for entry in entries} != {0, 1}:
        return entries[:top_k]
    selected = [False] * len(entries)
    chosen, taken = 0, {0: 0, 1: 0}
    for space in (0, 1):
        for index, entry in enumerate(entries):
            if taken[space] >= floor or chosen >= top_k:
                break
            if not selected[index] and entry["space"] == space:
                selected[index] = True
                taken[space] += 1
                chosen += 1
    for index in range(len(entries)):
        if chosen >= top_k:
            break
        if not selected[index]:
            selected[index] = True
            chosen += 1
    return [entry for entry, keep in zip(entries, selected) if keep]


def fuse_keys(bm25: list[str], dense: list[str], k: int = 60, space_of=space_by_prefix, top_k=None):
    entries = fuse_rrf(bm25, dense, k=k, space_of=space_of)
    keys = [entry["key"] for entry in entries]
    return keys[:top_k] if top_k else keys


def interleave(*rankings):
    seen, out = set(), []
    for position in range(max((len(ranking) for ranking in rankings), default=0)):
        for ranking in rankings:
            if position < len(ranking) and ranking[position] not in seen:
                seen.add(ranking[position])
                out.append(ranking[position])
    return out


def oracle_best(first: list[str], second: list[str], gold: set[str]):
    """两通道事后最优的 oracle 参照：按命中位置取更好的那一路。"""
    def rank_of(ranking):
        for position, key in enumerate(ranking[:CUTOFF], start=1):
            if key in gold:
                return position
        return None
    left, right = rank_of(first), rank_of(second)
    candidates = [value for value in (left, right) if value is not None]
    return min(candidates) if candidates else None


def metrics(ranking: list[str], gold: set[str]) -> dict:
    if not gold:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "rank": None}
    rank = None
    for position, key in enumerate(ranking[:CUTOFF], start=1):
        if key in gold:
            rank = position
            break
    dcg = csum(1.0 / math.log2(position + 1) for position, key in enumerate(ranking[:CUTOFF], start=1) if key in gold)
    ideal = csum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold), CUTOFF) + 1))
    return {
        "hit@1": 1.0 if rank == 1 else 0.0,
        "hit@3": 1.0 if rank is not None and rank <= 3 else 0.0,
        "hit@5": 1.0 if rank is not None else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg@5": dcg / ideal if ideal else 0.0,
        "rank": rank,
    }


PER_SOURCE_FLOOR = 2


def unified_strategies(bm25: list[str], dense: list[str]) -> dict:
    return {
        "bm25_unified": bm25,
        "dense_unified": dense,
        "rrf_k60_unified": fuse_keys(bm25, dense, k=60, space_of=space_by_channel),
        "rrf_k1_unified": fuse_keys(bm25, dense, k=1, space_of=space_by_channel),
        "rrf_k60_unified_prefixkey": fuse_keys(bm25, dense, k=60, space_of=space_by_prefix),
        "interleave_unified": interleave(bm25, dense),
    }


ORACLE_KEY = "oracle_best_single_channel"


def aggregate(rows) -> dict:
    return {key: csum(row[key] for row in rows) / len(rows)
            for key in ("hit@1", "hit@3", "hit@5", "mrr", "ndcg@5")}


def cluster_bootstrap(left: list, right: list, n_boot: int = N_BOOT):
    """问题级配对 Bootstrap（与 v3/v4 口径一致）：重采样单位是 qid。"""
    diffs = defaultdict(lambda: defaultdict(list))
    for a, b in zip(left, right):
        for metric in ("ndcg@5", "mrr"):
            diffs[metric][a["qid"]].append(a[metric] - b[metric])
    qids = sorted(diffs["ndcg@5"])
    rng = random.Random(SEED + 17)
    sampled = {"ndcg@5": [], "mrr": []}
    for _ in range(n_boot):
        drawn = [qids[rng.randrange(len(qids))] for _ in qids]
        for metric in ("ndcg@5", "mrr"):
            pool = diffs[metric]
            total = csum(csum(pool[q]) for q in drawn)
            count = sum(len(pool[q]) for q in drawn)
            sampled[metric].append(total / count)
    out = {}
    for metric, values in sampled.items():
        values.sort()
        observed = csum(csum(diffs[metric][q]) for q in qids) / sum(len(diffs[metric][q]) for q in qids)
        out[metric] = {
            "delta": observed,
            "ci": [values[int(0.025 * n_boot)], values[int(0.975 * n_boot) - 1]],
        }
    return out


# ── validate ───────────────────────────────────────────────────────────
def command_validate(args) -> None:
    """证明融合端口与 Rust 生产实现逐位一致。

    在 baseline 副本上，用 `space_by_prefix`（= 生产 `source_space`）复算
    `apply_source_quota(fuse_rrf(bm25, dense))`，与 harness `hybrid` 配置的输出
    逐条比对。全部相等才说明移植正确；统一语料上的 `space_by_channel` 只改这一处。
    """
    checked = matched = 0
    mismatches = []
    for form in FORMS:
        base = load_rankings("base", form, remap_keys=False)
        for qid, configs in base.items():
            expected = configs["hybrid"]
            fused = fuse_rrf(configs["bm25"], configs["dense"], k=60, space_of=space_by_prefix)
            actual = [entry["key"] for entry in apply_source_quota(fused, TOP_K, PER_SOURCE_FLOOR)]
            checked += 1
            if actual == expected:
                matched += 1
            else:
                mismatches.append({"queryForm": form, "qid": qid,
                                   "expected": expected[:5], "actual": actual[:5]})
    print(f"融合端口校验：{matched}/{checked} 条与 Rust harness `hybrid` 输出逐位一致")
    if mismatches:
        print(json.dumps(mismatches[:3], ensure_ascii=False, indent=2))
        raise SystemExit("融合端口与生产实现不一致，评测结果不可用")

    record = {
        "checked": checked,
        "matched": matched,
        "target": "baseline 副本 `hybrid` 配置输出（= retrieve_context 的 RRF + 来源配额）",
        "ported": ["fuse_rrf（标准 RRF + 奇偶交替破局键）", "apply_source_quota（每源 floor=2）"],
        "differenceInUnifiedRun": "仅 source_space 由「id 前缀」改为「召回通道」，见 space_by_channel",
    }
    (WORK / "fair_candidate_space_validation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


# ── report ─────────────────────────────────────────────────────────────
def command_report(args) -> None:
    questions = {row["qid"]: row for row in read_jsonl(QA_PATH)}
    covered = sorted(qid for qid in questions if qid not in NO_GOLD)

    per_query = []
    rankings = {}
    for tag in ("base", "unified"):
        for form in FORMS:
            rankings[(tag, form)] = load_rankings(tag, form)

    for form in FORMS:
        unified = rankings[("unified", form)]
        base = rankings[("base", form)]
        for qid in covered:
            gold = set(questions[qid]["gold_ids"])
            bm25 = unified[qid]["bm25"]
            dense = unified[qid]["dense"]
            base_bm25 = base[qid]["bm25"]
            base_hybrid = base[qid]["hybrid"]

            rows = unified_strategies(bm25, dense)
            rows["bm25_prod_chunkonly"] = base_bm25
            rows["hybrid_prod_quota"] = base_hybrid
            rows = {strategy: metrics(ranking, gold) for strategy, ranking in rows.items()}
            rank = oracle_best(bm25, dense, gold)
            rows[ORACLE_KEY] = {
                "hit@1": 1.0 if rank == 1 else 0.0,
                "hit@3": 1.0 if rank is not None and rank <= 3 else 0.0,
                "hit@5": 1.0 if rank is not None else 0.0,
                "mrr": 1.0 / rank if rank else 0.0,
                "ndcg@5": 1.0 if rank is not None else 0.0,
                "rank": rank,
            }
            for strategy, value in rows.items():
                per_query.append({
                    "scope": "conditional_93",
                    "queryForm": form,
                    "qid": qid,
                    "category": questions[qid]["category"],
                    "target_source": questions[qid]["target_source"],
                    "strategy": strategy,
                    **value,
                })
            # overall_100：7 个无相关候选的问题计 0
            for strategy, value in rows.items():
                per_query.append({
                    "scope": "overall_100",
                    "queryForm": form,
                    "qid": qid,
                    "category": questions[qid]["category"],
                    "target_source": questions[qid]["target_source"],
                    "strategy": strategy,
                    **value,
                })
    for qid in NO_GOLD:
        for form in FORMS:
            for strategy in [*unified_strategies([], []), "bm25_prod_chunkonly", "hybrid_prod_quota", ORACLE_KEY]:
                per_query.append({
                    "scope": "overall_100", "queryForm": form, "qid": qid,
                    "category": questions[qid]["category"], "target_source": questions[qid]["target_source"],
                    "strategy": strategy,
                    "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "rank": None,
                })

    comparisons = [
        ("rrf_k60_unified", "bm25_unified"),
        ("rrf_k60_unified", "dense_unified"),
        ("rrf_k60_unified", ORACLE_KEY),
        ("bm25_unified", "bm25_prod_chunkonly"),
        ("dense_unified", "bm25_prod_chunkonly"),
    ]

    report = {
        "protocol": {
            "question": "BM25 / Dense / RRF 在同一候选集合（1696 条实体的统一语料）上的排序质量对照",
            "candidateSpace": "knowledge_node 773 + question 764 + private_chunk 159 = 1696",
            "rankingSource": "全部排名来自 Rust 生产实现（src-tauri/src/rag/eval.rs::hybrid_retrieval_eval）",
            "topK": TOP_K, "cutoff": CUTOFF, "kRrf": 60,
            "bootstrap": {"unit": "问题（qid）", "n": N_BOOT, "pairing": "同 qid 同日形态配对"},
            "scopes": {"conditional_93": "有 gold 的 93 题 × 2 形态",
                       "overall_100": "含 7 个无相关候选问题的完整口径（计 0）"},
        },
        "results": defaultdict(dict), "comparisons": {},
    }

    summary_rows = []
    for scope in ("conditional_93", "overall_100"):
        for form in [*FORMS, "both"]:
            rows = [row for row in per_query
                    if row["scope"] == scope and (form == "both" or row["queryForm"] == form)]
            for strategy in sorted({row["strategy"] for row in rows}):
                values = aggregate([row for row in rows if row["strategy"] == strategy])
                report["results"][f"{scope}/{form}"][strategy] = values
                summary_rows.append({"scope": scope, "queryForm": form, "strategy": strategy, **values})

    by_strategy = defaultdict(list)
    for row in per_query:
        if row["scope"] == "conditional_93":
            by_strategy[(row["queryForm"], row["strategy"])].append(row)
    for left_name, right_name in comparisons:
        entry = {}
        for form in FORMS:
            left = by_strategy[(form, left_name)]
            right = by_strategy[(form, right_name)]
            merged = {row["qid"]: row for row in left}
            paired_left = [merged[row["qid"]] for row in right]
            entry[form] = cluster_bootstrap(paired_left, right)
        report["comparisons"][f"{left_name} - {right_name}"] = entry

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = BENCH_ROOT / f"{timestamp}-rag-fair-candidate-space"
    out_dir.mkdir(parents=True, exist_ok=False)

    def write_csv(name, fields, rows):
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("results_overall.csv",
              ["scope", "queryForm", "strategy", "hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"],
              summary_rows)
    write_csv("per_query_metrics.csv",
              ["scope", "queryForm", "qid", "category", "target_source", "strategy",
               "hit@1", "hit@3", "hit@5", "mrr", "ndcg@5", "rank"],
              per_query)
    write_csv("paired_bootstrap.csv",
              ["comparison", "queryForm", "metric", "delta", "ciLow", "ciHigh", "bootstrapRuns"],
              [{"comparison": name, "queryForm": form, "metric": metric,
                "delta": values[metric]["delta"],
                "ciLow": values[metric]["ci"][0], "ciHigh": values[metric]["ci"][1],
                "bootstrapRuns": N_BOOT}
               for name, forms in report["comparisons"].items()
               for form, values in forms.items()
               for metric in ("ndcg@5", "mrr")])

    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "workItem": "p1-1 common-candidate-set fair comparison",
        "qa": str(QA_PATH), "qaSha256": sha256_of(QA_PATH),
        "groundTruth": {
            "annotators": "2 human + 3 LLM, equal-weight majority vote, third-human adjudication on conflicts",
            "limitation": "不是纯人工 gold standard；7 道无相关候选的题目在 overall_100 中计 0",
        },
        "copies": {
            # The two corpus mirrors are built by `prepare` from the private
            # knowledge base and are not part of the published release, so a
            # public checkout legitimately has nothing to hash here.
            name: {"path": str(path),
                   "sha256": sha256_of(path) if path.exists() else None,
                   "status": "present" if path.exists() else "corpus mirror not published"}
            for name, path in (("baseline", BASELINE_DB), ("unified", UNIFIED_DB))
        },
        "rankings": {f"{tag}_{form}": f"work/fair_rank_{tag}_{form}.jsonl"
                     for tag in ("base", "unified") for form in FORMS},
        "seed": SEED, "bootstrapRuns": N_BOOT,
        "notes": [
            "两种形态（template/bare）× 93 题；7 道无相关候选题不进条件口径。",
            "统一候选集让两通道竞争同一批文档，BM25 的 IDF/avgdl 也随语料从 159 变为 1696。",
            "hybrid_prod_quota 与 bm25_prod_chunkonly 是**部署口径**参照，不是同候选集结果。",
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fair_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== 同候选集（1696）条件口径 93 题 =====")
    for scope in ("conditional_93", "overall_100"):
        print(f"\n-- {scope}")
        for form in [*FORMS, "both"]:
            table = report["results"][f"{scope}/{form}"]
            print(f"  [{form}] 策略{'':<16}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}{'NDCG@5':>8}")
            for strategy, values in sorted(table.items(), key=lambda kv: -kv[1]["ndcg@5"]):
                print(f"    {strategy:<24}{values['hit@1']:>8.4f}{values['hit@3']:>8.4f}"
                      f"{values['hit@5']:>8.4f}{values['mrr']:>8.4f}{values['ndcg@5']:>8.4f}")
    print("\n===== 配对 Bootstrap（问题级，2000 次，93 题条件口径） =====")
    for name, forms in report["comparisons"].items():
        for form, values in forms.items():
            print(f"  {name:<40} [{form}] ΔNDCG={values['ndcg@5']['delta']:+.4f} "
                  f"CI=[{values['ndcg@5']['ci'][0]:+.4f},{values['ndcg@5']['ci'][1]:+.4f}] | "
                  f"ΔMRR={values['mrr']['delta']:+.4f} "
                  f"CI=[{values['mrr']['ci'][0]:+.4f},{values['mrr']['ci'][1]:+.4f}]")
    print("\n归档:", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="p1-1 同候选空间公平对照")
    parser.add_argument("--profile", choices=["release", "debug"], default="release")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("rank")
    sub.add_parser("validate")
    sub.add_parser("report")
    sub.add_parser("all")
    args = parser.parse_args()

    if args.command in ("prepare", "all"):
        command_prepare(args)
    if args.command in ("rank", "all"):
        command_rank(args)
    if args.command in ("validate", "all"):
        command_validate(args)
    if args.command in ("report", "all"):
        command_report(args)


if __name__ == "__main__":
    sys.exit(main())