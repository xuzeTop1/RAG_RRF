//! 工作项 E：混合检索质量评测的批处理入口（评测专用，不在生产路径中）。
//!
//! 通过环境变量驱动的 `#[ignore]` 测试暴露给 Python CLI，避免在 Python 里
//! 重写 BM25 / 余弦 / RRF —— 三配置全部走生产实现：
//!   * `bm25`   —— `retrieve_context(..., query_embedding = None)`（倒排通道退化态）
//!   * `dense`  —— `vector::search_similar` 对 [`DENSE_ENTITY_TYPES`] 逐类召回后按分数归并
//!   * `hybrid` —— `retrieve_context(..., query_embedding = Some(vec))`（RRF 融合）
//!
//! 环境变量：
//!   * `HYBRID_EVAL_DB`    待评测数据库路径（评测会在该副本上补齐迁移）
//!   * `HYBRID_EVAL_IN`    JSONL 输入，每行 `{"qid":..,"query":..,"vector":[f32]|null}`
//!   * `HYBRID_EVAL_OUT`   输出 JSONL 路径
//!   * `HYBRID_EVAL_TOP`   每配置返回命中数（默认 5）
//!   * `HYBRID_EVAL_MODEL` 向量模型名（默认 `bge-m3`）
//!   * `HYBRID_EVAL_REPS`  每查询每配置重复次数（时延统计，默认 1）
//!
//! 输出 JSONL 每行：`{"qid","config","hits":[{"key","score","rank"}],"elapsedMicros":[...]}`

use std::time::Instant;

use rusqlite::Connection;
use serde_json::{json, Value};

use crate::rag::hybrid::{retrieve_context, DENSE_ENTITY_TYPES};

const CONFIGS: [&str; 3] = ["bm25", "dense", "hybrid"];

fn environment(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|_| format!("missing environment variable {name}"))
}

/// 稠密通道独立召回：与 `retrieve_context` 内部同一套调用（同模型、同 Top-K），
/// 两个实体类型各自取 Top-K 后按余弦分数归并，再截断到 `top_k`。
fn dense_only(connection: &Connection, embedding: &[f32], model: &str, top_k: usize) -> Result<Vec<(String, f64)>, String> {
    let mut scored: Vec<(f64, String)> = Vec::new();
    for entity_type in DENSE_ENTITY_TYPES {
        let hits = crate::vector::search_similar(
            connection,
            entity_type,
            embedding,
            model,
            top_k as i64,
            None,
        )?;
        for hit in hits {
            scored.push((hit.score, format!("{entity_type}:{}", hit.entity_id)));
        }
    }
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    scored.truncate(top_k);
    Ok(scored.into_iter().map(|(score, key)| (key, score)).collect())
}

/// 一条评测命中：除余弦分外还带出 RRF 分与两通道名次，便于在 Python 侧直接断言
/// 融合行为（例如并列时的来源交替、来源配额是否生效）。
struct RankedHit {
    key: String,
    dense_score: Option<f64>,
    rrf_score: Option<f64>,
    bm25_rank: Option<usize>,
    dense_rank: Option<usize>,
}

impl RankedHit {
    fn to_json(&self, rank: usize) -> Value {
        json!({
            "key": self.key,
            "score": self.dense_score,
            "rrfScore": self.rrf_score,
            "bm25Rank": self.bm25_rank,
            "denseRank": self.dense_rank,
            "rank": rank,
        })
    }
}

fn ranked_for(
    connection: &Connection,
    config: &str,
    query: &str,
    embedding: Option<&[f32]>,
    model: &str,
    top_k: usize,
) -> Result<Vec<RankedHit>, String> {
    match config {
        // 纯倒排：query_embedding 为 None 时 retrieve_context 只走 BM25 通道。
        "bm25" => Ok(retrieve_context(connection, query, model, None, None, top_k)?
            .into_iter()
            .map(hybrid_hit_to_ranked)
            .collect()),
        "dense" => Ok(dense_only(
            connection,
            embedding.ok_or_else(|| "dense config requires a query vector".to_string())?,
            model,
            top_k,
        )?
        .into_iter()
        .map(|(key, score)| RankedHit {
            key,
            dense_score: Some(score),
            rrf_score: None,
            bm25_rank: None,
            dense_rank: None,
        })
        .collect()),
        "hybrid" => Ok(retrieve_context(connection, query, model, embedding, None, top_k)?
            .into_iter()
            .map(hybrid_hit_to_ranked)
            .collect()),
        other => Err(format!("unknown config {other}")),
    }
}

fn hybrid_hit_to_ranked(hit: crate::rag::hybrid::HybridHit) -> RankedHit {
    RankedHit {
        key: format!("{}:{}", hit.source, hit.id),
        dense_score: hit.dense_score,
        rrf_score: Some(hit.rrf_score),
        bm25_rank: hit.bm25_rank,
        dense_rank: hit.dense_rank,
    }
}

#[test]
#[ignore = "benchmark harness: driven by scripts, reads HYBRID_EVAL_* env vars"]
fn hybrid_retrieval_eval() {
    let database_path = environment("HYBRID_EVAL_DB").expect("HYBRID_EVAL_DB");
    let input_path = environment("HYBRID_EVAL_IN").expect("HYBRID_EVAL_IN");
    let output_path = environment("HYBRID_EVAL_OUT").expect("HYBRID_EVAL_OUT");
    let top_k: usize = std::env::var("HYBRID_EVAL_TOP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let model = std::env::var("HYBRID_EVAL_MODEL").unwrap_or_else(|_| "bge-m3".to_string());
    let reps: usize = std::env::var("HYBRID_EVAL_REPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
        .max(1);

    let connection = Connection::open(&database_path).expect("open eval database");
    // 评测副本上补齐迁移（线上库可能仍是旧版本，FTS5 表尚未建立）。
    crate::database::apply_migrations(&connection).expect("apply migrations on eval copy");

    let raw = std::fs::read_to_string(&input_path).expect("read eval input");
    let mut lines: Vec<String> = Vec::new();
    let mut queries = 0usize;

    for (index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line).expect("parse eval input line");
        let qid = row["qid"].as_str().expect("qid").to_string();
        let query = row["query"].as_str().expect("query").to_string();
        let embedding: Option<Vec<f32>> = match row.get("vector") {
            Some(Value::Array(values)) => Some(
                values
                    .iter()
                    .map(|value| value.as_f64().expect("vector element") as f32)
                    .collect(),
            ),
            _ => None,
        };
        queries += 1;

        for config in CONFIGS {
            if config == "dense" && embedding.is_none() {
                continue;
            }
            let mut hits: Vec<RankedHit> = Vec::new();
            let mut elapsed: Vec<u128> = Vec::with_capacity(reps);
            for rep in 0..reps {
                let started = Instant::now();
                let ranked = ranked_for(&connection, config, &query, embedding.as_deref(), &model, top_k)
                    .unwrap_or_else(|error| panic!("{qid}/{config}: {error}"));
                let micros = started.elapsed().as_micros();
                if rep == 0 {
                    hits = ranked;
                }
                elapsed.push(micros);
            }
            let payload = json!({
                "qid": qid,
                "config": config,
                "hits": hits
                    .iter()
                    .enumerate()
                    .map(|(position, hit)| hit.to_json(position + 1))
                    .collect::<Vec<Value>>(),
                "elapsedMicros": elapsed,
            });
            lines.push(payload.to_string());
        }

        if (index + 1) % 20 == 0 {
            eprintln!("hybrid eval: processed {}/{} queries", index + 1, raw.lines().count());
        }
    }

    std::fs::write(&output_path, lines.join("\n") + "\n").expect("write eval output");
    eprintln!(
        "hybrid eval: {queries} queries × {reps} reps × {} configs → {output_path}",
        CONFIGS.len()
    );
}
