//! 论文 §4.2：私有文档混合检索（BM25 + 暴力余弦 + RRF 融合）。
//!
//! 设计约束：
//!   * 倒排通道复用 SQLite 同源编译的 FTS5（外部内容表，见 migration 0016），
//!     不引入任何外部检索库；FTS5 内建 `bm25()` 的 k1/b 固定为 1.2/0.75，
//!     与论文 §4.2 声明的经验默认值一致（无需也不能在调用处覆盖）。
//!   * 分词器为 trigram：支持中文子串匹配，但要求查询串 ≥ 3 个字符；
//!     更短的查询由调用方退化为纯稠密通道（见 `bm25_top_k` 的短路分支）。
//!   * RRF 只依赖名次，天然规避 BM25 与余弦的量纲差异；单通道为空或故障时
//!     融合结果自动退化为另一通道的排序。

use rusqlite::{Connection, OptionalExtension};

/// RRF 平滑常数 k（论文 §4.2 取 60）。
pub(crate) const DEFAULT_K_RRF: u32 = 60;
/// 每个通道的候选集大小 K（论文 §4.2 取 20）。
pub(crate) const DEFAULT_CHANNEL_TOP_K: usize = 20;
/// Top-K 内每个来源空间的保底槽位数（抗饿死，见 [`apply_source_quota`]）。
const PER_SOURCE_FLOOR: usize = 2;
/// trigram 分词器要求的最小查询长度。
const TRIGRAM_MIN_CHARS: usize = 3;

/// 中文自然提问的疑问前缀。无空白的整串查询在短语匹配前先剥离这些外壳，
/// 否则 "X是什么意思" 会因讲义正文不存在该整串而召回为零。
const QUESTION_PREFIXES: [&str; 13] = [
    "请问什么是", "请问谁是", "请问如何", "请问怎么", "请问为什么", "请问",
    "什么是", "什么叫", "为什么", "如何", "怎么", "求问", "想了解",
];
/// 中文自然提问的疑问后缀（按长度降序匹配，先长后短）。
const QUESTION_SUFFIXES: [&str; 22] = [
    "是什么意思", "是什么含义", "是指什么", "指的是什么", "是什么",
    "的定义是什么", "的含义是什么", "的计算方法", "怎么计算", "如何计算",
    "怎么推导", "如何推导", "怎么算", "怎么求", "怎么做", "怎么样",
    "的定义", "的含义", "的概念",
    "呢", "吗", "呀",
];
/// 问句外壳剥离时一并去掉的首尾标点与空白。
const QUESTION_SHELL_PUNCTUATION: [char; 15] = [
    '?', '？', '!', '！', '.', '。', '~', '～', '，', ',', ';', '；', ':', '：', ' ',
];
/// 3-gram 兜底表达式的窗口上限：长叙述句若不做限幅，MATCH 表达式会随长度线性膨胀。
const MAX_FALLBACK_GRAMS: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScoredChunk {
    pub chunk_id: String,
    /// RRF 融合分数 Σ 1/(k + rank_m)
    pub rrf_score: f64,
    /// 在倒排通道中的名次（1-based）；未入榜为 None
    pub bm25_rank: Option<usize>,
    /// 在稠密通道中的名次（1-based）；未入榜为 None
    pub dense_rank: Option<usize>,
}

/// 执行一次 FTS5 BM25 查询，返回按相关度升序（FTS5 的 bm25() 越小越相关）排列的 chunk id。
fn execute_bm25_match(
    connection: &Connection,
    match_expression: &str,
    subject_code: Option<&str>,
    top_k: usize,
) -> Result<Vec<String>, String> {
    let top_k = top_k.clamp(1, 200) as i64;
    let mut statement = connection
        .prepare(
            "SELECT c.id
             FROM private_document_chunks_fts f
             JOIN private_document_chunks c ON c.rowid = f.rowid
             JOIN private_documents d ON d.id = c.document_id
             WHERE private_document_chunks_fts MATCH ?1
               AND (?2 IS NULL OR d.subject_code = ?2)
               AND d.deleted_at IS NULL
             ORDER BY bm25(private_document_chunks_fts, 1.0, 1.0)
             LIMIT ?3",
        )
        .map_err(|error| format!("failed to prepare bm25 search: {error}"))?;

    let rows = statement
        .query_map(rusqlite::params![match_expression, subject_code, top_k], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("failed to query bm25 search: {error}"))?;

    let mut ids = Vec::new();
    for row in rows {
        ids.push(row.map_err(|error| format!("failed to read bm25 row: {error}"))?);
    }
    Ok(ids)
}

/// 倒排通道：三阶段召回，返回按 BM25 升序（越相关分数越小）排列的 chunk id。
///
/// `subject_code` 为 None 时不做学科过滤。三阶段的设计动因（论文 §4.2 修订口径）：
///
/// 1. **含空白分词**：沿用 OR 分词表达式（原有行为，精度与召回都最好）。
/// 2. **剥离问句外壳后的整串短语**：无空白的自然问句（"偏振光是什么意思"）在
///    trigram 下若整串当短语，等价于要求正文逐字包含该问句——必然为空。剥离
///    "是什么意思/怎么算/如何" 等外壳后按核心词作短语匹配，保持高精度。
/// 3. **3-gram OR 兜底**：短语仍无命中时（长叙述句、未覆盖的问法），把核心文本切成
///    滑动 3-gram 做 OR，让 BM25 按重合度排序。只要正文出现核心词即可召回。
///
/// 第 2、3 阶段只会在"原表达式命中为空"时生效，因此对当前可正常召回的查询结果不变。
pub(crate) fn bm25_top_k(
    connection: &Connection,
    query: &str,
    subject_code: Option<&str>,
    top_k: usize,
) -> Result<Vec<String>, String> {
    let stripped = strip_question_shell(query);
    let core = stripped.trim();

    // 阶段 1：含空白分词的查询沿用 OR 分词表达式。
    if let Some(expression) = build_match_expression(query) {
        let hits = execute_bm25_match(connection, &expression, subject_code, top_k)?;
        if !hits.is_empty() {
            return Ok(hits);
        }
    }

    // 阶段 2：剥离外壳后的核心词（要求无空白，整串作短语）。
    if core.chars().count() >= TRIGRAM_MIN_CHARS && !core.contains(char::is_whitespace) {
        let expression = format!("\"{}\"", core.replace('"', "\"\""));
        let hits = execute_bm25_match(connection, &expression, subject_code, top_k)?;
        if !hits.is_empty() {
            return Ok(hits);
        }
    }

    // 阶段 3：3-gram OR 兜底。核心词不足 3 字时退回原串（可能被外壳剥离削得过短）。
    let gram_source = if core.chars().count() >= TRIGRAM_MIN_CHARS { core } else { query };
    if let Some(expression) = build_trigram_or_expression(gram_source) {
        return execute_bm25_match(connection, &expression, subject_code, top_k);
    }
    Ok(Vec::new())
}

/// 把用户查询转成安全的 FTS5 MATCH 表达式。
///
/// 全部 token 加引号（内部引号双写）以避免 FTS5 语法错误；多 token 以 OR 连接。
/// 任何候选 token 长度不足 trigram 下限时，整条查询退回为原串短语（若同样过短则返回 None）。
fn build_match_expression(query: &str) -> Option<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }
    let quoted = |value: &str| format!("\"{}\"", value.replace('"', "\"\""));

    let tokens: Vec<&str> = trimmed
        .split_whitespace()
        .filter(|token| token.chars().count() >= TRIGRAM_MIN_CHARS)
        .collect();
    if !tokens.is_empty() {
        return Some(tokens.iter().map(|token| quoted(token)).collect::<Vec<_>>().join(" OR "));
    }
    // 无空白分词的整串（典型中文短查询）：整体作为短语。
    if trimmed.chars().count() >= TRIGRAM_MIN_CHARS {
        return Some(quoted(trimmed));
    }
    None
}

/// 去掉首尾的问句标点与空白。
fn trim_shell_punctuation(value: &str) -> &str {
    value.trim_matches(|character: char| {
        QUESTION_SHELL_PUNCTUATION.contains(&character) || character.is_whitespace()
    })
}

/// 剥离中文问句外壳（疑问前后缀与首尾标点），返回核心查询词。
///
/// 反复剥离直到不再变化：先剥前缀再剥后缀，因此 "请问什么是极限的定义" 会先去掉
/// "请问什么是" 再去掉 "的定义"。剥离过程严格缩短字符串，必然终止。
fn strip_question_shell(query: &str) -> String {
    let mut current = trim_shell_punctuation(query.trim());
    loop {
        let before = current;
        for prefix in QUESTION_PREFIXES {
            if let Some(rest) = current.strip_prefix(prefix) {
                current = trim_shell_punctuation(rest);
                break;
            }
        }
        for suffix in QUESTION_SUFFIXES {
            if let Some(rest) = current.strip_suffix(suffix) {
                current = trim_shell_punctuation(rest);
                break;
            }
        }
        if current == before {
            return current.to_string();
        }
    }
}

/// 把核心文本切成滑动 3-gram 并 OR 连接，作为短语无命中时的召回兜底。
///
/// trigram 分词器索引的就是 3 字序列，所以单个 3-gram 恰是一个可精确匹配的词元；
/// OR 之后由 BM25 按命中数量与重合度排序，正文出现核心词即可召回。表达式长度按
/// [`MAX_FALLBACK_GRAMS`] 限幅。
fn build_trigram_or_expression(text: &str) -> Option<String> {
    let characters: Vec<char> = text
        .chars()
        .filter(|character| {
            !character.is_whitespace() && !QUESTION_SHELL_PUNCTUATION.contains(character)
        })
        .collect();
    if characters.len() < TRIGRAM_MIN_CHARS {
        return None;
    }
    let mut expressions: Vec<String> = Vec::new();
    for window in characters.windows(TRIGRAM_MIN_CHARS) {
        let gram: String = window.iter().collect();
        let quoted = format!("\"{}\"", gram.replace('"', "\"\""));
        if !expressions.contains(&quoted) {
            expressions.push(quoted);
        }
        if expressions.len() >= MAX_FALLBACK_GRAMS {
            break;
        }
    }
    if expressions.is_empty() {
        None
    } else {
        Some(expressions.join(" OR "))
    }
}

/// RRF 融合：按 Σ 1/(k + rank) 排序（论文 §4.2 式 (2)，Cormack 等 2009 的标准定义）。
///
/// 未入榜通道**记 0**，不再加 `K + 1` 的平坦兜底：兜底会给所有"单通道命中"的文档加同一个
/// 常数，使跨通道同名词次（倒排第 1 与稠密第 1）得分完全相等，只能靠 id 字典序破并列——
/// 而 `knowledge_node:`/`question:` 在 ASCII 中恒小于 `private_chunk:`，私有片段因此被
/// 系统性压低。去掉兜底后，双通道同时命中的文档自然得分翻倍并排到单通道文档之前。
///
/// 并列破局改用**按名次奇偶交替**：奇数名次倒排优先、偶数名次稠密优先，两通道在 Top-K
/// 内公平交替，不再偏向任何一方。实现为全序排序键（而非互相比较的闭包），保证比较器
/// 传递、不会触发排序实现的全序校验。
pub(crate) fn fuse_rrf(
    bm25_ranked: &[String],
    dense_ranked: &[String],
    k_rrf: u32,
) -> Vec<ScoredChunk> {
    let mut ids: Vec<&String> = Vec::new();
    for chunk_id in bm25_ranked.iter().chain(dense_ranked.iter()) {
        if !ids.contains(&chunk_id) {
            ids.push(chunk_id);
        }
    }
    let mut scored: Vec<ScoredChunk> = ids
        .into_iter()
        .map(|chunk_id| {
            let bm25_rank = bm25_ranked
                .iter()
                .position(|candidate| candidate == chunk_id)
                .map(|index| index + 1);
            let dense_rank = dense_ranked
                .iter()
                .position(|candidate| candidate == chunk_id)
                .map(|index| index + 1);
            let component = |rank: Option<usize>| {
                rank.map_or(0.0, |rank| 1.0 / (k_rrf as f64 + rank as f64))
            };
            ScoredChunk {
                chunk_id: chunk_id.clone(),
                rrf_score: component(bm25_rank) + component(dense_rank),
                bm25_rank,
                dense_rank,
            }
        })
        .collect();

    scored.sort_by(|left, right| {
        right
            .rrf_score
            .partial_cmp(&left.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| tie_break_key(left).cmp(&tie_break_key(right)))
    });
    scored
}

/// 融合结果的来源空间：倒排通道只覆盖私有片段，稠密通道只覆盖知识库实体。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceSpace {
    PrivateChunk,
    Dense,
}

fn source_space(chunk_id: &str) -> SourceSpace {
    if chunk_id.starts_with(PRIVATE_CHUNK_PREFIX) {
        SourceSpace::PrivateChunk
    } else {
        SourceSpace::Dense
    }
}

/// 同分并列时的全序排序键（越小越靠前）。
///
/// 依次比较：双通道命中优先 → 奇偶交替位次 → 来源 → id。交替位次使并列的
/// 「倒排第 r 名」与「稠密第 r 名」在奇数 r 时倒排靠前、偶数 r 时稠密靠前，
/// 于是 Top-K 内呈现 `倒排#1, 稠密#1, 稠密#2, 倒排#2, 倒排#3…` 的公平交替。
fn tie_break_key(entry: &ScoredChunk) -> (u8, usize, u8, &str) {
    let dual = entry.bm25_rank.is_some() && entry.dense_rank.is_some();
    let rank = match (entry.bm25_rank, entry.dense_rank) {
        (Some(bm25), Some(dense)) => bm25.min(dense),
        (Some(bm25), None) => bm25,
        (None, Some(dense)) => dense,
        (None, None) => usize::MAX,
    };
    let space = source_space(&entry.chunk_id);
    // 奇数名次倒排优先，偶数名次稠密优先。
    let private_first = rank % 2 == 1;
    let offset = usize::from((space == SourceSpace::PrivateChunk) != private_first);
    (
        u8::from(!dual),
        rank.saturating_sub(1) * 2 + offset,
        match space {
            SourceSpace::PrivateChunk => 0,
            SourceSpace::Dense => 1,
        },
        entry.chunk_id.as_str(),
    )
}

/// Top-K 来源配额：两通道都有候选时，每侧至少保留 `floor` 个槽位。
///
/// 融合后的候选可能被单一来源整段占据（例如稠密通道的前 5 名都在前），此时私有课件
/// 片段会被挤出注入提示词的上下文。这里显式保留每侧下限，剩余槽位再按融合顺序补足；
/// 返回结果仍保持融合顺序，不改变已入选条目的相对次序。
fn apply_source_quota(sorted: Vec<ScoredChunk>, top_k: usize, floor: usize) -> Vec<ScoredChunk> {
    if sorted.len() <= top_k {
        return sorted;
    }
    let has_private = sorted
        .iter()
        .any(|entry| source_space(&entry.chunk_id) == SourceSpace::PrivateChunk);
    let has_dense = sorted
        .iter()
        .any(|entry| source_space(&entry.chunk_id) == SourceSpace::Dense);
    if !(has_private && has_dense) {
        return sorted.into_iter().take(top_k).collect();
    }

    let mut selected = vec![false; sorted.len()];
    let mut chosen = 0usize;
    for space in [SourceSpace::PrivateChunk, SourceSpace::Dense] {
        let mut taken = 0usize;
        for (index, entry) in sorted.iter().enumerate() {
            if taken >= floor || chosen >= top_k {
                break;
            }
            if !selected[index] && source_space(&entry.chunk_id) == space {
                selected[index] = true;
                taken += 1;
                chosen += 1;
            }
        }
    }
    for keep in selected.iter_mut() {
        if chosen >= top_k {
            break;
        }
        if !*keep {
            *keep = true;
            chosen += 1;
        }
    }
    sorted
        .into_iter()
        .zip(selected)
        .filter_map(|(entry, keep)| keep.then_some(entry))
        .collect()
}

/// 完整混合检索：倒排通道 + 稠密通道 + RRF 融合。
///
/// `dense_ranked` 由调用方用**同一套过滤条件**（学科、未删除）的向量检索结果提供；
/// 这样融合层只负责排名合并，不重复实现向量扫描。
pub(crate) fn retrieve(
    connection: &Connection,
    query: &str,
    subject_code: Option<&str>,
    dense_ranked: &[String],
    top_k: usize,
) -> Result<Vec<ScoredChunk>, String> {
    let bm25_ranked = bm25_top_k(connection, query, subject_code, DEFAULT_CHANNEL_TOP_K)?;
    Ok(fuse_rrf(&bm25_ranked, dense_ranked, DEFAULT_K_RRF)
        .into_iter()
        .take(top_k)
        .collect())
}

/// 倒排索引当前覆盖的 chunk 数（测试与诊断用）。
///
/// ⚠ 外部内容表的 `COUNT(*)` 会**读穿到主表**：索引为空时这里依然返回正常行数。
/// 判断索引是否真的可用必须用 [`fts_index_entries`]。
pub(crate) fn fts_index_size(connection: &Connection) -> Result<i64, String> {
    connection
        .query_row("SELECT COUNT(*) FROM private_document_chunks_fts", [], |row| {
            row.get(0)
        })
        .optional()
        .map_err(|error| format!("failed to count fts rows: {error}"))
        .map(|value| value.unwrap_or(0))
}

fn fts_table_exists(connection: &Connection) -> Result<bool, String> {
    connection
        .query_row(
            "SELECT COUNT(1) FROM sqlite_master
              WHERE type = 'table' AND name = 'private_document_chunks_fts'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .map_err(|error| format!("failed to inspect fts table: {error}"))
}

/// 倒排索引是否与主表一致（功能性覆盖率探针）。
///
/// 外部内容表的 `COUNT(*)` 会读穿主表，索引为空时依然返回正常行数；
/// `'integrity-check'` 对"被清空但主表仍在"的索引同样不报错（两者均已实测）。
/// 因此这里不做结构检查，而是抽样若干主表片段、各取一段 ≥3 字的连续文本做真实
/// MATCH：**任一样本查不到就说明索引与主表脱节**（可能是全空，也可能是部分过期，
/// 例如触发器缺失期间写入的行），返回 false。
pub(crate) fn fts_index_complete(connection: &Connection) -> Result<bool, String> {
    if !fts_table_exists(connection)? {
        return Ok(false);
    }
    const SAMPLE_ROWS: usize = 5;
    let mut statement = connection
        .prepare(
            "SELECT text FROM private_document_chunks
              WHERE text IS NOT NULL ORDER BY rowid LIMIT ?1",
        )
        .map_err(|error| format!("failed to prepare fts probe: {error}"))?;
    let rows = statement
        .query_map([SAMPLE_ROWS as i64], |row| row.get::<_, String>(0))
        .map_err(|error| format!("failed to query fts probe: {error}"))?;

    let mut sampled = 0usize;
    for row in rows {
        let text = row.map_err(|error| format!("failed to read fts probe row: {error}"))?;
        // 取第一段 ≥3 字的连续非空白文本：trigram 分词器要求查询串至少 3 个字符。
        let Some(token) = text
            .split_whitespace()
            .find(|run| run.chars().count() >= TRIGRAM_MIN_CHARS)
            .map(|run| run.chars().take(TRIGRAM_MIN_CHARS).collect::<String>())
        else {
            continue;
        };
        sampled += 1;
        let expression = format!("\"{}\"", token.replace('"', "\"\""));
        let hits: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM private_document_chunks_fts
                  WHERE private_document_chunks_fts MATCH ?1",
                [&expression],
                |row| row.get(0),
            )
            .map_err(|error| format!("failed to run fts probe match: {error}"))?;
        if hits == 0 {
            return Ok(false);
        }
    }
    // 一条样本都取不到（主表为空或无可用 token）时不宣称索引健康。
    Ok(sampled > 0)
}

/// 倒排索引自愈：主表有片段而覆盖率探针不通过时执行一次 `rebuild`，返回是否回填。
///
/// migration 0018 覆盖了"老库升级到 0016"这一已知路径；本函数是此后任何
/// "索引与主表脱节"的兜底修复入口（例如索引被手工清空，或未来再有建表型迁移
/// 忘了回填）。不挂在启动路径上，避免每次启动都付一次全量索引扫描的代价。
#[allow(dead_code)]
pub(crate) fn ensure_fts_index(connection: &Connection) -> Result<bool, String> {
    let rows: i64 = connection
        .query_row("SELECT COUNT(*) FROM private_document_chunks", [], |row| {
            row.get(0)
        })
        .map_err(|error| format!("failed to count private chunks: {error}"))?;
    if rows == 0 || fts_index_complete(connection)? {
        return Ok(false);
    }
    connection
        .execute_batch(
            "INSERT INTO private_document_chunks_fts(private_document_chunks_fts) VALUES('rebuild');",
        )
        .map_err(|error| format!("failed to rebuild fts index: {error}"))?;
    Ok(true)
}

/// 跨源混合检索的一条命中。
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HybridHit {
    /// 来源："private_chunk" | "knowledge_node" | "question"
    pub source: String,
    /// 原始实体 id（已去掉来源前缀）
    pub id: String,
    pub rrf_score: f64,
    pub bm25_rank: Option<usize>,
    pub dense_rank: Option<usize>,
    /// 稠密通道原始余弦分数（未入榜为 None）
    pub dense_score: Option<f64>,
}

/// 稠密通道覆盖的实体类型：知识库种子知识点、题库，以及私有文档片段。
///
/// 私有文档片段在**生产导入路径**中仍不生成 embedding（导入不调用任何嵌入服务，
/// 见 privateDocumentService），因此线上私有资料依旧只参与倒排通道。把
/// `private_chunk` 列入本表**不改变线上行为**：线上没有切片向量，
/// `vector::search_similar` 对该类型返回空。
///
/// 列入的唯一目的，是让评测副本（`experiments/hybrid_retrieval`，切片向量由
/// `build_chunk_vectors_eval.py` 注入到副本）能够测量「两通道覆盖同一候选集时」
/// 的融合增益。原口径下 `bm25_top_k` 只查 `private_document_chunks_fts`、稠密通道
/// 只覆盖知识点与题库，两者候选空间完全不相交，整体指标只反映**通道覆盖率**而
/// 非融合质量（hybrid 在任一来源子集上都不优于该来源的最优单通道）。
///
/// 若将来把**本机** embedding（本地 bge-m3，向量计算不出设备）接入私有资料导入，
/// 即对应 `docs/document-worker.md` 的 3b，线上才会真正启用该通道；届时需同时补上
/// 文档删除时清理切片向量行的级联逻辑。
pub(crate) const DENSE_ENTITY_TYPES: [&str; 3] = ["knowledge_node", "question", "private_chunk"];

const PRIVATE_CHUNK_PREFIX: &str = "private_chunk:";

fn split_prefixed(value: &str) -> (String, String) {
    match value.split_once(':') {
        Some((prefix, rest)) => (prefix.to_string(), rest.to_string()),
        None => ("unknown".to_string(), value.to_string()),
    }
}

/// 跨源混合检索：私有文档倒排通道 + 知识库稠密通道，RRF 融合。
///
/// * `query_embedding` 为 None 时（前端未配置嵌入服务）自动退化为纯倒排结果；
/// * 倒排通道无命中时退化为纯稠密结果；
/// * 两通道使用不同实体空间，融合前以来源前缀区分，故互不串扰。
pub(crate) fn retrieve_context(
    connection: &Connection,
    query: &str,
    embedding_model: &str,
    query_embedding: Option<&[f32]>,
    subject_code: Option<&str>,
    top_k: usize,
) -> Result<Vec<HybridHit>, String> {
    let bm25_ranked: Vec<String> = bm25_top_k(connection, query, subject_code, DEFAULT_CHANNEL_TOP_K)?
        .into_iter()
        .map(|chunk_id| format!("{PRIVATE_CHUNK_PREFIX}{chunk_id}"))
        .collect();

    let mut dense_scored: Vec<(f64, String)> = Vec::new();
    let mut dense_scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    if let Some(embedding) = query_embedding {
        if !embedding.is_empty() {
            for entity_type in DENSE_ENTITY_TYPES {
                let hits = crate::vector::search_similar(
                    connection,
                    entity_type,
                    embedding,
                    embedding_model,
                    DEFAULT_CHANNEL_TOP_K as i64,
                    None,
                )?;
                for hit in hits {
                    let key = format!("{entity_type}:{}", hit.entity_id);
                    dense_scores.insert(key.clone(), hit.score);
                    dense_scored.push((hit.score, key));
                }
            }
        }
    }
    dense_scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let dense_ranked: Vec<String> = dense_scored
        .into_iter()
        .take(DEFAULT_CHANNEL_TOP_K)
        .map(|(_, key)| key)
        .collect();

    // 先整体融合，再按来源配额截断：两通道都有候选时各保留下限槽位，避免单源霸榜。
    let fused = apply_source_quota(fuse_rrf(&bm25_ranked, &dense_ranked, DEFAULT_K_RRF), top_k, PER_SOURCE_FLOOR);
    Ok(fused
        .into_iter()
        .map(|entry| {
            let (source, id) = split_prefixed(&entry.chunk_id);
            let dense_score = dense_scores.get(&entry.chunk_id).copied();
            HybridHit {
                source,
                id,
                rrf_score: entry.rrf_score,
                bm25_rank: entry.bm25_rank,
                dense_rank: entry.dense_rank,
                dense_score,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connection() -> Connection {
        let connection = Connection::open_in_memory().expect("in-memory db");
        crate::database::apply_migrations(&connection).expect("migrations");
        connection
    }

    fn seed_document(connection: &Connection, doc_id: &str, subject: &str, chunks: &[(&str, &str)]) {
        connection
            .execute(
                "INSERT INTO private_documents (
                   id, student_id, subject_code, file_name, file_type, title,
                   source_type, status, chunk_count, created_at, updated_at, deleted_at
                 ) VALUES (?1, 'local-default-student', ?2, 'f.pdf', 'pdf', '讲义',
                           'private_user_import', 'draft', ?3, '2026-09-10T00:00:00.000Z',
                           '2026-09-10T00:00:00.000Z', NULL)",
                rusqlite::params![doc_id, subject, chunks.len() as i64],
            )
            .expect("document row");
        for (index, (chunk_id, text)) in chunks.iter().enumerate() {
            connection
                .execute(
                    "INSERT INTO private_document_chunks (
                       id, document_id, chunk_index, heading, text, token_estimate, created_at
                     ) VALUES (?1, ?2, ?3, NULL, ?4, 10, '2026-09-10T00:00:00.000Z')",
                    rusqlite::params![chunk_id, doc_id, index as i64, text],
                )
                .expect("chunk row");
        }
    }

    #[test]
    fn fts_index_tracks_main_table_row_count() {
        let connection = connection();
        assert_eq!(fts_index_size(&connection).unwrap(), 0);
        seed_document(
            &connection,
            "doc-1",
            "math",
            &[("chunk-1", "柯西不等式的等号成立条件"), ("chunk-2", "极限的直观含义")],
        );
        assert_eq!(fts_index_size(&connection).unwrap(), 2);

        connection
            .execute("DELETE FROM private_document_chunks WHERE id = 'chunk-1'", [])
            .expect("delete chunk");
        assert_eq!(fts_index_size(&connection).unwrap(), 1);
    }

    #[test]
    fn bm25_channel_finds_exact_term_first() {
        let connection = connection();
        seed_document(
            &connection,
            "doc-1",
            "math",
            &[
                ("chunk-1", "极限的直观含义与夹逼定理"),
                ("chunk-2", "柯西不等式的等号成立条件"),
            ],
        );
        let ranked = bm25_top_k(&connection, "柯西不等式", Some("math"), 20).expect("bm25");
        assert_eq!(ranked.first().map(String::as_str), Some("chunk-2"));
    }

    #[test]
    fn bm25_channel_respects_subject_and_soft_delete() {
        let connection = connection();
        seed_document(&connection, "doc-math", "math", &[("chunk-m", "柯西不等式推导")]);
        seed_document(&connection, "doc-eng", "english", &[("chunk-e", "柯西不等式英文表达")]);
        let math_only = bm25_top_k(&connection, "柯西不等式", Some("math"), 20).unwrap();
        assert_eq!(math_only, vec!["chunk-m".to_string()]);

        connection
            .execute(
                "UPDATE private_documents SET deleted_at = '2026-09-10T01:00:00.000Z' WHERE id = 'doc-math'",
                [],
            )
            .unwrap();
        assert!(bm25_top_k(&connection, "柯西不等式", Some("math"), 20).unwrap().is_empty());
        // 不过滤学科时仍能命中未删除文档
        assert_eq!(
            bm25_top_k(&connection, "柯西不等式", None, 20).unwrap(),
            vec!["chunk-e".to_string()]
        );
    }

    #[test]
    fn short_query_skips_bm25_channel_instead_of_erroring() {
        let connection = connection();
        seed_document(&connection, "doc-1", "math", &[("chunk-1", "极限的直观含义")]);
        // trigram 要求 ≥3 字符：2 字符查询必须安全退化为空候选而非 SQL 错误
        assert!(bm25_top_k(&connection, "极限", Some("math"), 20).unwrap().is_empty());
        assert!(bm25_top_k(&connection, "  ", Some("math"), 20).unwrap().is_empty());
    }

    #[test]
    fn hostile_query_text_never_breaks_match_syntax() {
        let connection = connection();
        seed_document(&connection, "doc-1", "math", &[("chunk-1", "极限的直观含义")]);
        for query in ["\"quoted\"", "NEAR(", "极限 OR (", "*", "a\"b\"c"] {
            let result = bm25_top_k(&connection, query, Some("math"), 20);
            assert!(result.is_ok(), "查询 {query:?} 不应触发 FTS5 语法错误: {result:?}");
        }
    }

    #[test]
    fn rrf_fusion_matches_hand_computed_scores() {
        let bm25 = vec!["a".to_string(), "b".to_string()];
        let dense = vec!["b".to_string(), "c".to_string()];
        let fused = fuse_rrf(&bm25, &dense, 60);

        let expected_b = 1.0 / (60.0 + 2.0) + 1.0 / (60.0 + 1.0);
        // 标准 RRF：未入榜通道记 0，不再加 K+1 的平坦兜底。
        let expected_a = 1.0 / (60.0 + 1.0);
        let expected_c = 1.0 / (60.0 + 2.0);

        assert_eq!(fused[0].chunk_id, "b", "双通道命中的文档应排最前");
        assert!((fused[0].rrf_score - expected_b).abs() < 1e-9);
        let a = fused.iter().find(|entry| entry.chunk_id == "a").expect("a");
        let c = fused.iter().find(|entry| entry.chunk_id == "c").expect("c");
        assert!((a.rrf_score - expected_a).abs() < 1e-9);
        assert!((c.rrf_score - expected_c).abs() < 1e-9);
        assert_eq!(a.bm25_rank, Some(1));
        assert_eq!(a.dense_rank, None);
        assert!(
            a.rrf_score > c.rrf_score,
            "同为单通道命中时，第 1 名必须高于第 2 名"
        );
    }

    #[test]
    fn empty_bm25_channel_degenerates_to_dense_order() {
        let dense = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let fused = fuse_rrf(&[], &dense, 60);
        let order: Vec<&str> = fused.iter().map(|entry| entry.chunk_id.as_str()).collect();
        assert_eq!(order, vec!["x", "y", "z"]);
    }

    #[test]
    fn empty_dense_channel_degenerates_to_bm25_order() {
        let bm25 = vec!["p".to_string(), "q".to_string()];
        let fused = fuse_rrf(&bm25, &[], 60);
        let order: Vec<&str> = fused.iter().map(|entry| entry.chunk_id.as_str()).collect();
        assert_eq!(order, vec!["p", "q"]);
    }

    #[test]
    fn retrieve_fuses_both_channels_end_to_end() {
        let connection = connection();
        seed_document(
            &connection,
            "doc-1",
            "math",
            &[("chunk-term", "柯西不等式的等号成立条件"), ("chunk-semantic", "向量内积与夹角公式")],
        );
        // chunk-term 同时被两通道命中（bm25 第 1、dense 第 1），chunk-semantic 仅被稠密命中
        let dense = vec!["chunk-term".to_string(), "chunk-semantic".to_string()];
        let fused = retrieve(&connection, "柯西不等式", Some("math"), &dense, 10).unwrap();
        let ids: Vec<&str> = fused.iter().map(|entry| entry.chunk_id.as_str()).collect();
        assert!(ids.contains(&"chunk-term"));
        assert!(ids.contains(&"chunk-semantic"));
        assert_eq!(
            fused[0].chunk_id, "chunk-term",
            "双通道同时命中的应排最前"
        );
        assert_eq!(fused[0].bm25_rank, Some(1));
        assert_eq!(fused[0].dense_rank, Some(1));
        let semantic = fused
            .iter()
            .find(|entry| entry.chunk_id == "chunk-semantic")
            .expect("semantic");
        assert_eq!(semantic.bm25_rank, None);
        assert_eq!(semantic.dense_rank, Some(2));
    }

    #[test]
    fn deleting_chunks_shrinks_hybrid_results() {
        let connection = connection();
        let chunks: Vec<(String, String)> = (0..100)
            .map(|index| (format!("chunk-{index}"), format!("柯西不等式 讲义片段 {index}")))
            .collect();
        let borrowed: Vec<(&str, &str)> = chunks
            .iter()
            .map(|(id, text)| (id.as_str(), text.as_str()))
            .collect();
        seed_document(&connection, "doc-big", "math", &borrowed);
        assert_eq!(bm25_top_k(&connection, "柯西不等式", Some("math"), 20).unwrap().len(), 20);

        connection
            .execute(
                "DELETE FROM private_document_chunks WHERE chunk_index < 50",
                [],
            )
            .unwrap();
        let remaining = bm25_top_k(&connection, "柯西不等式", Some("math"), 200).unwrap();
        assert_eq!(remaining.len(), 50);
        assert!(!remaining.iter().any(|id| id == "chunk-0"));
    }

    /// 性能基线（论文 §4.2）：1500 片段规模下单次混合检索 < 15 ms（release 构建）。
    /// 运行：cargo test --release bench_hybrid_retrieval_latency -- --ignored --nocapture
    #[test]
    #[ignore = "性能基准，需 --ignored 显式运行（release 构建）"]
    fn bench_hybrid_retrieval_latency() {
        let connection = connection();
        let chunks: Vec<(String, String)> = (0..1500)
            .map(|index| {
                (
                    format!("chunk-{index}"),
                    format!("第 {index} 段讲义：柯西不等式的等号成立条件与向量内积推导"),
                )
            })
            .collect();
        let borrowed: Vec<(&str, &str)> = chunks
            .iter()
            .map(|(id, text)| (id.as_str(), text.as_str()))
            .collect();
        seed_document(&connection, "doc-bench", "math", &borrowed);

        let dense: Vec<String> = (0..20).map(|index| format!("chunk-{index}")).collect();
        let mut timings = Vec::new();
        for _ in 0..30 {
            let started = std::time::Instant::now();
            let fused = retrieve(&connection, "柯西不等式 等号成立", Some("math"), &dense, 5).unwrap();
            assert_eq!(fused.len(), 5);
            timings.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = timings.iter().sum::<f64>() / timings.len() as f64;
        let p95 = timings[((timings.len() - 1) as f64 * 0.95).round() as usize];
        println!("hybrid_retrieval_ms mean={mean:.2} p95={p95:.2} max={:.2}", timings.last().unwrap());
    }

    fn seed_embedding(connection: &Connection, entity_type: &str, entity_id: &str, embedding: Vec<f32>, model: &str) {
        crate::vector::upsert_embedding(
            connection,
            &crate::vector::EmbeddingEntry {
                id: format!("{entity_type}:{entity_id}"),
                entity_type: entity_type.to_string(),
                entity_id: entity_id.to_string(),
                embedding,
                embedding_model: model.to_string(),
                embedding_dim: 3,
            },
        )
        .expect("embedding row");
    }

    #[test]
    fn context_retrieval_without_embedding_returns_private_chunks_only() {
        let connection = connection();
        seed_document(&connection, "doc-1", "math", &[("chunk-1", "柯西不等式的等号成立条件")]);

        let hits = retrieve_context(&connection, "柯西不等式", "bge-m3", None, Some("math"), 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|hit| hit.source == "private_chunk"));
        assert_eq!(hits[0].id, "chunk-1");
        assert_eq!(hits[0].bm25_rank, Some(1));
        assert_eq!(hits[0].dense_rank, None);
    }

    #[test]
    fn context_retrieval_with_embedding_merges_knowledge_base_and_private_chunks() {
        let connection = connection();
        seed_document(&connection, "doc-1", "math", &[("chunk-1", "柯西不等式的等号成立条件")]);
        seed_embedding(&connection, "knowledge_node", "node-1", vec![1.0, 0.0, 0.0], "bge-m3");
        seed_embedding(&connection, "question", "q-1", vec![0.0, 1.0, 0.0], "bge-m3");

        // 查询向量与 node-1 完全同向 → 稠密第 1；倒排通道命中 chunk-1
        let hits = retrieve_context(
            &connection,
            "柯西不等式",
            "bge-m3",
            Some(&[1.0, 0.0, 0.0]),
            Some("math"),
            5,
        )
        .unwrap();

        let sources: Vec<&str> = hits.iter().map(|hit| hit.source.as_str()).collect();
        assert!(sources.contains(&"private_chunk"));
        assert!(sources.contains(&"knowledge_node"));
        let node_hit = hits.iter().find(|hit| hit.source == "knowledge_node").expect("node hit");
        assert_eq!(node_hit.id, "node-1");
        assert_eq!(node_hit.dense_rank, Some(1));
        assert!(node_hit.dense_score.unwrap() > 0.99);
        // 私有片段只可能来自倒排通道
        let chunk_hit = hits.iter().find(|hit| hit.source == "private_chunk").expect("chunk hit");
        assert_eq!(chunk_hit.dense_rank, None);
    }

    #[test]
    fn context_retrieval_filters_dense_channel_by_embedding_model() {
        let connection = connection();
        seed_embedding(&connection, "knowledge_node", "node-bge", vec![1.0, 0.0, 0.0], "bge-m3");
        seed_embedding(&connection, "knowledge_node", "node-openai", vec![1.0, 0.0, 0.0], "other-model");

        let hits = retrieve_context(
            &connection,
            "任何查询",
            "bge-m3",
            Some(&[1.0, 0.0, 0.0]),
            None,
            5,
        )
        .unwrap();
        let ids: Vec<&str> = hits.iter().map(|hit| hit.id.as_str()).collect();
        assert!(ids.contains(&"node-bge"));
        assert!(!ids.contains(&"node-openai"), "不同嵌入模型的向量不得混入同一通道");
    }

    #[test]
    fn context_retrieval_degrades_to_dense_when_no_private_match() {
        let connection = connection();
        seed_document(&connection, "doc-1", "math", &[("chunk-1", "完全无关的另一段讲义文字")]);
        seed_embedding(&connection, "knowledge_node", "node-1", vec![1.0, 0.0, 0.0], "bge-m3");

        let hits = retrieve_context(
            &connection,
            "柯西不等式",
            "bge-m3",
            Some(&[1.0, 0.0, 0.0]),
            Some("math"),
            5,
        )
        .unwrap();
        assert_eq!(hits[0].source, "knowledge_node");
        assert_eq!(hits[0].id, "node-1");
    }

    /// 回归：FTS5 外部内容表不会索引"迁移前已存在"的行，且其 COUNT(*) 会读穿主表
    /// 从而掩盖索引为空（诊断接口显示正常、MATCH 却查不到）——这正是 migration 0018
    /// 存在的理由。本测试复现该失效模式，并证明 rebuild 能回填。
    #[test]
    fn fts_rebuild_indexes_rows_that_predate_the_virtual_table() {
        let connection = connection();
        // 触发器存在时插入的行会被正常索引，先确认基线可用。
        seed_document(
            &connection,
            "doc-1",
            "math",
            &[("chunk-1", "柯西不等式的等号成立条件与配方法证明")],
        );
        assert_eq!(
            bm25_top_k(&connection, "柯西不等式", None, 5).unwrap(),
            vec!["chunk-1".to_string()]
        );

        // 模拟升级前就已存在的数据：去掉触发器后直接写入主表，倒排索引不会更新。
        connection
            .execute_batch(
                "DROP TRIGGER private_document_chunks_fts_insert;
                 DROP TRIGGER private_document_chunks_fts_delete;
                 DROP TRIGGER private_document_chunks_fts_update;",
            )
            .expect("drop triggers");
        seed_document(
            &connection,
            "doc-2",
            "math",
            &[("chunk-legacy", "单调有界准则与夹逼准则判定极限存在")],
        );

        // 失效模式：MATCH 查不到既有行……
        assert!(bm25_top_k(&connection, "夹逼准则", None, 5).unwrap().is_empty());
        // ……但外部内容表的计数读穿主表，看上去仍然"正常"（2 行）。
        assert_eq!(fts_index_size(&connection).unwrap(), 2);
        // 可信探针（真实 MATCH 自检）能看出索引里其实什么都查不到。
        assert!(!fts_index_complete(&connection).unwrap());
        assert!(ensure_fts_index(&connection).unwrap());

        // 0018 的动作：显式 rebuild 回填既有语料。
        assert_eq!(
            bm25_top_k(&connection, "夹逼准则", None, 5).unwrap(),
            vec!["chunk-legacy".to_string()]
        );
        assert!(fts_index_complete(&connection).unwrap());
        // 已健康时自愈不再重复回填。
        assert!(!ensure_fts_index(&connection).unwrap());
    }

    #[test]
    fn question_shell_stripping_keeps_core_term() {
        assert_eq!(strip_question_shell("偏振光是什么意思"), "偏振光");
        assert_eq!(strip_question_shell("柯西不等式怎么算"), "柯西不等式");
        assert_eq!(strip_question_shell("什么是矩阵的秩"), "矩阵的秩");
        assert_eq!(strip_question_shell("请问什么是极限的定义"), "极限");
        assert_eq!(strip_question_shell("夹逼准则？"), "夹逼准则");
        assert_eq!(strip_question_shell(" 泰勒公式  "), "泰勒公式");
        // 不含外壳的查询原样返回
        assert_eq!(strip_question_shell("柯西不等式"), "柯西不等式");
    }

    #[test]
    fn bm25_channel_recalls_question_form_query() {
        let connection = connection();
        seed_document(
            &connection,
            "doc-1",
            "math",
            &[("chunk-optics", "偏振光；双折射现象；布儒斯特定律与马吕斯定律")],
        );
        // 回归（D-407）：整串短语匹配下 "偏振光是什么意思" 必然召回为空
        let ranked = bm25_top_k(&connection, "偏振光是什么意思", Some("math"), 20).expect("bm25");
        assert_eq!(ranked, vec!["chunk-optics".to_string()]);
        // 疑问前缀同样可剥离
        let ranked = bm25_top_k(&connection, "请问什么是偏振光", Some("math"), 20).expect("bm25");
        assert_eq!(ranked, vec!["chunk-optics".to_string()]);
    }

    #[test]
    fn trigram_fallback_recalls_when_phrase_misses() {
        let connection = connection();
        seed_document(
            &connection,
            "doc-1",
            "math",
            &[("chunk-phase", "相位差与光程差的关系及其干涉条件")],
        );
        // 长叙述句：剥离外壳后仍不构成正文中的整串，短语必失手，靠 3-gram OR 兜底召回
        let ranked =
            bm25_top_k(&connection, "椭圆偏振光通过玻片后相位差怎么变化", Some("math"), 20)
                .expect("bm25");
        assert_eq!(ranked, vec!["chunk-phase".to_string()]);
    }

    #[test]
    fn tied_ranks_alternate_between_sources() {
        // 两通道实体空间正交：并列只发生在「同名次的跨通道」之间。
        let bm25: Vec<String> = ["c1", "c2", "c3"]
            .iter()
            .map(|id| format!("{PRIVATE_CHUNK_PREFIX}{id}"))
            .collect();
        let dense: Vec<String> = ["n1", "n2", "n3"]
            .iter()
            .map(|id| format!("knowledge_node:{id}"))
            .collect();
        let fused = fuse_rrf(&bm25, &dense, 60);
        let order: Vec<&str> = fused.iter().map(|entry| entry.chunk_id.as_str()).collect();
        // 奇数名次倒排优先、偶数名次稠密优先 → 公平交替，不再按 id 字典序偏向节点
        assert_eq!(
            order,
            vec![
                "private_chunk:c1",
                "knowledge_node:n1",
                "knowledge_node:n2",
                "private_chunk:c2",
                "private_chunk:c3",
                "knowledge_node:n3",
            ]
        );
    }

    #[test]
    fn source_quota_reserves_slots_for_each_source() {
        // 构造单源霸榜：稠密占前 5，私有片段掉在第 6、7 位
        let entry = |chunk_id: &str, bm25_rank: Option<usize>, dense_rank: Option<usize>| {
            let component =
                |rank: Option<usize>| rank.map_or(0.0, |rank| 1.0 / (60.0 + rank as f64));
            ScoredChunk {
                chunk_id: chunk_id.to_string(),
                rrf_score: component(bm25_rank) + component(dense_rank),
                bm25_rank,
                dense_rank,
            }
        };
        let sorted = vec![
            entry("knowledge_node:n1", None, Some(1)),
            entry("knowledge_node:n2", None, Some(2)),
            entry("knowledge_node:n3", None, Some(3)),
            entry("knowledge_node:n4", None, Some(4)),
            entry("knowledge_node:n5", None, Some(5)),
            entry("private_chunk:c1", Some(6), None),
            entry("private_chunk:c2", Some(7), None),
        ];
        let selected = apply_source_quota(sorted, 5, PER_SOURCE_FLOOR);
        let ids: Vec<&str> = selected.iter().map(|item| item.chunk_id.as_str()).collect();
        assert_eq!(ids.len(), 5);
        assert!(ids.contains(&"private_chunk:c1"));
        assert!(ids.contains(&"private_chunk:c2"));
        // 剩余槽位仍按融合顺序补足
        assert_eq!(ids[0], "knowledge_node:n1");
    }
}
