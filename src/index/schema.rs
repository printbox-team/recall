use crate::session::{SearchResult, Session, SessionSource};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::{
    BooleanQuery, BoostQuery, DisjunctionMaxQuery, Occur, PhrasePrefixQuery, Query, TermQuery,
};
use tantivy::schema::*;
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::{
    AsciiFoldingFilter, LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer,
};
use tantivy::{
    doc, DocId, Index, IndexReader, IndexWriter, ReloadPolicy, Score, Searcher, SegmentReader, Term,
};

/// Name of the text analyzer used for searchable fields and queries
const ANALYZER: &str = "recall";

/// Age at which a session's recency boost drops to half its peak.
const RECENCY_HALF_LIFE_SECS: f64 = 14.0 * 24.0 * 3600.0;
/// Peak extra weight for a session active right now (multiplier `1 + RECENCY_MAX_BOOST`).
const RECENCY_MAX_BOOST: f64 = 2.0;

/// Relevance multiplier for a session last active `age_secs` ago:
/// 3x for a session active now, 2x at 14 days, 1.5x at 28 days, approaching 1x.
fn recency_multiplier(age_secs: f64) -> Score {
    let age = age_secs.max(0.0);
    (1.0 + RECENCY_MAX_BOOST * (-age / RECENCY_HALF_LIFE_SECS).exp2()) as Score
}

/// Title hits outrank the same words inside a message
const TITLE_BOOST: Score = 5.0;
/// A project or branch name match ranks above a passing mention in a message
const PROJECT_BOOST: Score = 2.0;
const CONTENT_PHRASE_BOOST: Score = 10.0;
const TITLE_PHRASE_BOOST: Score = 50.0;

/// Score of a prefix expansion relative to an exact match, scaled further by how
/// much of the expanded word was typed ("recenc" -> "recency" scores higher than
/// "rec" -> "recency").
const PREFIX_WEIGHT: Score = 0.5;
const MAX_PREFIX_EXPANSIONS: usize = 64;

/// How many terms a typed word expands to: none for a single character (it would
/// match nearly everything), a few for two characters, the most frequent 64 beyond.
fn prefix_expansion_limit(word_chars: usize) -> usize {
    match word_chars {
        0 | 1 => 0,
        2 => 16,
        _ => MAX_PREFIX_EXPANSIONS,
    }
}

fn term_query(field: Field, text: &str, boost: Score) -> Box<dyn Query> {
    let query = TermQuery::new(Term::from_field_text(field, text), IndexRecordOption::WithFreqs);
    Box::new(BoostQuery::new(Box::new(query), boost))
}

/// Smallest byte string greater than every string starting with `prefix`
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut bound = prefix.to_vec();
    while let Some(last) = bound.pop() {
        if last != u8::MAX {
            bound.push(last + 1);
            return Some(bound);
        }
    }
    None
}

/// Indexed terms of `field` that start with `prefix` (excluding `prefix` itself),
/// the `limit` most frequent first.
fn expand_prefix(searcher: &Searcher, field: Field, prefix: &str, limit: usize) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut doc_freqs: HashMap<Vec<u8>, u32> = HashMap::new();
    for segment in searcher.segment_readers() {
        let inverted_index = segment.inverted_index(field)?;
        let terms = inverted_index.terms();
        let mut range = terms.range().gt(prefix.as_bytes());
        if let Some(upper) = prefix_upper_bound(prefix.as_bytes()) {
            range = range.lt(upper);
        }
        let mut stream = range.into_stream()?;
        while stream.advance() {
            *doc_freqs.entry(stream.key().to_vec()).or_default() += stream.value().doc_freq;
        }
    }
    let mut expansions: Vec<(Vec<u8>, u32)> = doc_freqs.into_iter().collect();
    expansions.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(expansions
        .into_iter()
        .take(limit)
        .filter_map(|(term, _)| String::from_utf8(term).ok())
        .collect())
}

/// Get the default cache directory for the index
pub fn default_index_path() -> PathBuf {
    std::env::var("RECALL_HOME_OVERRIDE")
        .map(|h| PathBuf::from(h).join(".cache").join("recall").join("index"))
        .unwrap_or_else(|_| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("recall")
                .join("index")
        })
}

/// Wrapper around Tantivy index for session search
pub struct SessionIndex {
    index: Index,
    reader: IndexReader,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    session_id: Field,
    source: Field,
    file_path: Field,
    cwd: Field,
    git_branch: Field,
    timestamp: Field,
    content: Field,
    title: Field,
    project: Field,
    message_index: Field,
}

impl SessionIndex {
    /// Open existing index or create a new one at the default path
    pub fn open_default() -> Result<Self> {
        Self::open_or_create(&default_index_path())
    }

    /// Open existing index or create a new one
    pub fn open_or_create(index_path: &Path) -> Result<Self> {
        // If the persisted schema version differs from the current build,
        // wipe the index dir + state.json so the new schema is used. The
        // sibling state file is the source of truth for the version.
        let state_path = index_path
            .parent()
            .map(|p| p.join("state.json"))
            .unwrap_or_else(|| index_path.join("state.json"));
        crate::index::IndexState::migrate_if_needed(&state_path, index_path)?;

        std::fs::create_dir_all(index_path)?;

        let schema = Self::build_schema();

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(index_path).context("Failed to open existing index")?
        } else {
            Index::create_in_dir(index_path, schema.clone())
                .context("Failed to create new index")?
        };

        index.tokenizers().register(ANALYZER, Self::analyzer());

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        Ok(Self {
            index,
            reader,
            session_id: schema.get_field("session_id").unwrap(),
            source: schema.get_field("source").unwrap(),
            file_path: schema.get_field("file_path").unwrap(),
            cwd: schema.get_field("cwd").unwrap(),
            git_branch: schema.get_field("git_branch").unwrap(),
            timestamp: schema.get_field("timestamp").unwrap(),
            content: schema.get_field("content").unwrap(),
            title: schema.get_field("title").unwrap(),
            project: schema.get_field("project").unwrap(),
            message_index: schema.get_field("message_index").unwrap(),
            schema,
        })
    }

    fn build_schema() -> Schema {
        let mut builder = Schema::builder();

        // Stored metadata fields
        builder.add_text_field("session_id", STRING | STORED);
        builder.add_text_field("source", STRING | STORED);
        builder.add_text_field("file_path", STRING | STORED);
        builder.add_text_field("cwd", STRING | STORED);
        builder.add_text_field("git_branch", STRING | STORED);

        // Timestamp for recency sorting (stored as i64 unix timestamp)
        builder.add_i64_field("timestamp", INDEXED | STORED | FAST);

        // Message index within the session (for match-recency)
        builder.add_u64_field("message_index", STORED);

        let searchable = TextFieldIndexing::default()
            .set_tokenizer(ANALYZER)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);

        // Searchable content field
        builder.add_text_field(
            "content",
            TextOptions::default().set_indexing_options(searchable.clone()).set_stored(),
        );

        // Searchable session title (custom-title / ai-title for Claude, title for Factory/OpenCode)
        builder.add_text_field(
            "title",
            TextOptions::default().set_indexing_options(searchable.clone()).set_stored(),
        );

        // Project folder name and git branch, so a session can be found by where it ran
        builder.add_text_field("project", TextOptions::default().set_indexing_options(searchable));

        builder.build()
    }

    /// Word splitting + lowercase + diacritic folding, so "blad" matches "błąd"
    fn analyzer() -> TextAnalyzer {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(AsciiFoldingFilter)
            .build()
    }

    /// Get a writer for indexing operations
    pub fn writer(&self) -> Result<IndexWriter> {
        self.index
            .writer(50_000_000) // 50MB heap
            .context("Failed to create index writer")
    }

    /// Index a single session (all its messages)
    pub fn index_session(&self, writer: &mut IndexWriter, session: &Session) -> Result<()> {
        let timestamp_secs = session.timestamp.timestamp();
        let title = session.title.clone().unwrap_or_default();
        let project = format!(
            "{} {}",
            session.project_name(),
            session.git_branch.as_deref().unwrap_or_default()
        );

        // Index each message separately for match-recency ranking.
        // Title is duplicated on each message doc so a title hit returns any
        // message in the session, and the per-session grouping later collapses them.
        for (idx, message) in session.messages.iter().enumerate() {
            let doc = doc!(
                self.session_id => session.id.clone(),
                self.source => session.source.as_str(),
                self.file_path => session.file_path.to_string_lossy().to_string(),
                self.cwd => session.cwd.clone(),
                self.git_branch => session.git_branch.clone().unwrap_or_default(),
                self.timestamp => timestamp_secs,
                self.message_index => idx as u64,
                self.content => message.content.clone(),
                self.title => title.clone(),
                self.project => project.clone(),
            );
            writer.add_document(doc)?;
        }

        Ok(())
    }

    /// Delete all documents for a session (by file path)
    pub fn delete_session(&self, writer: &mut IndexWriter, file_path: &Path) {
        let term = tantivy::Term::from_field_text(
            self.file_path,
            &file_path.to_string_lossy(),
        );
        writer.delete_term(term);
    }

    /// Reload the reader to see recent changes
    pub fn reload(&self) -> Result<()> {
        self.reader.reload().context("Failed to reload reader")
    }

    /// Search for sessions matching the query
    /// Returns results grouped by session, ranked by match-recency
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<SearchResult>> {
        if query_str.trim().is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let tokens = self.tokenize_query(query_str);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for (_, word) in &tokens {
            clauses.push((Occur::Should, self.word_query(&searcher, word)?));
        }

        // Boost phrase matches for multi-word queries; the last word may still be
        // half-typed, so it matches as a prefix.
        if tokens.len() > 1 {
            for (field, boost) in [(self.content, CONTENT_PHRASE_BOOST), (self.title, TITLE_PHRASE_BOOST)] {
                let terms = tokens
                    .iter()
                    .map(|(position, word)| (*position, Term::from_field_text(field, word)))
                    .collect();
                let mut phrase = PhrasePrefixQuery::new_with_offset(terms);
                phrase.set_max_expansions(MAX_PREFIX_EXPANSIONS as u32);
                clauses.push((Occur::Should, Box::new(BoostQuery::new(Box::new(phrase), boost))));
            }
        }

        let query: Box<dyn Query> = Box::new(BooleanQuery::new(clauses));

        // Get more results than limit to group by session. Recency is applied inside
        // the collector so recent sessions compete for the candidate pool instead of
        // being crowded out by older sessions that repeat the term more often.
        let now = chrono::Utc::now().timestamp();
        let collector = TopDocs::with_limit(limit * 20).tweak_score(
            move |segment_reader: &SegmentReader| {
                let timestamps = segment_reader
                    .fast_fields()
                    .i64("timestamp")
                    .expect("timestamp is a fast field")
                    .first_or_default_col(now);
                move |doc: DocId, score: Score| {
                    let age_secs = (now - timestamps.get_val(doc)) as f64;
                    score * recency_multiplier(age_secs)
                }
            },
        );
        let top_docs = searcher.search(&query, &collector)?;

        // Group by session, keeping the best-scoring message per session.
        // Scores already include the recency multiplier.
        let mut best_per_session: std::collections::HashMap<String, (f32, f32, TantivyDocument)> =
            std::collections::HashMap::new();
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_addr)?;
            let session_id = doc
                .get_first(self.session_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let message_index = doc
                .get_first(self.message_index)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            // Among similarly scored messages prefer the later one in the session
            let rank = score + (message_index as f32) * 0.01;
            match best_per_session.entry(session_id) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if rank > e.get().0 {
                        e.insert((rank, score, doc));
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((rank, score, doc));
                }
            }
        }

        let mut best: Vec<_> = best_per_session.into_values().collect();
        best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        best.truncate(limit);

        // Snippets are only built for the sessions that are actually returned
        let mut snippet_generator = SnippetGenerator::create(&searcher, &*query, self.content)?;
        snippet_generator.set_max_num_chars(200);

        let results = best
            .into_iter()
            .map(|(_, score, doc)| self.search_result(&doc, score, &snippet_generator))
            .collect();

        Ok(results)
    }

    /// Build a search result from a matched message doc, with a highlighted snippet
    fn search_result(
        &self,
        doc: &TantivyDocument,
        score: f32,
        snippet_generator: &SnippetGenerator,
    ) -> SearchResult {
        let text = |field: Field| {
            doc.get_first(field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let non_empty = |s: String| Some(s).filter(|s| !s.is_empty());

        let source = SessionSource::parse(&text(self.source)).unwrap_or(SessionSource::ClaudeCode);
        let timestamp_secs = doc
            .get_first(self.timestamp)
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let message_index = doc
            .get_first(self.message_index)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let tantivy_snippet = snippet_generator.snippet_from_doc(doc);
        let match_fragment = tantivy_snippet.fragment().to_string();
        let snippet = match_fragment.replace('\n', " ");
        let match_spans = tantivy_snippet
            .highlighted()
            .iter()
            .map(|r| (r.start, r.end))
            .collect();

        SearchResult {
            session: Session {
                id: text(self.session_id),
                source,
                file_path: PathBuf::from(text(self.file_path)),
                cwd: text(self.cwd),
                git_branch: non_empty(text(self.git_branch)),
                timestamp: chrono::DateTime::from_timestamp(timestamp_secs, 0).unwrap_or_default(),
                title: non_empty(text(self.title)),
                messages: Vec::new(), // We don't load all messages for search results
            },
            score,
            matched_message_index: message_index,
            snippet,
            match_spans,
            match_fragment,
        }
    }

    /// Split the query into (position, term text) tokens with the analyzer used
    /// to index content, so query terms match indexed terms.
    fn tokenize_query(&self, query_str: &str) -> Vec<(usize, String)> {
        let mut tokens = Vec::new();
        if let Some(mut analyzer) = self.index.tokenizers().get(ANALYZER) {
            let mut stream = analyzer.token_stream(query_str);
            stream.process(&mut |token| tokens.push((token.position, token.text.clone())));
        }
        tokens
    }

    /// Match one query word in content, title and project: the exact term, plus the most
    /// frequent terms it is a prefix of (weighted lower) so half-typed words match.
    /// Per field the best alternative counts, so a short prefix doesn't add up
    /// the scores of every word it expands to.
    fn word_query(&self, searcher: &Searcher, word: &str) -> Result<Box<dyn Query>> {
        let word_chars = word.chars().count();
        let mut fields: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for (field, field_boost) in [
            (self.content, 1.0),
            (self.title, TITLE_BOOST),
            (self.project, PROJECT_BOOST),
        ] {
            let mut alternatives = vec![term_query(field, word, 1.0)];
            for term in expand_prefix(searcher, field, word, prefix_expansion_limit(word_chars))? {
                let weight = PREFIX_WEIGHT * word_chars as f32 / term.chars().count() as f32;
                alternatives.push(term_query(field, &term, weight));
            }
            let best = DisjunctionMaxQuery::with_tie_breaker(alternatives, 0.1);
            fields.push((Occur::Should, Box::new(BoostQuery::new(Box::new(best), field_boost))));
        }
        Ok(Box::new(BooleanQuery::new(fields)))
    }

    /// Get recent sessions sorted by timestamp (most recent first)
    pub fn recent(&self, limit: usize) -> Result<Vec<SearchResult>> {
        use tantivy::collector::TopDocs;
        use tantivy::query::AllQuery;

        let searcher = self.reader.searcher();

        // Get all docs sorted by timestamp descending
        // Fetch many more docs since each session has multiple messages indexed
        let top_docs = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(limit * 100).order_by_fast_field::<i64>("timestamp", tantivy::Order::Desc),
        )?;

        // Group by session, keeping only the most recent per session
        let mut session_results: std::collections::HashMap<String, SearchResult> =
            std::collections::HashMap::new();

        for (_score, doc_addr) in top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(doc_addr)?;

            let session_id = doc
                .get_first(self.session_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Skip if we already have this session
            if session_results.contains_key(&session_id) {
                continue;
            }

            let source_str = doc
                .get_first(self.source)
                .and_then(|v| v.as_str())
                .unwrap_or("claude");

            let source = SessionSource::parse(source_str).unwrap_or(SessionSource::ClaudeCode);

            let file_path = doc
                .get_first(self.file_path)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let cwd = doc
                .get_first(self.cwd)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let git_branch = doc
                .get_first(self.git_branch)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty());

            let timestamp_secs = doc
                .get_first(self.timestamp)
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let content = doc
                .get_first(self.content)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let title = doc
                .get_first(self.title)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty());

            // Use first part of content as snippet
            let snippet: String = content.chars().take(200).collect();
            let snippet = snippet.replace('\n', " ");

            let result = SearchResult {
                session: Session {
                    id: session_id.clone(),
                    source,
                    file_path: std::path::PathBuf::from(&file_path),
                    cwd,
                    git_branch,
                    timestamp: chrono::DateTime::from_timestamp(timestamp_secs, 0)
                        .unwrap_or_default(),
                    title,
                    messages: Vec::new(),
                },
                score: 0.0,
                matched_message_index: 0,
                snippet,
                match_spans: Vec::new(),
                match_fragment: String::new(),
            };

            session_results.insert(session_id, result);

            if session_results.len() >= limit {
                break;
            }
        }

        // Sort by timestamp descending
        let mut results: Vec<_> = session_results.into_values().collect();
        results.sort_by_key(|r| std::cmp::Reverse(r.session.timestamp));
        results.truncate(limit);

        Ok(results)
    }

    /// Look up a session by ID and return its file path
    pub fn get_by_id(&self, session_id: &str) -> Result<Option<PathBuf>> {
        let searcher = self.reader.searcher();

        let term = tantivy::Term::from_field_text(self.session_id, session_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);

        let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

        if let Some((_score, doc_addr)) = top_docs.first() {
            let doc: tantivy::TantivyDocument = searcher.doc(*doc_addr)?;

            let file_path = doc
                .get_first(self.file_path)
                .and_then(|v| v.as_str())
                .map(PathBuf::from);

            Ok(file_path)
        } else {
            Ok(None)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Message, Role};
    use chrono::{Duration, Utc};

    fn session(id: &str, age_days: i64, contents: &[&str]) -> Session {
        let timestamp = Utc::now() - Duration::days(age_days);
        Session {
            id: id.to_string(),
            source: SessionSource::ClaudeCode,
            file_path: PathBuf::from(format!("/sessions/{id}.jsonl")),
            cwd: "/project".to_string(),
            git_branch: None,
            timestamp,
            title: None,
            messages: contents
                .iter()
                .map(|c| Message {
                    role: Role::User,
                    content: c.to_string(),
                    timestamp,
                })
                .collect(),
        }
    }

    fn index_with(sessions: &[Session]) -> (tempfile::TempDir, SessionIndex) {
        let dir = tempfile::tempdir().unwrap();
        let index = SessionIndex::open_or_create(dir.path()).unwrap();
        let mut writer = index.writer().unwrap();
        for s in sessions {
            index.index_session(&mut writer, s).unwrap();
        }
        writer.commit().unwrap();
        index.reload().unwrap();
        (dir, index)
    }

    fn ids(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|r| r.session.id.as_str()).collect()
    }

    #[test]
    fn half_typed_word_matches_and_highlights_the_full_word() {
        let (_dir, index) = index_with(&[session("a", 1, &["tuning the recency boost"])]);

        let results = index.search("recenc", 10).unwrap();

        assert_eq!(ids(&results), ["a"]);
        let r = &results[0];
        let highlighted: Vec<&str> = r.match_spans.iter().map(|&(s, e)| &r.snippet[s..e]).collect();
        assert_eq!(highlighted, ["recency"]);
    }

    #[test]
    fn exact_word_outranks_a_longer_word_it_prefixes() {
        let (_dir, index) = index_with(&[
            session("prefix", 1, &["running the testing suite"]),
            session("exact", 1, &["running the test suite"]),
        ]);

        let results = index.search("test", 10).unwrap();

        assert_eq!(ids(&results), ["exact", "prefix"]);
    }

    #[test]
    fn single_character_does_not_expand_to_every_word() {
        let (_dir, index) = index_with(&[session("a", 1, &["recall search"])]);

        assert!(index.search("r", 10).unwrap().is_empty());
    }

    #[test]
    fn half_typed_last_word_still_gets_the_phrase_boost() {
        let (_dir, index) = index_with(&[
            session("scattered", 1, &["boost the index by recency"]),
            session("phrase", 1, &["so the recency boost works well enough for older sessions too"]),
        ]);

        let results = index.search("recency boo", 10).unwrap();

        assert_eq!(ids(&results)[0], "phrase");
    }

    #[test]
    fn query_without_diacritics_matches_and_highlights_polish_words() {
        let (_dir, index) = index_with(&[session("a", 1, &["Znalazłem błąd w zamówieniu"])]);

        let results = index.search("blad zamowie", 10).unwrap();

        assert_eq!(ids(&results), ["a"]);
        let r = &results[0];
        let highlighted: Vec<&str> = r.match_spans.iter().map(|&(s, e)| &r.snippet[s..e]).collect();
        assert_eq!(highlighted, ["błąd", "zamówieniu"]);
    }

    #[test]
    fn session_is_found_by_project_folder_and_git_branch() {
        let mut in_project = session("in-project", 1, &["fixed the flaky test"]);
        in_project.cwd = "/Users/me/Projects/printbox-editor".to_string();
        in_project.git_branch = Some("feature/PBX-72802-cart".to_string());
        let (_dir, index) = index_with(&[in_project, session("other", 1, &["fixed the flaky test"])]);

        assert_eq!(ids(&index.search("printbox", 10).unwrap()), ["in-project"]);
        assert_eq!(ids(&index.search("72802", 10).unwrap()), ["in-project"]);
    }

    #[test]
    fn recency_multiplier_halves_boost_every_half_life() {
        let day = 24.0 * 3600.0;
        assert!((recency_multiplier(0.0) - 3.0).abs() < 1e-4);
        assert!((recency_multiplier(14.0 * day) - 2.0).abs() < 1e-4);
        assert!((recency_multiplier(28.0 * day) - 1.5).abs() < 1e-4);
        assert!((recency_multiplier(-day) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn recent_session_is_not_crowded_out_by_older_sessions_repeating_the_term() {
        let dir = tempfile::tempdir().unwrap();
        let index = SessionIndex::open_or_create(dir.path()).unwrap();
        let mut writer = index.writer().unwrap();

        // Old sessions whose many messages all match strongly fill a BM25-only
        // candidate pool (limit * 20 docs) before the recent session gets in.
        let heavy = ["deploy deploy deploy"; 20];
        for i in 0..3 {
            index
                .index_session(&mut writer, &session(&format!("old-{i}"), 365, &heavy))
                .unwrap();
        }
        index
            .index_session(
                &mut writer,
                &session("recent", 0, &["let's deploy the fix to staging now"]),
            )
            .unwrap();
        writer.commit().unwrap();
        index.reload().unwrap();

        let results = index.search("deploy", 2).unwrap();

        assert_eq!(results[0].session.id, "recent");
    }
}
