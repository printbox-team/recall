use crate::session::{Message, Role, Session, SessionSource};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use super::{join_consecutive_messages, SessionParser};

#[derive(Debug, Deserialize)]
struct CodexLine {
    #[serde(rename = "type")]
    entry_type: String,
    timestamp: Option<String>,
    payload: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SessionMeta {
    id: String,
    cwd: Option<String>,
    git: Option<GitInfo>,
    source: Option<serde_json::Value>,
}

impl SessionMeta {
    fn is_subagent(&self) -> bool {
        self.source.as_ref().is_some_and(|source| {
            source.as_str() == Some("subagent") || source.get("subagent").is_some()
        })
    }
}

#[derive(Debug, Deserialize)]
struct GitInfo {
    branch: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseItem {
    role: Option<String>,
    content: Option<Vec<ContentBlock>>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ThreadNameEntry {
    id: String,
    thread_name: String,
}

/// Thread names read from one `session_index.jsonl`, kept until the file changes.
struct ThreadNames {
    index_path: PathBuf,
    stamp: (SystemTime, u64),
    names: HashMap<String, String>,
}

static THREAD_NAMES: Mutex<Option<ThreadNames>> = Mutex::new(None);

/// The name Codex shows for a thread. Codex keeps names outside the rollout, in
/// `<codex home>/session_index.jsonl`, appending a line each time a thread is named
/// or renamed, so the last line for an id holds its current name.
pub fn thread_name(rollout_path: &Path, thread_id: &str) -> Option<String> {
    let index_path = rollout_path
        .ancestors()
        .find(|dir| dir.ends_with("sessions"))?
        .parent()?
        .join("session_index.jsonl");
    let metadata = std::fs::metadata(&index_path).ok()?;
    let stamp = (metadata.modified().ok()?, metadata.len());

    let mut cache = THREAD_NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let fresh = cache
        .as_ref()
        .is_some_and(|c| c.index_path == index_path && c.stamp == stamp);
    if !fresh {
        *cache = Some(ThreadNames {
            names: read_thread_names(&index_path),
            index_path,
            stamp,
        });
    }
    cache.as_ref()?.names.get(thread_id).cloned()
}

fn read_thread_names(index_path: &Path) -> HashMap<String, String> {
    let mut names = HashMap::new();
    let Ok(file) = File::open(index_path) else {
        return names;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(entry) = serde_json::from_str::<ThreadNameEntry>(&line) else {
            continue;
        };
        let name = entry.thread_name.trim();
        if name.is_empty() {
            names.remove(&entry.id);
        } else {
            names.insert(entry.id, name.to_string());
        }
    }
    names
}

pub struct CodexParser;

impl SessionParser for CodexParser {
    fn can_parse(path: &Path) -> bool {
        // Codex sessions are in ~/.codex/sessions/ and start with "rollout-"
        path.to_str()
            .map(|s| s.contains(".codex/sessions"))
            .unwrap_or(false)
    }

    fn parse_file(path: &Path) -> Result<Session> {
        let file = File::open(path).context("Failed to open file")?;
        let reader = BufReader::with_capacity(64 * 1024, file);

        let mut session_id: Option<String> = None;
        let mut cwd: Option<String> = None;
        let mut git_branch: Option<String> = None;
        let mut latest_timestamp: Option<DateTime<Utc>> = None;
        let mut messages: Vec<Message> = Vec::new();

        for line in reader.lines() {
            let line = line.context("Failed to read line")?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: CodexLine = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Parse timestamp from entry
            let timestamp = entry
                .timestamp
                .as_ref()
                .and_then(|ts| DateTime::parse_from_rfc3339(ts).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            match entry.entry_type.as_str() {
                "session_meta" => {
                    if let Some(payload) = &entry.payload {
                        if let Ok(meta) = serde_json::from_value::<SessionMeta>(payload.clone()) {
                            let is_subagent = meta.is_subagent();
                            // Only set if not already set (first session_meta wins)
                            if session_id.is_none() {
                                session_id = Some(meta.id);
                            }
                            if cwd.is_none() {
                                cwd = meta.cwd;
                            }
                            if git_branch.is_none() {
                                git_branch = meta.git.and_then(|g| g.branch);
                            }
                            // The indexer skips sessions with no messages.
                            if is_subagent {
                                messages.clear();
                                break;
                            }
                        }
                    }
                }
                "response_item" => {
                    if let Some(payload) = &entry.payload {
                        if let Ok(item) = serde_json::from_value::<ResponseItem>(payload.clone()) {
                            let role = match item.role.as_deref() {
                                Some("user") => Role::User,
                                Some("assistant") => Role::Assistant,
                                Some("developer" | "system") => continue,
                                _ => {
                                    // Try to infer role from content type
                                    if let Some(content) = &item.content {
                                        if content.iter().any(|c| c.content_type == "input_text") {
                                            Role::User
                                        } else if content
                                            .iter()
                                            .any(|c| c.content_type == "output_text")
                                        {
                                            Role::Assistant
                                        } else {
                                            continue;
                                        }
                                    } else {
                                        continue;
                                    }
                                }
                            };

                            let content = extract_codex_content(&item);
                            if !content.is_empty() {
                                messages.push(Message {
                                    role,
                                    content,
                                    timestamp,
                                });

                                // Update latest timestamp
                                if latest_timestamp.is_none()
                                    || timestamp > latest_timestamp.unwrap()
                                {
                                    latest_timestamp = Some(timestamp);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Fall back to filename for session ID if not found
        let session_id = session_id.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        let title = thread_name(path, &session_id);

        Ok(Session {
            id: session_id,
            source: SessionSource::CodexCli,
            file_path: path.to_path_buf(),
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            git_branch,
            timestamp: latest_timestamp.unwrap_or_else(Utc::now),
            title,
            messages: join_consecutive_messages(messages),
        })
    }
}

/// Extract text content from a Codex response item.
/// Filters out CLI-injected blocks (AGENTS.md instructions, environment_context).
fn extract_codex_content(item: &ResponseItem) -> String {
    let Some(content) = &item.content else {
        return String::new();
    };

    let mut texts = Vec::new();
    for block in content {
        // Extract from input_text or output_text blocks
        if (block.content_type == "input_text" || block.content_type == "output_text")
            && block.text.is_some()
        {
            if let Some(text) = &block.text {
                // Skip CLI-injected blocks (must have both open and close tags)
                let trimmed = text.trim();
                if trimmed.starts_with("# AGENTS.md instructions for ")
                    && trimmed.ends_with("</INSTRUCTIONS>")
                {
                    continue;
                }
                if trimmed.starts_with("<environment_context>")
                    && trimmed.ends_with("</environment_context>")
                {
                    continue;
                }
                // Legacy user_instructions format
                if trimmed.starts_with("<user_instructions>")
                    && trimmed.ends_with("</user_instructions>")
                {
                    continue;
                }
                texts.push(text.clone());
            }
        }
    }
    texts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_name_takes_last_entry_for_id() {
        let codex_home = tempfile::TempDir::new().unwrap();
        let rollout = codex_home.path().join("sessions/2026/11/26/rollout.jsonl");
        std::fs::write(
            codex_home.path().join("session_index.jsonl"),
            [
                r#"{"id":"a","thread_name":"first","updated_at":"2026-11-26T10:00:00Z"}"#,
                r#"{"id":"b","thread_name":"other","updated_at":"2026-11-26T10:00:01Z"}"#,
                r#"{"id":"a","thread_name":" renamed ","updated_at":"2026-11-26T10:00:02Z"}"#,
                r#"{"id":"b","thread_name":"","updated_at":"2026-11-26T10:00:03Z"}"#,
            ]
            .join("\n"),
        )
        .unwrap();

        assert_eq!(thread_name(&rollout, "a").as_deref(), Some("renamed"));
        assert_eq!(thread_name(&rollout, "b"), None);
        assert_eq!(thread_name(&rollout, "c"), None);
    }

    #[test]
    fn test_extract_codex_content() {
        let item = ResponseItem {
            role: Some("user".to_string()),
            content: Some(vec![ContentBlock {
                content_type: "input_text".to_string(),
                text: Some("Hello Codex".to_string()),
            }]),
        };
        assert_eq!(extract_codex_content(&item), "Hello Codex");
    }

    #[test]
    fn test_extract_codex_content_filters_injected_blocks() {
        let item = ResponseItem {
            role: Some("user".to_string()),
            content: Some(vec![
                ContentBlock {
                    content_type: "input_text".to_string(),
                    text: Some("# AGENTS.md instructions for /some/path\n\n<INSTRUCTIONS>\nsome instructions\n</INSTRUCTIONS>".to_string()),
                },
                ContentBlock {
                    content_type: "input_text".to_string(),
                    text: Some("<environment_context>\n  <cwd>/some/path</cwd>\n</environment_context>".to_string()),
                },
                ContentBlock {
                    content_type: "input_text".to_string(),
                    text: Some("actual user message".to_string()),
                },
            ]),
        };
        assert_eq!(extract_codex_content(&item), "actual user message");
    }

    #[test]
    fn test_extract_codex_content_keeps_partial_tags() {
        // User might ask about these tags - don't filter if not properly closed
        let item = ResponseItem {
            role: Some("user".to_string()),
            content: Some(vec![ContentBlock {
                content_type: "input_text".to_string(),
                text: Some("<environment_context> what is this?".to_string()),
            }]),
        };
        assert_eq!(
            extract_codex_content(&item),
            "<environment_context> what is this?"
        );
    }
}
