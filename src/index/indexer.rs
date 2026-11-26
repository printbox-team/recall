//! Shared indexing logic for both background (TUI) and synchronous (CLI) modes

use super::state::IndexState;
use super::SessionIndex;
use crate::parser;
use anyhow::Result;
use std::path::PathBuf;
use tantivy::IndexWriter;

/// Progress information during indexing
pub struct IndexProgress {
    pub indexed: usize,
    pub total: usize,
}

/// Callback for reporting indexing progress
pub type ProgressCallback = Box<dyn FnMut(IndexProgress) + Send>;

/// Callback for notifying that the index should be reloaded
pub type ReloadCallback = Box<dyn FnMut() + Send>;

/// Discovers session files and sorts them by modification time (most recent first)
pub fn discover_and_sort_files() -> Vec<PathBuf> {
    let mut files = parser::discover_session_files();
    files.sort_by(|a, b| {
        let mtime_a = std::fs::metadata(a)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let mtime_b = std::fs::metadata(b)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        mtime_b.cmp(&mtime_a) // Descending (most recent first)
    });
    files
}

/// Index a batch of files, calling progress callbacks as work proceeds.
///
/// - `on_progress`: Called every 50 files with current progress
/// - `on_reload`: Called every 200 files after a commit (for incremental updates)
///
/// Returns the number of files successfully indexed.
pub fn index_files(
    index: &SessionIndex,
    writer: &mut IndexWriter,
    state: &mut IndexState,
    files: &[PathBuf],
    mut on_progress: Option<ProgressCallback>,
    mut on_reload: Option<ReloadCallback>,
) -> Result<usize> {
    let total = files.len();
    let mut indexed = 0;

    for (i, file_path) in files.iter().enumerate() {
        // Delete existing documents for this file (in case of update)
        index.delete_session(writer, file_path);

        // Parse and index
        match parser::parse_session_file(file_path) {
            Ok(session) => {
                if !session.messages.is_empty() {
                    let _ = index.index_session(writer, &session);
                }
                // Mark as indexed even if empty (so we don't reprocess it)
                state.mark_indexed(file_path, &session);
                indexed += 1;
            }
            Err(_) => {
                // Skip failed files (they might be incomplete/corrupted)
                // Don't mark as indexed so we retry next time
            }
        }

        // Progress update every 50 files or at the end
        if (i + 1) % 50 == 0 || i + 1 == total {
            if let Some(ref mut callback) = on_progress {
                callback(IndexProgress {
                    indexed: i + 1,
                    total,
                });
            }
        }

        // Commit and notify for reload every 200 files
        if (i + 1) % 200 == 0 {
            writer.commit()?;
            if let Some(ref mut callback) = on_reload {
                callback();
            }
        }
    }

    // Final commit
    writer.commit()?;

    Ok(indexed)
}
