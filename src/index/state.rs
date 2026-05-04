use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Tracks which files have been indexed and their modification times
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexState {
    pub indexed_files: HashMap<PathBuf, FileState>,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub mtime: u64,
    pub size: u64,
}

impl IndexState {
    /// Bump whenever the Tantivy schema or parser output changes in a way that
    /// invalidates previously-indexed documents. `migrate_if_needed` wipes the
    /// index dir + state file when the on-disk version doesn't match.
    const CURRENT_VERSION: u32 = 2;

    /// Load state from disk or create new
    pub fn load(state_path: &Path) -> Result<Self> {
        if state_path.exists() {
            let content = std::fs::read_to_string(state_path)
                .context("Failed to read state file")?;
            let state: Self = serde_json::from_str(&content)
                .context("Failed to parse state file")?;
            Ok(state)
        } else {
            Ok(Self {
                indexed_files: HashMap::new(),
                version: Self::CURRENT_VERSION,
            })
        }
    }

    /// Wipe the index directory + state file if the persisted schema version
    /// doesn't match `CURRENT_VERSION`. Call before opening the Tantivy index
    /// so the new schema is used. No-op when versions match.
    pub fn migrate_if_needed(state_path: &Path, index_path: &Path) -> Result<()> {
        let needs_reset = if state_path.exists() {
            match std::fs::read_to_string(state_path)
                .ok()
                .and_then(|s| serde_json::from_str::<Self>(&s).ok())
            {
                Some(state) => state.version != Self::CURRENT_VERSION,
                None => true, // unreadable state file -> reset
            }
        } else {
            // No state but a Tantivy index exists -> almost certainly an old
            // build. Wipe it so the fresh schema is used.
            index_path.join("meta.json").exists()
        };

        if needs_reset {
            if index_path.exists() {
                std::fs::remove_dir_all(index_path)
                    .context("Failed to remove old index directory")?;
            }
            if state_path.exists() {
                std::fs::remove_file(state_path)
                    .context("Failed to remove old state file")?;
            }
        }
        Ok(())
    }

    /// Save state to disk
    pub fn save(&self, state_path: &Path) -> Result<()> {
        if let Some(parent) = state_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize state")?;
        std::fs::write(state_path, content)
            .context("Failed to write state file")?;
        Ok(())
    }

    /// Check if a file needs reindexing
    pub fn needs_reindex(&self, path: &Path) -> bool {
        let Some(current_state) = get_file_state(path) else {
            return false; // File doesn't exist
        };

        match self.indexed_files.get(path) {
            Some(indexed) => {
                // Reindex if mtime or size changed
                indexed.mtime != current_state.mtime || indexed.size != current_state.size
            }
            None => true, // Not indexed yet
        }
    }

    /// Mark a file as indexed
    pub fn mark_indexed(&mut self, path: &Path) {
        if let Some(state) = get_file_state(path) {
            self.indexed_files.insert(path.to_path_buf(), state);
        }
    }

    /// Remove a file from the index state
    pub fn remove(&mut self, path: &Path) {
        self.indexed_files.remove(path);
    }
}

/// Get the current file state (mtime and size)
fn get_file_state(path: &Path) -> Option<FileState> {
    let metadata = std::fs::metadata(path).ok()?;
    let mtime = metadata
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let size = metadata.len();

    Some(FileState { mtime, size })
}
