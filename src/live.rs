//! Detect which Claude Code sessions currently have a live `claude` process.
//!
//! Claude Code writes `~/.claude/sessions/<pid>.json` for every running CLI
//! instance, containing `sessionId`, `cwd`, `status` (`idle` / `busy`), and
//! related metadata. We read this directory, verify each PID is still alive
//! via `kill(pid, 0)`, and return a map of `session_id -> LiveStatus`.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveStatus {
    Idle,
    Busy,
}

pub type LiveMap = HashMap<String, LiveStatus>;

#[derive(Debug, Deserialize)]
struct SessionFile {
    pid: i32,
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(default)]
    status: Option<String>,
}

/// Scan `~/.claude/sessions/` and return live sessions. Errors are swallowed —
/// a missing directory or an unparseable file just means "no live sessions
/// detected for that entry".
pub fn scan_live_sessions() -> LiveMap {
    match sessions_dir() {
        Some(dir) => scan_dir(&dir, is_pid_alive),
        None => LiveMap::new(),
    }
}

fn scan_dir(dir: &std::path::Path, alive: impl Fn(i32) -> bool) -> LiveMap {
    let mut result = LiveMap::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return result;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<SessionFile>(&contents) else {
            continue;
        };
        if !alive(parsed.pid) {
            continue;
        }
        let status = match parsed.status.as_deref() {
            Some("busy") => LiveStatus::Busy,
            _ => LiveStatus::Idle,
        };
        result.insert(parsed.session_id, status);
    }
    result
}

fn sessions_dir() -> Option<PathBuf> {
    if let Ok(h) = std::env::var("RECALL_HOME_OVERRIDE") {
        return Some(PathBuf::from(h).join(".claude").join("sessions"));
    }
    dirs::home_dir().map(|h| h.join(".claude").join("sessions"))
}

#[cfg(unix)]
fn is_pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // kill(pid, 0) → 0 means alive; -1 with EPERM still means alive (we just
    // can't signal it); -1 with ESRCH means it's gone.
    unsafe {
        if libc::kill(pid, 0) == 0 {
            return true;
        }
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn is_pid_alive(_pid: i32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn write_session_file(dir: &std::path::Path, pid: i32, sid: &str, status: &str) {
        let body = serde_json::json!({
            "pid": pid,
            "sessionId": sid,
            "cwd": "/tmp",
            "status": status,
        });
        std::fs::write(dir.join(format!("{}.json", pid)), body.to_string()).unwrap();
    }

    #[test]
    fn idle_and_busy_status_parsed() {
        let dir = tempfile::tempdir().unwrap();
        write_session_file(dir.path(), 1, "sid-idle", "idle");
        write_session_file(dir.path(), 2, "sid-busy", "busy");

        let alive = |_pid: i32| true;
        let map = scan_dir(dir.path(), alive);

        assert_eq!(map.get("sid-idle"), Some(&LiveStatus::Idle));
        assert_eq!(map.get("sid-busy"), Some(&LiveStatus::Busy));
    }

    #[test]
    fn dead_pids_excluded() {
        let dir = tempfile::tempdir().unwrap();
        write_session_file(dir.path(), 100, "sid-alive", "idle");
        write_session_file(dir.path(), 200, "sid-dead", "idle");

        let alive_pids: HashSet<i32> = [100].into_iter().collect();
        let map = scan_dir(dir.path(), |pid| alive_pids.contains(&pid));

        assert!(map.contains_key("sid-alive"));
        assert!(!map.contains_key("sid-dead"));
    }

    #[test]
    fn unknown_status_defaults_to_idle() {
        let dir = tempfile::tempdir().unwrap();
        write_session_file(dir.path(), 1, "sid", "something-weird");

        let map = scan_dir(dir.path(), |_| true);
        assert_eq!(map.get("sid"), Some(&LiveStatus::Idle));
    }

    #[test]
    fn malformed_files_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("1.json"), "not json").unwrap();
        write_session_file(dir.path(), 2, "sid-ok", "idle");
        // Non-.json file ignored
        std::fs::write(dir.path().join("3.txt"), "{}").unwrap();

        let map = scan_dir(dir.path(), |_| true);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("sid-ok"));
    }

    #[test]
    fn missing_dir_returns_empty() {
        let map = scan_dir(std::path::Path::new("/nonexistent/sessions"), |_| true);
        assert!(map.is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn is_pid_alive_self_true() {
        let my_pid = std::process::id() as i32;
        assert!(is_pid_alive(my_pid));
    }

    #[test]
    #[cfg(unix)]
    fn is_pid_alive_dead_false() {
        // Pick a high PID very unlikely to be assigned
        assert!(!is_pid_alive(2_000_000));
    }

    #[test]
    #[cfg(unix)]
    fn is_pid_alive_zero_false() {
        assert!(!is_pid_alive(0));
        assert!(!is_pid_alive(-1));
    }
}
