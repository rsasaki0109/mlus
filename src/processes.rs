//! Linux userspace lifecycle management for cooperative process groups.
//! Not a cgroup or security boundary. Never signal a PID/PGID restored from disk.
use serde::Serialize;
use std::{
    collections::HashMap,
    io,
    os::unix::fs::MetadataExt,
    process::{Child, ExitStatus},
    thread,
    time::{Duration, Instant},
};
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(3);

pub fn enable_subreaper() -> Result<(), String> {
    // SAFETY: prctl's subreaper option takes integer arguments, no pointers.
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
        return Err(format!(
            "cannot enable Linux child subreaper: {}",
            io::Error::last_os_error()
        ));
    }
    // Some kernels omit /proc/<pid>/task/<pid>/children (CONFIG_CHECKPOINT_RESTORE).
    // Verify the procfs parent-relation fallback before admitting any workers.
    direct_children()?;
    Ok(())
}

fn direct_children() -> Result<Vec<libc::pid_t>, String> {
    let pid = std::process::id();
    match std::fs::read_to_string(format!("/proc/{pid}/task/{pid}/children")) {
        Ok(text) => {
            return text
                .split_whitespace()
                .map(|p| p.parse().map_err(|_| "invalid procfs child PID".into()))
                .collect()
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("cannot inspect procfs children: {e}")),
    }
    // Linux kernels without the children file still expose PPid in /proc/PID/stat.
    // Validate our own stat first; then inspect only same-UID processes.
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .map_err(|e| format!("procfs process inspection unavailable: {e}"))?;
    // SAFETY: geteuid has no arguments and does not modify process state.
    let uid = unsafe { libc::geteuid() };
    let mut children = vec![];
    for entry in std::fs::read_dir("/proc").map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        let meta = match entry.metadata() {
            Ok(meta) => meta,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot inspect process owner: {e}")),
        };
        if meta.uid() != uid {
            continue;
        }
        let text = match std::fs::read_to_string(entry.path().join("stat")) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot inspect same-UID process: {e}")),
        };
        // comm may contain spaces and parentheses; fields after its final ')' are stable.
        let fields = text
            .rsplit_once(')')
            .ok_or("invalid procfs process stat")?
            .1;
        let parent = fields
            .split_whitespace()
            .nth(1)
            .ok_or("missing procfs parent PID")?
            .parse::<u32>()
            .map_err(|_| "invalid procfs parent PID")?;
        if parent == pid {
            children.push(id);
        }
    }
    children.sort_unstable();
    Ok(children)
}

pub struct Worker {
    child: Child,
    pgid: libc::pid_t,
    status: Option<ExitStatus>,
    cleanup_started: Option<Instant>,
}
#[derive(Serialize)]
pub struct WorkerStatus {
    pub job_id: u64,
    pub pgid: libc::pid_t,
    pub leader_exited: bool,
    pub cleanup_pending: bool,
}
impl Worker {
    /// Child must have been spawned with CommandExt::process_group(0).
    pub fn new(child: Child) -> Self {
        Self {
            pgid: child.id() as libc::pid_t,
            child,
            status: None,
            cleanup_started: None,
        }
    }
    fn start_cleanup(&mut self) -> Result<(), String> {
        if self.cleanup_started.is_none() {
            // The unreaped leader pins its PID while sending to the newly created group.
            // SAFETY: negative pgid targets only this live job's process group.
            if unsafe { libc::kill(-self.pgid, libc::SIGKILL) } != 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(format!("process group termination failed: {error}"));
                }
            }
            self.cleanup_started = Some(Instant::now());
        }
        Ok(())
    }
    pub fn poll(&mut self) -> Result<Option<ExitStatus>, String> {
        if self.status.is_none() {
            // Observe without reaping; keep the leader PID until the group is signalled.
            // SAFETY: info points to an initialized siginfo_t; this is our Child's PID.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let result = unsafe {
                libc::waitid(
                    libc::P_PID,
                    self.child.id(),
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if result != 0 {
                return Err(format!(
                    "worker waitid failed: {}",
                    io::Error::last_os_error()
                ));
            }
            // SAFETY: waitid initialized the signal information above.
            if unsafe { info.si_pid() } == 0 {
                return Ok(None);
            }
            self.start_cleanup()?;
            self.status = Some(
                self.child
                    .wait()
                    .map_err(|e| format!("worker reap failed: {e}"))?,
            );
        }
        reap_group(self.pgid)?;
        if !group_exists(self.pgid)? {
            return Ok(self.status);
        }
        if self
            .cleanup_started
            .is_some_and(|t| t.elapsed() >= CLEANUP_TIMEOUT)
        {
            return Err("process group cleanup timed out; reservation retained".into());
        }
        Ok(None)
    }
}

fn group_exists(pgid: libc::pid_t) -> Result<bool, String> {
    // SAFETY: signal 0 queries group existence without signalling its members.
    if unsafe { libc::kill(-pgid, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(format!("process group query failed: {error}")),
    }
}
fn reap_group(pgid: libc::pid_t) -> Result<(), String> {
    loop {
        let mut status = 0;
        // SAFETY: this runs only after Child::wait has reaped the group's leader.
        // Negative pgid limits reaping to adopted descendants in that job group.
        let result = unsafe { libc::waitpid(-pgid, &mut status, libc::WNOHANG) };
        if result > 0 {
            continue;
        }
        if result == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ECHILD) => return Ok(()),
            Some(libc::EINTR) => continue,
            _ => return Err(format!("descendant reap failed: {error}")),
        }
    }
}

pub struct ManagedWorkers(pub HashMap<u64, Worker>);
impl ManagedWorkers {
    pub fn status(&self) -> Vec<WorkerStatus> {
        let mut states: Vec<_> = self
            .0
            .iter()
            .map(|(&job_id, w)| WorkerStatus {
                job_id,
                pgid: w.pgid,
                leader_exited: w.status.is_some(),
                cleanup_pending: w.cleanup_started.is_some(),
            })
            .collect();
        states.sort_by_key(|s| s.job_id);
        states
    }
    /// Inspect currently adopted children outside known groups; do not signal them.
    pub fn escaped_children(&self) -> Result<Vec<libc::pid_t>, String> {
        let mut escaped = vec![];
        for child in direct_children()? {
            if self.0.values().any(|w| w.child.id() == child as u32) {
                continue;
            }
            // SAFETY: getpgid reads a current child identity; no signal is sent.
            let group = unsafe { libc::getpgid(child) };
            if group < 0 {
                if io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
                    continue;
                }
                return Err("cannot inspect adopted child's process group".into());
            }
            if self.0.values().any(|w| w.pgid == group) {
                continue;
            }
            // Reap an exited adopted child. Live children require explicit operator cleanup.
            let mut status = 0;
            // SAFETY: child is a direct child listed by this daemon's /proc children file.
            let reaped = unsafe { libc::waitpid(child, &mut status, libc::WNOHANG) };
            if reaped == 0 {
                escaped.push(child);
            } else if reaped < 0 && io::Error::last_os_error().raw_os_error() != Some(libc::ECHILD)
            {
                return Err("cannot reap adopted child".into());
            }
        }
        Ok(escaped)
    }
    pub fn shutdown(&mut self) -> Vec<(u64, Result<ExitStatus, String>)> {
        // Signal all groups before waiting; one shared deadline bounds daemon cleanup.
        let mut results = vec![];
        for (&id, worker) in &mut self.0 {
            if let Err(error) = worker.start_cleanup() {
                results.push((id, Err(error)));
            }
        }
        for (id, _) in &results {
            self.0.remove(id);
        }
        let deadline = Instant::now() + CLEANUP_TIMEOUT;
        while !self.0.is_empty() {
            let mut completed = vec![];
            for (&id, worker) in &mut self.0 {
                match worker.poll() {
                    Ok(Some(status)) => {
                        results.push((id, Ok(status)));
                        completed.push(id);
                    }
                    Err(error) => {
                        results.push((id, Err(error)));
                        completed.push(id);
                    }
                    Ok(None) => {}
                }
            }
            for id in completed {
                self.0.remove(&id);
            }
            if Instant::now() >= deadline {
                for (id, _) in self.0.drain() {
                    results.push((id, Err("shutdown group cleanup timed out".into())));
                }
                break;
            }
            if !self.0.is_empty() {
                thread::sleep(Duration::from_millis(10));
            }
        }
        results
    }
}
impl Drop for ManagedWorkers {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
