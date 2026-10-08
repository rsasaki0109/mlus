//! Durable local snapshots. The caller must hold ProfileStore's state-directory lock.
use crate::{checkpoint::MAX_ATTEMPTS, profiles::Measurement, Job};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
const MAX_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_JOBS: usize = 10_000;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema_version: u32,
    measurement: Measurement,
    working_directory: PathBuf,
    log_directory: PathBuf,
    recovery_required: bool,
    jobs: Vec<Job>,
}
pub struct Journal {
    root: PathBuf,
    path: PathBuf,
    measurement: Measurement,
    pub working_directory: PathBuf,
    pub log_directory: PathBuf,
    pub recovery_required: bool,
}
impl Journal {
    pub fn open(root: &Path, measurement: Measurement) -> Result<(Self, Vec<Job>), String> {
        let root = fs::canonicalize(root).map_err(|e| e.to_string())?;
        let label = if measurement == Measurement::Simulation {
            "mock"
        } else {
            "cuda"
        };
        let path = root.join(format!("{label}-jobs.json"));
        let doc = match fs::symlink_metadata(&path) {
            Ok(meta) => {
                if !meta.is_file() {
                    return Err("job journal must be a regular file".into());
                }
                let mut bytes = Vec::new();
                File::open(&path)
                    .map_err(|e| e.to_string())?
                    .take(MAX_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                if bytes.len() as u64 > MAX_BYTES {
                    return Err("job journal exceeds 32 MiB".into());
                }
                let doc: Document = serde_json::from_slice(&bytes).map_err(|_| {
                    "invalid job journal; preserve and inspect it before recovery".to_string()
                })?;
                if doc.schema_version != 1 || doc.measurement != measurement {
                    return Err("job journal schema/backend mismatch".into());
                }
                validate_jobs(&doc.jobs, measurement)?;
                if !doc.working_directory.is_absolute() || !doc.working_directory.is_dir() {
                    return Err("saved working directory unavailable".into());
                }
                let logs = fs::canonicalize(&doc.log_directory)
                    .map_err(|_| "saved log directory unavailable")?;
                if !doc.log_directory.is_absolute()
                    || !logs.starts_with(root.join("runs"))
                    || !logs.is_dir()
                {
                    return Err("invalid saved log directory".into());
                }
                doc
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let runs = root.join("runs");
                if !runs.exists() {
                    fs::DirBuilder::new()
                        .mode(0o700)
                        .create(&runs)
                        .map_err(|e| e.to_string())?;
                }
                if fs::canonicalize(&runs).map_err(|e| e.to_string())? != runs {
                    return Err("runs directory must not be a symlink".into());
                }
                let logs = runs.join(format!("{label}-{}-{}", std::process::id(), stamp()?));
                fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&logs)
                    .map_err(|e| e.to_string())?;
                File::open(&runs)
                    .and_then(|f| f.sync_all())
                    .map_err(|e| e.to_string())?;
                Document {
                    schema_version: 1,
                    measurement,
                    working_directory: std::env::current_dir().map_err(|e| e.to_string())?,
                    log_directory: logs,
                    recovery_required: false,
                    jobs: vec![],
                }
            }
            Err(e) => return Err(format!("cannot read job journal: {e}")),
        };
        let mut journal = Self {
            root,
            path,
            measurement,
            working_directory: doc.working_directory,
            log_directory: doc.log_directory,
            recovery_required: doc.recovery_required,
        };
        let mut jobs = doc.jobs;
        for job in &mut jobs {
            if matches!(job.state.as_str(), "starting" | "running") {
                job.state = "interrupted".into();
                job.message = "daemon interrupted; historical PID is untrusted; inspect remaining workers before recover".into();
                journal.recovery_required = true;
            }
        }
        journal.save(&jobs)?;
        Ok((journal, jobs))
    }
    pub fn save(&self, jobs: &[Job]) -> Result<(), String> {
        let doc = Document {
            schema_version: 1,
            measurement: self.measurement,
            working_directory: self.working_directory.clone(),
            log_directory: self.log_directory.clone(),
            recovery_required: self.recovery_required,
            jobs: jobs.to_vec(),
        };
        let bytes = serde_json::to_vec(&doc).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err("job journal exceeds 32 MiB".into());
        }
        let temp = self
            .root
            .join(format!("jobs-{}-{}.tmp", std::process::id(), stamp()?));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp)
                .map_err(|e| e.to_string())?;
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .map_err(|e| e.to_string())?;
            fs::rename(&temp, &self.path).map_err(|e| e.to_string())?;
            File::open(&self.root)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    }
}
fn stamp() -> Result<u128, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .map_err(|e| e.to_string())
}
fn validate_jobs(jobs: &[Job], measurement: Measurement) -> Result<(), String> {
    if jobs.len() > MAX_JOBS {
        return Err("job history exceeds 10000 entries".into());
    }
    let mut tickets = std::collections::BTreeSet::new();
    for (index, j) in jobs.iter().enumerate() {
        j.request.validate()?;
        if j.id != index as u64 + 1
            || j.queue_ticket == 0
            || j.queue_ticket > (MAX_JOBS as u64 * (u64::from(MAX_ATTEMPTS) + 1))
            || !tickets.insert(j.queue_ticket)
            || j.attempts > MAX_ATTEMPTS
        {
            return Err("invalid saved job identity/attempt/ticket".into());
        }
        if !matches!(
            j.state.as_str(),
            "queued"
                | "starting"
                | "running"
                | "waiting_resume"
                | "succeeded"
                | "failed"
                | "rejected"
                | "cancelled"
                | "interrupted"
        ) {
            return Err("invalid saved job state".into());
        }
        if j.state == "queued"
            && (j.attempts != 0
                || j.pid.is_some()
                || j.gpu.is_some()
                || j.reservation_mib != 0
                || j.checkpoint.is_some())
        {
            return Err("invalid queued job".into());
        }
        if matches!(j.state.as_str(), "starting" | "running")
            && (j.attempts == 0
                || j.reservation_mib == 0
                || j.gpu.as_ref().is_none_or(|g| g.is_empty())
                || (j.state == "running" && j.pid.is_none()))
        {
            return Err("invalid active job".into());
        }
        if j.state == "waiting_resume" {
            let r = j.checkpoint.as_ref().ok_or("missing saved checkpoint")?;
            if !j.request.cooperative
                || j.pid.is_some()
                || j.reservation_mib != 0
                || j.gpu.as_ref().is_none_or(|g| g.is_empty())
                || r.schema_version != 1
                || r.measurement != measurement
                || r.attempt != j.attempts
                || j.attempts == 0
                || j.attempts >= MAX_ATTEMPTS
                || r.step == 0
                || r.resume_vram_mib == 0
                || !r.checkpoint_path.is_absolute()
            {
                return Err("invalid saved checkpoint wait".into());
            }
        }
    }
    Ok(())
}
