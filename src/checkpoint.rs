//! Cooperative restart: release only after the worker actually exits.
use crate::{profiles::Measurement, Job, Submit};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

pub const CHECKPOINT_EXIT_CODE: i32 = 75;
pub const MAX_ATTEMPTS: u32 = 32;
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointReport {
    pub schema_version: u32,
    pub measurement: Measurement,
    pub attempt: u32,
    pub step: u64,
    pub checkpoint_path: PathBuf,
    pub resume_vram_mib: u64,
}
impl CheckpointReport {
    pub fn validate(&self, job: &Job, expected: Measurement) -> Result<(), String> {
        if !job.request.cooperative {
            return Err("job did not opt in to cooperative restart".into());
        }
        if self.schema_version != 1 || self.measurement != expected || self.attempt != job.attempts
        {
            return Err("checkpoint version/backend/attempt mismatch".into());
        }
        if job.attempts >= MAX_ATTEMPTS {
            return Err("checkpoint attempt limit reached".into());
        }
        if self.resume_vram_mib == 0 {
            return Err("positive resume VRAM is required".into());
        }
        if job
            .checkpoint
            .as_ref()
            .is_some_and(|previous| self.step <= previous.step)
        {
            return Err("checkpoint step did not advance".into());
        }
        if !self.checkpoint_path.is_absolute() || self.checkpoint_path.as_os_str().len() > 4096 {
            return Err("checkpoint path must be absolute and at most 4096 bytes".into());
        }
        let meta = fs::symlink_metadata(&self.checkpoint_path)
            .map_err(|_| "checkpoint file missing or unreadable".to_string())?;
        if !meta.is_file() {
            return Err("checkpoint must be a regular file".into());
        }
        File::open(&self.checkpoint_path).map_err(|_| "checkpoint file unreadable".to_string())?;
        Ok(())
    }
}
pub fn read_checkpoint_report(path: &Path) -> Result<CheckpointReport, String> {
    let meta = fs::symlink_metadata(path)
        .map_err(|_| "checkpoint report missing or unreadable".to_string())?;
    if !meta.is_file() {
        return Err("checkpoint report must be a regular file".into());
    }
    let mut bytes = vec![];
    File::open(path)
        .map_err(|_| "checkpoint report unreadable")?
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(|_| "checkpoint report read failed")?;
    if bytes.len() > 65536 {
        return Err("checkpoint report exceeds 64 KiB".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "invalid checkpoint report JSON".into())
}
pub fn scheduling_request(job: &Job) -> Submit {
    let mut request = job.request.clone();
    if job.state == "waiting_resume" {
        if let Some(checkpoint) = &job.checkpoint {
            request.vram_mib = checkpoint.resume_vram_mib;
        }
    }
    request
}
pub fn accept_handoff(job: &mut Job, report: CheckpointReport, ticket: u64) {
    // Called only after try_wait confirmed exit code 75 and validation passed.
    job.checkpoint = Some(report);
    job.state = "waiting_resume".into();
    job.reservation_mib = 0;
    job.pid = None;
    job.queue_ticket = ticket;
    job.message = "checkpoint accepted; worker exited; waiting to reacquire the same GPU".into();
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{available_mib, queue_order, Gpu};
    fn job(id: u64) -> Job {
        Job {
            id,
            request: Submit {
                command: vec!["true".into()],
                vram_mib: 7000,
                priority: 0,
                profile: None,
                cooperative: true,
            },
            state: "running".into(),
            gpu: Some("GPU-a".into()),
            pid: Some(1),
            exit_code: None,
            message: String::new(),
            reservation_mib: 7000,
            memory_report: None,
            report_error: None,
            attempts: 1,
            queue_ticket: id,
            checkpoint: None,
        }
    }
    fn path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "mlus-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
    fn report(path: PathBuf) -> CheckpointReport {
        CheckpointReport {
            schema_version: 1,
            measurement: Measurement::Simulation,
            attempt: 1,
            step: 50,
            checkpoint_path: path,
            resume_vram_mib: 6000,
        }
    }
    #[test]
    fn contract_checks() {
        let path = path();
        fs::write(&path, b"checkpoint").unwrap();
        let mut j = job(1);
        let r = report(path.clone());
        assert!(r.validate(&j, Measurement::Simulation).is_ok());
        assert!(r.validate(&j, Measurement::Cuda).is_err());
        j.request.cooperative = false;
        assert!(r.validate(&j, Measurement::Simulation).is_err());
        j.request.cooperative = true;
        j.attempts = 2;
        assert!(r.validate(&j, Measurement::Simulation).is_err());
        j.attempts = 1;
        j.checkpoint = Some(r.clone());
        assert!(r.validate(&j, Measurement::Simulation).is_err());
        j.checkpoint = None;
        let mut invalid = r.clone();
        invalid.resume_vram_mib = 0;
        assert!(invalid.validate(&j, Measurement::Simulation).is_err());
        invalid = r.clone();
        invalid.checkpoint_path = PathBuf::from("relative");
        assert!(invalid.validate(&j, Measurement::Simulation).is_err());
        invalid = r.clone();
        invalid.attempt = MAX_ATTEMPTS;
        j.attempts = MAX_ATTEMPTS;
        assert!(invalid.validate(&j, Measurement::Simulation).is_err());
        fs::remove_file(path).unwrap();
        assert!(r.validate(&job(1), Measurement::Simulation).is_err());
    }
    #[test]
    fn release_and_fifo_requeue() {
        let mut resumed = job(1);
        let gpu = Gpu {
            uuid: "GPU-a".into(),
            total_mib: 8192,
            used_mib: 0,
            utilization: 0,
        };
        assert_eq!(
            available_mib(&gpu, std::slice::from_ref(&resumed), 512),
            680
        );
        accept_handoff(&mut resumed, report(PathBuf::from("/example")), 3);
        assert_eq!(
            available_mib(&gpu, std::slice::from_ref(&resumed), 512),
            7680
        );
        let mut other = job(2);
        other.state = "queued".into();
        other.gpu = None;
        assert_eq!(queue_order(&[resumed.clone(), other]), vec![1, 0]);
        assert_eq!(scheduling_request(&resumed).vram_mib, 6000);
        assert_eq!(resumed.request.vram_mib, 7000);
    }
}
