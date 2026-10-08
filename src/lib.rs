pub mod checkpoint;
pub mod journal;
pub mod processes;
pub mod profiles;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gpu {
    pub uuid: String,
    pub total_mib: u64,
    pub used_mib: u64,
    pub utilization: u32,
}

#[derive(Debug, Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub gpu_uuid: String,
    pub used_mib: u64,
}

pub trait Backend {
    fn processes(&self) -> Result<Vec<GpuProcess>, String> {
        Ok(vec![])
    }
    fn sample(&self) -> Result<Vec<Gpu>, String>;
}
pub struct Nvidia;
impl Backend for Nvidia {
    fn processes(&self) -> Result<Vec<GpuProcess>, String> {
        let out = Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err("GPU process query failed".into());
        }
        parse_process_csv(&String::from_utf8(out.stdout).map_err(|e| e.to_string())?)
    }
    fn sample(&self) -> Result<Vec<Gpu>, String> {
        let out = Command::new("nvidia-smi")
            .args([
                "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| format!("nvidia-smi: {e}"))?;
        if !out.status.success() {
            return Err("nvidia-smi failed; check driver and permissions".into());
        }
        parse_gpu_csv(&String::from_utf8(out.stdout).map_err(|e| e.to_string())?)
    }
}
pub struct Mock(pub Vec<Gpu>);
impl Backend for Mock {
    fn sample(&self) -> Result<Vec<Gpu>, String> {
        Ok(self.0.clone())
    }
}
pub fn parse_gpu_csv(input: &str) -> Result<Vec<Gpu>, String> {
    let mut gpus = Vec::new();
    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 4 {
            return Err("invalid GPU CSV field count".into());
        }
        let total = fields[1].parse::<u64>().map_err(|_| "invalid total VRAM")?;
        let used = fields[2].parse::<u64>().map_err(|_| "invalid used VRAM")?;
        let utilization = fields[3]
            .parse::<u32>()
            .map_err(|_| "invalid utilization")?;
        if fields[0].is_empty() || used > total || utilization > 100 {
            return Err("invalid GPU sample".into());
        }
        gpus.push(Gpu {
            uuid: fields[0].into(),
            total_mib: total,
            used_mib: used,
            utilization,
        });
    }
    if gpus.is_empty() {
        return Err("no GPUs detected".into());
    }
    Ok(gpus)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Submit {
    pub command: Vec<String>,
    pub vram_mib: u64,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub cooperative: bool,
}
impl Submit {
    pub fn validate(&self) -> Result<(), String> {
        if self.vram_mib == 0 || self.command.is_empty() || self.command[0].is_empty() {
            return Err("positive VRAM and a command are required".into());
        }
        if self.command.len() > 256
            || self
                .command
                .iter()
                .any(|s| s.contains('\0') || s.len() > 8192)
        {
            return Err("invalid or excessive command arguments".into());
        }
        if self
            .profile
            .as_ref()
            .is_some_and(|p| p.is_empty() || p.len() > 128 || p.chars().any(char::is_control))
        {
            return Err("profile must be 1..128 bytes without control characters".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub id: u64,
    pub request: Submit,
    pub state: String,
    pub gpu: Option<String>,
    pub pid: Option<u32>,
    pub exit_code: Option<i32>,
    pub message: String,
    pub reservation_mib: u64,
    pub memory_report: Option<profiles::MemoryReport>,
    pub report_error: Option<String>,
    pub attempts: u32,
    pub queue_ticket: u64,
    pub checkpoint: Option<checkpoint::CheckpointReport>,
}
/// Reservations cover the full lifetime of a child. Subtracting them from
/// observed free memory intentionally double-counts managed allocations:
/// safety over packing until attributable telemetry is available.
pub fn available_mib(gpu: &Gpu, jobs: &[Job], margin: u64) -> u64 {
    let reserved = jobs
        .iter()
        .filter(|j| {
            matches!(j.state.as_str(), "running" | "starting")
                && j.gpu.as_deref() == Some(&gpu.uuid)
        })
        .fold(0u64, |n, j| n.saturating_add(j.reservation_mib));
    gpu.total_mib
        .saturating_sub(gpu.used_mib)
        .saturating_sub(margin)
        .saturating_sub(reserved)
}
pub fn choose_gpu(gpus: &[Gpu], jobs: &[Job], need: u64, margin: u64) -> Option<usize> {
    gpus.iter()
        .enumerate()
        .filter_map(|(index, gpu)| {
            let available = available_mib(gpu, jobs, margin);
            if need <= available {
                Some((index, available - need))
            } else {
                None
            }
        })
        .min_by_key(|(_, remaining)| *remaining)
        .map(|(index, _)| index)
}
pub fn queue_order(jobs: &[Job]) -> Vec<usize> {
    let mut indices: Vec<_> = jobs
        .iter()
        .enumerate()
        .filter(|(_, j)| j.state == "queued" || j.state == "waiting_resume")
        .map(|(i, _)| i)
        .collect();
    indices.sort_by_key(|&i| {
        (
            std::cmp::Reverse(jobs[i].request.priority),
            jobs[i].queue_ticket,
        )
    });
    indices
}
#[cfg(test)]
mod tests {
    use super::*;
    fn gpu(total: u64, used: u64) -> Gpu {
        Gpu {
            uuid: "GPU-a".into(),
            total_mib: total,
            used_mib: used,
            utilization: 0,
        }
    }
    fn job(id: u64, need: u64, priority: i32, state: &str) -> Job {
        Job {
            id,
            request: Submit {
                command: vec!["true".into()],
                vram_mib: need,
                priority,
                profile: None,
                cooperative: false,
            },
            state: state.into(),
            gpu: Some("GPU-a".into()),
            pid: None,
            exit_code: None,
            message: String::new(),
            reservation_mib: need,
            memory_report: None,
            report_error: None,
            attempts: 1,
            queue_ticket: id,
            checkpoint: None,
        }
    }
    #[test]
    fn reservation_prevents_overcommit() {
        let gs = vec![gpu(8192, 0)];
        let jobs = vec![job(1, 5000, 0, "running")];
        assert_eq!(choose_gpu(&gs, &jobs, 3000, 512), None);
        assert_eq!(choose_gpu(&gs, &jobs, 2680, 512), Some(0));
    }
    #[test]
    fn external_allocations_and_underflow() {
        assert_eq!(choose_gpu(&[gpu(8192, 8000)], &[], 1, 512), None);
        assert_eq!(choose_gpu(&[gpu(8192, 1000)], &[], 6680, 512), Some(0));
    }
    #[test]
    fn release_after_completion() {
        assert_eq!(
            choose_gpu(&[gpu(8192, 0)], &[job(1, 7000, 0, "succeeded")], 7000, 512),
            Some(0)
        );
    }
    #[test]
    fn priority_then_fifo() {
        assert_eq!(
            queue_order(&[
                job(2, 1, 1, "queued"),
                job(1, 1, 1, "queued"),
                job(3, 1, 2, "queued")
            ]),
            vec![2, 1, 0]
        );
    }
    #[test]
    fn best_fit_multi_gpu() {
        let mut b = gpu(16000, 0);
        b.uuid = "GPU-b".into();
        assert_eq!(choose_gpu(&[gpu(8192, 0), b], &[], 9000, 512), Some(1));
    }
    #[test]
    fn rejects_invalid_telemetry() {
        for text in [
            "",
            "GPU-x, 10, 11, 0",
            "GPU-x, N/A, 0, 0",
            "GPU-x, 10, 0, 101",
        ] {
            assert!(parse_gpu_csv(text).is_err());
        }
        assert_eq!(
            parse_gpu_csv("GPU-x, 8192, 1024, 20\n").unwrap()[0].used_mib,
            1024
        );
    }
    #[test]
    fn invalid_request() {
        assert!(Submit {
            command: vec![],
            vram_mib: 1,
            priority: 0,
            profile: None,
            cooperative: false,
        }
        .validate()
        .is_err());
    }
}

pub fn parse_process_csv(input: &str) -> Result<Vec<GpuProcess>, String> {
    input
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<_> = line.split(',').map(str::trim).collect();
            if f.len() != 3 || f[1].is_empty() {
                return Err("invalid process CSV".into());
            }
            Ok(GpuProcess {
                pid: f[0].parse().map_err(|_| "invalid PID")?,
                gpu_uuid: f[1].into(),
                used_mib: f[2].parse().map_err(|_| "process memory unavailable")?,
            })
        })
        .collect()
}
#[cfg(test)]
mod process_tests {
    use super::*;
    #[test]
    fn process_samples() {
        let p = parse_process_csv("123, GPU-a, 4096\n").unwrap();
        assert_eq!(p[0].pid, 123);
        assert_eq!(p[0].used_mib, 4096);
        assert!(parse_process_csv("123, GPU-a, N/A").is_err());
        assert!(parse_process_csv("").unwrap().is_empty());
    }
}
