//! Cooperative reports are observations, never guaranteed memory limits.
use crate::{Gpu, Job, Submit};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

pub const PROFILE_HEADROOM_MIB: u64 = 128;
const MAX_REPORT_BYTES: u64 = 65536;
const MAX_STORE_BYTES: u64 = 8 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Measurement {
    Cuda,
    Simulation,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Success,
    Oom,
    Error,
    Checkpointed,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryReport {
    pub schema_version: u32,
    pub measurement: Measurement,
    pub outcome: Outcome,
    pub peak_allocated_mib: u64,
    pub peak_reserved_mib: u64,
}
impl MemoryReport {
    pub fn validate(
        &self,
        expected: Measurement,
        child_success: bool,
        gpu_total: u64,
    ) -> Result<(), String> {
        if self.schema_version != 1 || self.measurement != expected {
            return Err("report version/backend mismatch".into());
        }
        if (self.outcome == Outcome::Success) != child_success {
            return Err("report outcome disagrees with child exit".into());
        }
        if self.peak_allocated_mib > self.peak_reserved_mib || self.peak_reserved_mib > gpu_total {
            return Err("report peaks are inconsistent with assigned GPU capacity".into());
        }
        Ok(())
    }
}
pub fn read_report(path: &Path) -> Result<MemoryReport, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| "memory report missing or unreadable".to_string())?;
    if !metadata.is_file() {
        return Err("memory report must be a regular file".into());
    }
    let file = File::open(path).map_err(|_| "memory report missing or unreadable".to_string())?;
    let mut bytes = vec![];
    file.take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "report read failed")?;
    if bytes.len() as u64 > MAX_REPORT_BYTES {
        return Err("memory report exceeds 64 KiB".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "invalid memory report JSON".into())
}
#[derive(Clone, Default, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryProfile {
    pub peak_reserved_mib: u64,
    pub peak_allocated_mib: u64,
    pub oom_floor_mib: u64,
    pub successful_samples: u64,
    pub oom_samples: u64,
    pub error_samples: u64,
    #[serde(default)]
    pub checkpoint_samples: u64,
}
impl MemoryProfile {
    pub fn observe(&mut self, report: &MemoryReport, attempted_reservation: u64) {
        self.peak_reserved_mib = self.peak_reserved_mib.max(report.peak_reserved_mib);
        self.peak_allocated_mib = self.peak_allocated_mib.max(report.peak_allocated_mib);
        match report.outcome {
            Outcome::Success => self.successful_samples = self.successful_samples.saturating_add(1),
            Outcome::Oom => {
                self.oom_samples = self.oom_samples.saturating_add(1);
                self.oom_floor_mib = self.oom_floor_mib.max(attempted_reservation);
            }
            Outcome::Error => self.error_samples = self.error_samples.saturating_add(1),
            Outcome::Checkpointed => {
                self.checkpoint_samples = self.checkpoint_samples.saturating_add(1)
            }
        }
    }
    pub fn demand(&self, declared: u64) -> u64 {
        declared.max(
            self.peak_reserved_mib
                .max(self.oom_floor_mib)
                .saturating_add(PROFILE_HEADROOM_MIB),
        )
    }
}
/// Explicit workload key -> assigned physical GPU UUID -> observed profile.
pub type Profiles = BTreeMap<String, BTreeMap<String, MemoryProfile>>;
pub fn demand(request: &Submit, gpu: &Gpu, profiles: &Profiles) -> u64 {
    request
        .profile
        .as_ref()
        .and_then(|key| profiles.get(key))
        .and_then(|per_gpu| per_gpu.get(&gpu.uuid))
        .map_or(request.vram_mib, |p| p.demand(request.vram_mib))
}
#[derive(Debug, PartialEq, Eq)]
pub struct Placement {
    pub gpu_index: usize,
    pub reservation_mib: u64,
}
pub fn choose_profiled_gpu(
    gpus: &[Gpu],
    jobs: &[Job],
    request: &Submit,
    profiles: &Profiles,
    margin: u64,
) -> Option<Placement> {
    gpus.iter()
        .enumerate()
        .filter_map(|(i, gpu)| {
            let need = demand(request, gpu, profiles);
            let free = crate::available_mib(gpu, jobs, margin);
            (need <= free).then_some((
                Placement {
                    gpu_index: i,
                    reservation_mib: need,
                },
                free.saturating_sub(need),
            ))
        })
        .min_by_key(|(_, remaining)| *remaining)
        .map(|(p, _)| p)
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StoreDocument {
    schema_version: u32,
    measurement: Measurement,
    profiles: Profiles,
}
pub struct ProfileStore {
    root: PathBuf,
    path: PathBuf,
    // Held for daemon lifetime; advisory OS lock is released on exit/crash.
    _lock: File,
    measurement: Measurement,
    pub profiles: Profiles,
}
impl ProfileStore {
    pub fn open(root: &Path, measurement: Measurement) -> Result<Self, String> {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(root)
            .map_err(|e| format!("state directory: {e}"))?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(root.join("daemon.lock"))
            .map_err(|e| e.to_string())?;
        lock.try_lock_exclusive()
            .map_err(|_| "state directory is already in use by another daemon".to_string())?;
        let name = if measurement == Measurement::Simulation {
            "mock-profiles.json"
        } else {
            "cuda-profiles.json"
        };
        let path = root.join(name);
        let profiles = match File::open(&path) {
            Ok(file) => {
                let mut bytes = vec![];
                file.take(MAX_STORE_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                if bytes.len() as u64 > MAX_STORE_BYTES {
                    return Err("profile store too large".into());
                }
                let document: StoreDocument = serde_json::from_slice(&bytes).map_err(|_| {
                    "invalid profile store; preserve and inspect it before recovery".to_string()
                })?;
                if document.schema_version != 1 || document.measurement != measurement {
                    return Err("profile store schema/backend mismatch".into());
                }
                for (key, gpus) in &document.profiles {
                    if key.is_empty() || key.len() > 128 || key.chars().any(char::is_control) {
                        return Err("invalid saved profile key".into());
                    }
                    for (uuid, p) in gpus {
                        if uuid.is_empty() || p.peak_allocated_mib > p.peak_reserved_mib {
                            return Err("invalid saved profile observation".into());
                        }
                    }
                }
                document.profiles
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Profiles::new(),
            Err(e) => return Err(format!("cannot read profile store: {e}")),
        };
        Ok(Self {
            root: root.into(),
            path,
            _lock: lock,
            measurement,
            profiles,
        })
    }
    pub fn save(&self) -> Result<(), String> {
        let doc = StoreDocument {
            schema_version: 1,
            measurement: self.measurement,
            profiles: self.profiles.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&doc).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_STORE_BYTES {
            return Err("profile store exceeds 8 MiB".into());
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let temp = self
            .root
            .join(format!("profiles-{}-{stamp}.tmp", std::process::id()));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp)
                .map_err(|e| e.to_string())?;
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            fs::rename(&temp, &self.path).map_err(|e| e.to_string())?;
            File::open(&self.root)
                .and_then(|d| d.sync_all())
                .map_err(|e| e.to_string())?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn report(peak: u64, outcome: Outcome) -> MemoryReport {
        MemoryReport {
            schema_version: 1,
            measurement: Measurement::Simulation,
            outcome,
            peak_allocated_mib: peak / 2,
            peak_reserved_mib: peak,
        }
    }
    #[test]
    fn peak_and_oom_are_monotonic() {
        let mut p = MemoryProfile::default();
        p.observe(&report(5000, Outcome::Success), 1000);
        assert_eq!(p.demand(1000), 5128);
        p.observe(&report(2000, Outcome::Success), 1000);
        assert_eq!(p.demand(1000), 5128);
        p.observe(&report(4000, Outcome::Oom), 6000);
        assert_eq!(p.demand(1000), 6128);
        assert_eq!(p.oom_samples, 1);
        assert_eq!(p.demand(7000), 7000);
        p.oom_floor_mib = u64::MAX;
        assert_eq!(p.demand(1), u64::MAX);
    }
    #[test]
    fn report_contract() {
        assert!(report(5000, Outcome::Success)
            .validate(Measurement::Simulation, true, 8192)
            .is_ok());
        assert!(report(5000, Outcome::Success)
            .validate(Measurement::Cuda, true, 8192)
            .is_err());
        assert!(report(5000, Outcome::Oom)
            .validate(Measurement::Simulation, true, 8192)
            .is_err());
        assert!(report(9000, Outcome::Success)
            .validate(Measurement::Simulation, true, 8192)
            .is_err());
        let mut r = report(5000, Outcome::Success);
        r.peak_allocated_mib = 6000;
        assert!(r.validate(Measurement::Simulation, true, 8192).is_err());
    }
    #[test]
    fn placement_uses_learned_reservations() {
        let gpu = Gpu {
            uuid: "GPU-a".into(),
            total_mib: 8192,
            used_mib: 0,
            utilization: 0,
        };
        let request = Submit {
            command: vec!["true".into()],
            vram_mib: 1000,
            priority: 0,
            profile: Some("model".into()),
            cooperative: false,
        };
        let mut profiles = Profiles::new();
        profiles.entry("model".into()).or_default().insert(
            "GPU-a".into(),
            MemoryProfile {
                peak_reserved_mib: 7000,
                ..Default::default()
            },
        );
        let p =
            choose_profiled_gpu(std::slice::from_ref(&gpu), &[], &request, &profiles, 512).unwrap();
        assert_eq!(p.reservation_mib, 7128);
        let job = Job {
            id: 1,
            request: request.clone(),
            state: "running".into(),
            gpu: Some("GPU-a".into()),
            pid: None,
            exit_code: None,
            message: String::new(),
            reservation_mib: 7128,
            memory_report: None,
            report_error: None,
            attempts: 1,
            queue_ticket: 1,
            checkpoint: None,
        };
        assert!(
            choose_profiled_gpu(std::slice::from_ref(&gpu), &[job], &request, &profiles, 512)
                .is_none()
        );
        let other = Gpu {
            uuid: "GPU-b".into(),
            ..gpu
        };
        assert_eq!(demand(&request, &other, &profiles), 1000);
    }
    #[test]
    fn store_roundtrip_lock_and_backend_separation() {
        let root = std::env::temp_dir().join(format!(
            "mlus-store-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let mut store = ProfileStore::open(&root, Measurement::Simulation).unwrap();
            assert!(ProfileStore::open(&root, Measurement::Cuda).is_err());
            store.profiles.entry("model".into()).or_default().insert(
                "GPU-a".into(),
                MemoryProfile {
                    peak_reserved_mib: 5000,
                    ..Default::default()
                },
            );
            store.save().unwrap();
        }
        {
            let store = ProfileStore::open(&root, Measurement::Simulation).unwrap();
            assert_eq!(store.profiles["model"]["GPU-a"].peak_reserved_mib, 5000);
        }
        assert!(ProfileStore::open(&root, Measurement::Cuda)
            .unwrap()
            .profiles
            .is_empty());
        fs::write(root.join("mock-profiles.json"), b"broken").unwrap();
        assert!(ProfileStore::open(&root, Measurement::Simulation).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
