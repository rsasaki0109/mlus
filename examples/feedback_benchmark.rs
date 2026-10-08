//! Synthetic underestimation experiment, not actual CUDA/OOM/performance data.
use mlus::profiles::{
    choose_profiled_gpu, Measurement, MemoryProfile, MemoryReport, Outcome, Profiles,
};
use mlus::{Gpu, Job, Submit};
use serde_json::json;
fn simulate(profiles: &Profiles) -> serde_json::Value {
    let gpu = Gpu {
        uuid: "SIM-a".into(),
        total_mib: 8192,
        used_mib: 0,
        utilization: 0,
    };
    let mut jobs: Vec<_> = (1..=4)
        .map(|id| Job {
            id,
            request: Submit {
                command: vec!["synthetic".into()],
                vram_mib: 1000,
                priority: 0,
                profile: Some("model:shape-fixed".into()),
                cooperative: false,
            },
            state: "queued".into(),
            gpu: None,
            pid: None,
            exit_code: None,
            message: String::new(),
            reservation_mib: 0,
            memory_report: None,
            report_error: None,
            attempts: 1,
            queue_ticket: id,
            checkpoint: None,
        })
        .collect();
    let mut finish = [0; 4];
    let mut tick = 0;
    let mut peak_footprint = 0;
    let mut peak_reserved = 0;
    while jobs.iter().any(|j| j.state != "succeeded") {
        for i in 0..4 {
            if jobs[i].state == "running" && finish[i] <= tick {
                jobs[i].state = "succeeded".into();
            }
        }
        for i in 0..4 {
            if jobs[i].state == "queued" {
                if let Some(p) = choose_profiled_gpu(
                    std::slice::from_ref(&gpu),
                    &jobs,
                    &jobs[i].request,
                    profiles,
                    512,
                ) {
                    jobs[i].state = "running".into();
                    jobs[i].gpu = Some(gpu.uuid.clone());
                    jobs[i].reservation_mib = p.reservation_mib;
                    finish[i] = tick + 3;
                }
            }
        }
        let running: Vec<_> = jobs.iter().filter(|j| j.state == "running").collect();
        peak_footprint = peak_footprint.max(running.len() as u64 * 5000);
        peak_reserved = peak_reserved.max(running.iter().map(|j| j.reservation_mib).sum::<u64>());
        if jobs.iter().all(|j| j.state == "succeeded") {
            break;
        }
        tick += 1;
        assert!(tick < 100);
    }
    json!({"peak_reserved_mib":peak_reserved,"peak_synthetic_footprint_mib":peak_footprint,"footprint_exceeds_budget":peak_footprint>7680,"makespan_ticks_if_every_job_runs":tick})
}
fn main() {
    let mut profiles = Profiles::new();
    let baseline = simulate(&profiles);
    let mut p = MemoryProfile::default();
    p.observe(
        &MemoryReport {
            schema_version: 1,
            measurement: Measurement::Simulation,
            outcome: Outcome::Success,
            peak_allocated_mib: 4000,
            peak_reserved_mib: 5000,
        },
        1000,
    );
    profiles
        .entry("model:shape-fixed".into())
        .or_default()
        .insert("SIM-a".into(), p);
    let feedback = simulate(&profiles);
    assert_eq!(baseline["footprint_exceeds_budget"], true);
    assert_eq!(feedback["footprint_exceeds_budget"], false);
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"synthetic_profile_feedback_not_gpu_measurement","declared_mib":1000,"synthetic_footprint_per_job_mib":5000,"calibration":"synthetic","without_profile":baseline,"with_profile":feedback,"actual_oom_measured":false,"timing_note":"fixed synthetic durations; infeasible baseline time is not usable throughput"})).unwrap());
}
