//! Deterministic model, not GPU performance measurement.
use mlus::{choose_gpu, queue_order, Gpu, Job, Submit};
fn main() {
    let gpu = Gpu {
        uuid: "SIM-0".into(),
        total_mib: 8192,
        used_mib: 0,
        utilization: 0,
    };
    let sizes = [3000, 3000, 3000, 3000];
    let durations = [3, 2, 4, 1];
    let mut jobs: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| Job {
            id: i as u64 + 1,
            request: Submit {
                command: vec!["simulated".into()],
                vram_mib: n,
                priority: 0,
                profile: None,
                cooperative: false,
            },
            state: "queued".into(),
            gpu: None,
            pid: None,
            exit_code: None,
            message: String::new(),
            reservation_mib: n,
            memory_report: None,
            report_error: None,
            attempts: 1,
            queue_ticket: i as u64 + 1,
            checkpoint: None,
        })
        .collect();
    let mut ends = [0; 4];
    let mut tick = 0;
    let mut peak = 0;
    let mut starts = [0; 4];
    while jobs.iter().any(|j| j.state != "succeeded") {
        for i in 0..4 {
            if jobs[i].state == "running" && ends[i] <= tick {
                jobs[i].state = "succeeded".into();
            }
        }
        for i in queue_order(&jobs) {
            if choose_gpu(
                std::slice::from_ref(&gpu),
                &jobs,
                jobs[i].request.vram_mib,
                512,
            )
            .is_some()
            {
                jobs[i].state = "running".into();
                jobs[i].gpu = Some(gpu.uuid.clone());
                starts[i] = tick;
                ends[i] = tick + durations[i];
            }
        }
        let reserved: u64 = jobs
            .iter()
            .filter(|j| j.state == "running")
            .map(|j| j.reservation_mib)
            .sum();
        assert!(reserved <= 7680);
        peak = peak.max(reserved);
        if jobs.iter().all(|j| j.state == "succeeded") {
            break;
        }
        tick += 1;
        assert!(tick < 100);
    }
    let report = serde_json::json!({"kind":"deterministic_simulation_not_gpu_measurement","gpu_total_mib":8192,"margin_mib":512,"jobs_mib":sizes,"durations_ticks":durations,"unrestricted_launch":{"peak_requested_mib":sizes.iter().sum::<u64>(),"exceeds_budget":true,"actual_oom_measured":false},"mlus":{"peak_reserved_mib":peak,"exceeds_budget":false,"completed":4,"makespan_ticks":tick,"start_ticks":starts},"serial":{"makespan_ticks":durations.iter().sum::<u64>(),"peak_reserved_mib":3000}});
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
