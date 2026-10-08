use mlus::checkpoint::{
    accept_handoff, read_checkpoint_report, scheduling_request, CHECKPOINT_EXIT_CODE,
};
use mlus::journal::{Journal, MAX_JOBS};
use mlus::processes::{enable_subreaper, ManagedWorkers, Worker};
use mlus::profiles::{choose_profiled_gpu, demand, read_report, Measurement, ProfileStore};
use mlus::{queue_order, Backend, Gpu, Job, Mock, Nvidia, Submit};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::TcpStream,
    os::unix::{fs::OpenOptionsExt, process::CommandExt},
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tiny_http::{Header, Method, Response, Server};

fn main() {
    if let Err(e) = run() {
        eprintln!("mlus: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags = &args[..args.iter().position(|a| a == "--").unwrap_or(args.len())];
    validate_options(flags)?;
    let port = option(flags, "--port")
        .unwrap_or("8787")
        .parse::<u16>()
        .map_err(|_| "invalid port")?;
    match args.first().map(String::as_str) {
        Some("serve") => serve(
            port,
            args.iter().any(|a| a == "--mock"),
            option(flags, "--margin-mib")
                .unwrap_or("512")
                .parse()
                .map_err(|_| "invalid margin")?,
            Path::new(option(flags, "--state-dir").unwrap_or(".mlus-state")),
        ),
        Some("status") => {
            println!(
                "{}",
                serde_json::to_string_pretty(&request(port, "GET", "/api/status", "")?).unwrap()
            );
            Ok(())
        }
        Some("recover") => {
            if !flags.iter().any(|a| a == "--confirm-cleanup") {
                return Err("recover requires --confirm-cleanup after checking and stopping remaining workers".into());
            }
            println!(
                "{}",
                request(port, "POST", "/api/recover", r#"{"confirm_cleanup":true}"#)?
            );
            Ok(())
        }
        Some("submit") => {
            let separator = args
                .iter()
                .position(|a| a == "--")
                .ok_or("submit requires -- before command")?;
            let flags = &args[..separator];
            let req = Submit {
                command: args[separator + 1..].to_vec(),
                vram_mib: option(flags, "--vram-mib")
                    .ok_or("--vram-mib required")?
                    .parse()
                    .map_err(|_| "invalid VRAM")?,
                profile: option(flags, "--profile").map(str::to_owned),
                cooperative: flags.iter().any(|a| a == "--cooperative"),
                priority: option(flags, "--priority")
                    .unwrap_or("0")
                    .parse()
                    .map_err(|_| "invalid priority")?,
            };
            req.validate()?;
            println!(
                "{}",
                request(
                    port,
                    "POST",
                    "/api/jobs",
                    &serde_json::to_string(&req).unwrap()
                )?
            );
            Ok(())
        }
        _ => {
            println!("MLus 0.5\n  mlus serve [--mock] [--port 8787] [--margin-mib 512] [--state-dir PATH]\n  mlus submit [--port 8787] --vram-mib N [--priority N] [--profile KEY] [--cooperative] -- COMMAND ARGS...\n  mlus status [--port 8787]\n  mlus recover [--port 8787] --confirm-cleanup");
            Ok(())
        }
    }
}
fn validate_options(args: &[String]) -> Result<(), String> {
    let allowed: &[&str] = match args.first().map(String::as_str) {
        Some("serve") => &["--port", "--margin-mib", "--state-dir"],
        Some("submit") => &["--port", "--vram-mib", "--priority", "--profile"],
        Some("status") | Some("recover") => &["--port"],
        None | Some("--help") => return Ok(()),
        _ => return Err("unknown command; run mlus --help".into()),
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut i = 1;
    while i < args.len() {
        let flag = &args[i];
        if !seen.insert(flag) {
            return Err(format!("duplicate option: {flag}"));
        }
        if (args[0] == "serve" && flag == "--mock")
            || (args[0] == "submit" && flag == "--cooperative")
            || (args[0] == "recover" && flag == "--confirm-cleanup")
        {
            i += 1;
            continue;
        }
        if !allowed.contains(&flag.as_str()) {
            return Err(format!("unknown option: {flag}"));
        }
        if i + 1 >= args.len() || args[i + 1].starts_with("--") {
            return Err(format!("value required for {flag}"));
        }
        i += 2;
    }
    Ok(())
}
fn option<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|a| a[0] == flag)
        .map(|a| a[1].as_str())
}
fn request(port: u16, method: &str, path: &str, body: &str) -> Result<Value, String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    write!(stream,"{method} {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).map_err(|e|e.to_string())?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|e| e.to_string())?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or("invalid server response")?;
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
        return Err(body.into());
    }
    serde_json::from_str(body).map_err(|e| e.to_string())
}
fn serve(port: u16, mock: bool, margin: u64, state_dir: &Path) -> Result<(), String> {
    let backend: Box<dyn Backend> = if mock {
        Box::new(Mock(vec![
            Gpu {
                uuid: "MOCK-0".into(),
                total_mib: 8192,
                used_mib: 0,
                utilization: 0,
            },
            Gpu {
                uuid: "MOCK-1".into(),
                total_mib: 12288,
                used_mib: 0,
                utilization: 0,
            },
        ]))
    } else {
        Box::new(Nvidia)
    };
    let mut gpus = backend.sample()?;
    let measurement = if mock {
        Measurement::Simulation
    } else {
        Measurement::Cuda
    };
    let mut store = ProfileStore::open(state_dir, measurement)?;
    store.save()?;
    let mut profile_store_error: Option<String> = None;
    let server = Server::http(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed)).map_err(|e| e.to_string())?;
    let (mut journal, mut jobs) = Journal::open(state_dir, measurement)?;
    let mut job_store_error: Option<String> = None;
    let mut next_ticket = jobs.iter().map(|j| j.queue_ticket).max().unwrap_or(0);
    enable_subreaper()?;
    let mut children = ManagedWorkers(HashMap::new());
    let log_dir = journal.log_directory.clone();
    println!("Job output directory: {}", log_dir.display());
    println!(
        "MLus listening on 127.0.0.1:{port}; backend={}",
        if mock {
            "simulation (no GPU execution validated)"
        } else {
            "NVIDIA"
        }
    );
    while !stop.load(Ordering::Relaxed) {
        let sample_error = match backend.sample() {
            Ok(sample) => {
                gpus = sample;
                None
            }
            Err(e) => Some(e),
        };
        let mut process_error: Option<String> = None;
        let mut profiles_dirty = false;
        let mut jobs_dirty = false;
        for job in jobs.iter_mut().filter(|j| j.state == "running") {
            if let Some(child) = children.0.get_mut(&job.id) {
                match child.poll() {
                    Ok(Some(status)) => {
                        jobs_dirty = true;
                        let attempted_reservation = job.reservation_mib;
                        let mut handoff_accepted = false;
                        job.state = if status.success() {
                            "succeeded"
                        } else {
                            "failed"
                        }
                        .into();
                        job.exit_code = status.code();
                        job.message = format!("worker group cleaned up; leader exited: {status}");
                        if job.request.cooperative && status.code() == Some(CHECKPOINT_EXIT_CODE) {
                            let handoff_path = log_dir.join(format!(
                                "job-{}-attempt-{}.checkpoint.json",
                                job.id, job.attempts
                            ));
                            match read_checkpoint_report(&handoff_path).and_then(|r| {
                                r.validate(job, measurement)?;
                                Ok(r)
                            }) {
                                Ok(report) => {
                                    next_ticket = next_ticket.saturating_add(1);
                                    accept_handoff(job, report, next_ticket);
                                    handoff_accepted = true;
                                }
                                Err(e) => job.message = format!("checkpoint rejected: {e}"),
                            }
                        }
                        if let Some(key) = &job.request.profile {
                            let report_path = log_dir.join(format!(
                                "job-{}-attempt-{}.memory.json",
                                job.id, job.attempts
                            ));
                            let result = read_report(&report_path).and_then(|report| {
                                let gpu = gpus
                                    .iter()
                                    .find(|g| Some(&g.uuid) == job.gpu.as_ref())
                                    .ok_or("assigned GPU no longer present")?;
                                report.validate(measurement, status.success(), gpu.total_mib)?;
                                if (report.outcome == mlus::profiles::Outcome::Checkpointed)
                                    != handoff_accepted
                                {
                                    return Err(
                                        "memory report disagrees with checkpoint handoff".into()
                                    );
                                }
                                Ok(report)
                            });
                            match result {
                                Ok(report) => {
                                    store
                                        .profiles
                                        .entry(key.clone())
                                        .or_default()
                                        .entry(job.gpu.clone().unwrap())
                                        .or_default()
                                        .observe(&report, attempted_reservation);
                                    job.memory_report = Some(report);
                                    profiles_dirty = true;
                                }
                                Err(e) => job.report_error = Some(e),
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        job.message = format!("worker cleanup blocked: {e}");
                        process_error = Some(e);
                    }
                }
            }
        }
        children
            .0
            .retain(|id, _| jobs.iter().any(|j| j.id == *id && j.state == "running"));
        let escaped_children = match children.escaped_children() {
            Ok(pids) => {
                if !pids.is_empty() {
                    process_error = Some(
                        "live adopted children escaped managed groups; inspect before recovery"
                            .into(),
                    );
                }
                pids
            }
            Err(e) => {
                process_error = Some(e);
                vec![]
            }
        };
        if process_error.is_some() && !journal.recovery_required {
            eprintln!("worker recovery gate: {process_error:?}; escaped={escaped_children:?}");
            journal.recovery_required = true;
            jobs_dirty = true;
        }
        if profiles_dirty {
            profile_store_error = store.save().err();
        }
        if jobs_dirty {
            job_store_error = journal.save(&jobs).err();
            jobs_dirty = false;
        }
        if sample_error.is_none() && !journal.recovery_required && job_store_error.is_none() {
            for i in queue_order(&jobs) {
                if job_store_error.is_some() {
                    break;
                }
                if jobs[i].request.profile.is_some() && profile_store_error.is_some() {
                    jobs[i].message =
                        "waiting: profile persistence failed; inspect state directory".into();
                    continue;
                }
                let is_resume = jobs[i].state == "waiting_resume";
                let scheduling_request = scheduling_request(&jobs[i]);
                let candidates: Vec<Gpu> = gpus
                    .iter()
                    .filter(|g| !is_resume || Some(&g.uuid) == jobs[i].gpu.as_ref())
                    .cloned()
                    .collect();
                if candidates.is_empty() {
                    jobs[i].message =
                        "waiting for previously assigned GPU UUID to become available".into();
                    continue;
                }
                if !candidates.iter().any(|g| {
                    demand(&scheduling_request, g, &store.profiles)
                        <= g.total_mib.saturating_sub(margin)
                }) {
                    jobs_dirty = true;
                    jobs[i].state = "rejected".into();
                    jobs[i].message =
                        "updated profile exceeds every GPU capacity minus margin".into();
                    continue;
                }
                if let Some(placement) = choose_profiled_gpu(
                    &candidates,
                    &jobs,
                    &scheduling_request,
                    &store.profiles,
                    margin,
                ) {
                    let gpu_index = placement.gpu_index;
                    let job = &mut jobs[i];
                    let gpu = &candidates[gpu_index];
                    let next_attempt = job.attempts + 1;
                    let mut cmd = Command::new(&job.request.command[0]);
                    cmd.args(&job.request.command[1..])
                        .current_dir(&journal.working_directory)
                        .process_group(0);
                    // This is placement, never a CUDA memory limit or security boundary.
                    let output_path = log_dir.join(format!("job-{}.log", job.id));
                    let output = std::fs::OpenOptions::new()
                        .append(true)
                        .create(true)
                        .mode(0o600)
                        .open(&output_path)
                        .map_err(|e| e.to_string())?;
                    cmd.env("CUDA_VISIBLE_DEVICES", &gpu.uuid)
                        .stdin(Stdio::null())
                        .stderr(Stdio::from(output.try_clone().map_err(|e| e.to_string())?))
                        .stdout(Stdio::from(output));
                    if mock {
                        cmd.env_remove("CUDA_VISIBLE_DEVICES");
                        cmd.env("MLUS_MOCK_GPU", &gpu.uuid);
                    }
                    if let Some(key) = &job.request.profile {
                        cmd.env(
                            "MLUS_REPORT_PATH",
                            log_dir.join(format!(
                                "job-{}-attempt-{}.memory.json",
                                job.id, next_attempt
                            )),
                        )
                        .env("MLUS_PROFILE_KEY", key)
                        .env("MLUS_BACKEND", if mock { "mock" } else { "nvidia" });
                    } else {
                        cmd.env_remove("MLUS_REPORT_PATH")
                            .env_remove("MLUS_PROFILE_KEY")
                            .env_remove("MLUS_BACKEND");
                    }
                    if job.request.cooperative {
                        cmd.env("MLUS_COOPERATIVE", "1")
                            .env("MLUS_BACKEND", if mock { "mock" } else { "nvidia" })
                            .env(
                                "MLUS_CHECKPOINT_REPORT_PATH",
                                log_dir.join(format!(
                                    "job-{}-attempt-{}.checkpoint.json",
                                    job.id, next_attempt
                                )),
                            )
                            .env("MLUS_ATTEMPT", next_attempt.to_string());
                        if is_resume {
                            let path = &job.checkpoint.as_ref().unwrap().checkpoint_path;
                            if !std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file()) {
                                jobs_dirty = true;
                                job.state = "failed".into();
                                job.message = "checkpoint file unavailable before resume".into();
                                continue;
                            }
                            cmd.env("MLUS_CHECKPOINT_PATH", path);
                        } else {
                            cmd.env_remove("MLUS_CHECKPOINT_PATH");
                        }
                    } else {
                        cmd.env_remove("MLUS_COOPERATIVE")
                            .env_remove("MLUS_CHECKPOINT_REPORT_PATH")
                            .env_remove("MLUS_ATTEMPT")
                            .env_remove("MLUS_CHECKPOINT_PATH");
                    }
                    // Durable intent precedes spawn. Any uncertain start blocks future recovery.
                    job.state = "starting".into();
                    job.attempts = next_attempt;
                    job.exit_code = None;
                    job.memory_report = None;
                    job.report_error = None;
                    job.reservation_mib = placement.reservation_mib;
                    job.gpu = Some(gpu.uuid.clone());
                    job.pid = None;
                    job.message = "launch intent persisted before spawn".into();
                    if let Err(e) = journal.save(&jobs) {
                        job_store_error = Some(e);
                        jobs[i].state = "failed".into();
                        jobs[i].reservation_mib = 0;
                        jobs[i].message = "not launched: job persistence failed".into();
                        break;
                    }
                    let job = &mut jobs[i];
                    match cmd.spawn() {
                        Ok(child) => {
                            job.state = "running".into();
                            job.pid = Some(child.id());
                            job.message = "admitted; reservation is advisory".into();
                            children.0.insert(job.id, Worker::new(child));
                        }
                        Err(e) => {
                            job.state = "failed".into();
                            job.reservation_mib = 0;
                            job.message = format!("spawn failed: {e}");
                        }
                    }
                    job_store_error = journal.save(&jobs).err();
                } else {
                    jobs[i].message =
                        "waiting for free VRAM plus safety margin and reservations".into();
                }
            }
        }
        if jobs_dirty {
            job_store_error = journal.save(&jobs).err();
        }
        if let Some(mut req) = server
            .recv_timeout(Duration::from_millis(100))
            .map_err(|e| e.to_string())?
        {
            let (status, body, content_type) = match (req.method(), req.url()) {
                (&Method::Get, "/") => (
                    200,
                    include_str!("../assets/dashboard.html").to_string(),
                    "text/html; charset=utf-8",
                ),
                (&Method::Get, "/api/status") => {
                    let processes = backend.processes();
                    let (processes, process_telemetry_error) = match processes {
                        Ok(p) => (p, None),
                        Err(e) => (vec![], Some(e)),
                    };
                    (200,json!({"backend":if mock{"mock"}else{"nvidia"},"daemon_pid":std::process::id(),"version":env!("CARGO_PKG_VERSION"),"state_directory":state_dir,"gpus":gpus,"jobs":jobs,"margin_mib":margin,"log_directory":log_dir,"telemetry_error":sample_error,"gpu_processes":processes,"process_telemetry_error":process_telemetry_error,"profiles":store.profiles,"profile_store_error":profile_store_error,"job_store_error":job_store_error,"recovery_required":journal.recovery_required,"working_directory":journal.working_directory,"process_management":{"mode":"linux_process_group","subreaper":true,"workers":children.status(),"escaped_children":escaped_children,"cleanup_error":process_error},"profile_headroom_mib":mlus::profiles::PROFILE_HEADROOM_MIB}).to_string(),"application/json")
                }
                (&Method::Post, "/api/jobs") => {
                    let result = local_json(&mut req, port).and_then(|body| {
                        let submit: Submit =
                            serde_json::from_str(&body).map_err(|e| e.to_string())?;
                        submit.validate()?;
                        Ok(submit)
                    });
                    match result {
                        Ok(submit) if jobs.len() < MAX_JOBS => {
                            if !gpus.iter().any(|g| {
                                demand(&submit, g, &store.profiles)
                                    <= g.total_mib.saturating_sub(margin)
                            }) {
                                (400,json!({"error":"request or learned profile exceeds every GPU capacity minus margin"}).to_string(),"application/json")
                            } else {
                                let id = jobs.len() as u64 + 1;
                                next_ticket = next_ticket.saturating_add(1);
                                jobs.push(Job {
                                    id,
                                    request: submit,
                                    state: "queued".into(),
                                    gpu: None,
                                    pid: None,
                                    exit_code: None,
                                    message: "awaiting admission".into(),
                                    reservation_mib: 0,
                                    memory_report: None,
                                    report_error: None,
                                    attempts: 0,
                                    queue_ticket: next_ticket,
                                    checkpoint: None,
                                });
                                job_store_error = journal.save(&jobs).err();
                                match &job_store_error {
                                    None => (200, json!({"id":id,"state":"queued"}).to_string(), "application/json"),
                                    Some(e) => (503, json!({"error":e,"id":id,"state":"queued","persistence":"unconfirmed"}).to_string(), "application/json"),
                                }
                            }
                        }
                        Ok(_) => (
                            400,
                            json!({"error":"job history full; archive the state directory after stopping all workers"})
                                .to_string(),
                            "application/json",
                        ),
                        Err(e) => (400, json!({"error":e}).to_string(), "application/json"),
                    }
                }
                (&Method::Post, "/api/recover") => {
                    let confirm = local_json(&mut req, port).and_then(|body| {
                        let value: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
                        if value != json!({"confirm_cleanup":true}) { return Err("explicit confirm_cleanup=true required after inspecting remaining workers".into()); }
                        Ok(())
                    });
                    match confirm {
                        Ok(()) if process_error.is_some() => (409, json!({"error":"worker cleanup still blocked; inspect current processes before recovery"}).to_string(), "application/json"),
                        Err(e) => (400, json!({"error":e}).to_string(), "application/json"),
                        Ok(()) => {
                            let previous = journal.recovery_required;
                            journal.recovery_required = false;
                            match journal.save(&jobs) {
                                Ok(()) => {
                                    job_store_error = None;
                                    (200,json!({"recovery_required":false,"interrupted_jobs":"not retried"}).to_string(),"application/json")
                                }
                                Err(e) => {
                                    journal.recovery_required = previous;
                                    job_store_error = Some(e.clone());
                                    (503, json!({"error":e}).to_string(), "application/json")
                                }
                            }
                        }
                    }
                }
                _ => (
                    404,
                    json!({"error":"not found"}).to_string(),
                    "application/json",
                ),
            };
            let response = Response::from_string(body)
                .with_status_code(status)
                .with_header(Header::from_bytes("Content-Type", content_type).unwrap());
            let _ = req.respond(response);
        }
    }
    // Reap only children owned by this daemon. Never signal a PID loaded from disk.
    for (id, result) in children.shutdown() {
        let job = jobs.iter_mut().find(|j| j.id == id).unwrap();
        match result {
            Ok(status) => {
                job.state = "cancelled".into();
                job.exit_code = status.code();
                job.pid = None;
                job.reservation_mib = 0;
                job.message =
                    "worker process group reaped during graceful shutdown; not retried".into();
            }
            Err(e) => {
                job.state = "interrupted".into();
                job.message = format!("shutdown wait failed: {e}");
                journal.recovery_required = true;
            }
        }
    }
    if !children
        .escaped_children()
        .is_ok_and(|pids| pids.is_empty())
    {
        journal.recovery_required = true;
    }
    journal.save(&jobs)?;
    Ok(())
}

fn local_json(req: &mut tiny_http::Request, port: u16) -> Result<String, String> {
    let json_content = req.headers().iter().any(|h| {
        h.field.equiv("Content-Type")
            && h.value.as_str().split(';').next() == Some("application/json")
    });
    let has_origin = req.headers().iter().any(|h| h.field.equiv("Origin"));
    let valid_host = req.headers().iter().any(|h| {
        h.field.equiv("Host")
            && [
                format!("127.0.0.1:{port}"),
                format!("localhost:{port}"),
                "localhost".into(),
            ]
            .contains(&h.value.as_str().to_string())
    });
    if !json_content || has_origin || !valid_host {
        return Err("local JSON clients only; browser submissions are disabled".into());
    }
    if req.body_length().is_some_and(|n| n > 65536) {
        return Err("request too large".into());
    }
    let mut body = String::new();
    req.as_reader()
        .take(65537)
        .read_to_string(&mut body)
        .map_err(|e| e.to_string())?;
    if body.len() > 65536 {
        return Err("request too large".into());
    }
    Ok(body)
}
