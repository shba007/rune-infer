//! Standalone Windows CPU parity check. Build with rustc; no gateway dependencies.
//! Usage: native_check INFER_EXE LITERT_DLL CANDIDATE_TFLITE FIXTURE_PREFIX
use std::{
    env, fs,
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const PIXELS: usize = 518 * 686;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 4 || !cfg!(all(windows, target_arch = "x86_64")) {
        return Err(
            "Windows x64 usage: native_check INFER_EXE LITERT_DLL CANDIDATE_TFLITE FIXTURE_PREFIX"
                .into(),
        );
    }
    let prefix = Path::new(&args[3]);
    let input = fs::read(prefix.with_extension("input.f32"))?;
    let reference = fs::read(prefix.with_extension("reference.f32"))?;
    if input.len() != PIXELS * 3 * 4 || reference.len() != PIXELS * 4 {
        return Err("Invalid parity fixture size".into());
    }
    let mut child = Command::new(&args[0])
        .arg("litert-worker")
        .arg("--runtime")
        .arg(&args[1])
        .arg("--model")
        .arg(&args[2])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("Missing input pipe")?;
    let stdout = child.stdout.take().ok_or("Missing output pipe")?;
    let stderr = child.stderr.take().ok_or("Missing error pipe")?;
    let writer = thread::spawn(move || stdin.write_all(&input));
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take((PIXELS * 4 + 1) as u64)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let errors = thread::spawn(move || {
        let mut stderr = stderr;
        let mut saved = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let n = stderr.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            let keep = n.min(8192_usize.saturating_sub(saved.len()));
            saved.extend_from_slice(&buffer[..keep]);
        }
        Ok::<_, std::io::Error>(saved)
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() > Duration::from_secs(120) {
            child.kill()?;
            child.wait()?;
            return Err("Native Large inference exceeded 120 seconds".into());
        }
        thread::sleep(Duration::from_millis(20));
    };
    let errors = errors.join().map_err(|_| "Diagnostic thread failed")??;
    if !status.success() {
        return Err(format!(
            "Native worker failed ({status}): {}",
            String::from_utf8_lossy(&errors)
        )
        .into());
    }
    writer.join().map_err(|_| "Writer thread failed")??;
    let actual = reader.join().map_err(|_| "Reader thread failed")??;
    if actual.len() != reference.len() {
        return Err("Invalid native output size".into());
    }
    let (mut maximum, mut sum, mut failures) = (0_f64, 0_f64, 0_usize);
    for (a, b) in actual.chunks_exact(4).zip(reference.chunks_exact(4)) {
        let actual = f32::from_le_bytes(a.try_into()?) as f64;
        let reference = f32::from_le_bytes(b.try_into()?) as f64;
        if !actual.is_finite() || !reference.is_finite() {
            return Err("Non-finite parity output".into());
        }
        let error = (actual - reference).abs();
        maximum = maximum.max(error);
        sum += error;
        failures += usize::from(error > 0.001 + 0.001 * reference.abs());
    }
    println!(
        "values={PIXELS} max_absolute_error={maximum} mean_absolute_error={} failed_values={failures} elapsed_seconds={}",
        sum / PIXELS as f64,
        started.elapsed().as_secs_f64()
    );
    if failures != 0 {
        return Err("Windows native output exceeds FP32 parity tolerance".into());
    }
    Ok(())
}
