//! Validation-only worker; never linked into Infer's server.
//! Build against Infer's existing sha2 and anyhow artifacts. The caller must
//! set RUNE_LITERT_CANDIDATE_SHA256 to the conversion report's candidate digest.
//! Experimental GPU bindings are isolated from the production CPU worker.
use std::{io::Read, path::Path};

#[path = "gpu_abi.rs"]
mod native;

const PIXELS: usize = 518 * 686;
const RUNTIME_SHA: &str = "6fe42cfa518879dd7bae4500244cf371d949bf0c8559cb60be207d81112da16d";

fn probe_shape() -> anyhow::Result<Option<Vec<i32>>> {
    let Some(value) = std::env::var_os("RUNE_LITERT_PROBE_SHAPE") else {
        return Ok(None);
    };
    parse_probe_shape(
        value
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid probe shape"))?,
    )
    .map(Some)
}

fn parse_probe_shape(value: &str) -> anyhow::Result<Vec<i32>> {
    let shape = value
        .split(',')
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(
        !shape.is_empty() && shape.len() <= 8 && shape.iter().all(|value| *value > 0),
        "Invalid probe dimensions"
    );
    let size = shape
        .iter()
        .try_fold(1_usize, |size, value| size.checked_mul(*value as usize))
        .ok_or_else(|| anyhow::anyhow!("Probe shape overflow"))?;
    anyhow::ensure!(size <= 16 * 1024 * 1024, "Probe output too large");
    Ok(shape)
}

#[test]
fn probe_shape_bounds_protect_native_reads() {
    assert_eq!(parse_probe_shape("1,1814,1024").unwrap(), [1, 1814, 1024]);
    for invalid in [
        "",
        "0",
        "-1",
        "1,NaN",
        "16777217",
        "1,1,1,1,1,1,1,1,1",
        "2147483647,2147483647,2147483647",
    ] {
        assert!(parse_probe_shape(invalid).is_err(), "accepted {invalid}");
    }
}

fn verify_artifacts(runtime: &Path, model: &Path) -> anyhow::Result<()> {
    let expected = std::env::var("RUNE_LITERT_CANDIDATE_SHA256")?;
    anyhow::ensure!(
        expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Set RUNE_LITERT_CANDIDATE_SHA256 to the conversion report's SHA-256"
    );
    anyhow::ensure!(
        file_sha256(runtime)? == RUNTIME_SHA,
        "Runtime checksum mismatch"
    );
    let directory = runtime
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Missing runtime directory"))?;
    let executable = std::env::current_exe()?;
    let executable_directory = executable
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Missing worker directory"))?;
    for (path, digest) in [
        (
            directory.join("libLiteRtWebGpuAccelerator.dll"),
            "ade56eff38715f3e7b3ada0cfb736db4cccb6bed4668a891a37ebcf4eee38548",
        ),
        (
            executable_directory.join("dxcompiler.dll"),
            "5888d3e590f5bfd8743484e7f31313996734adec5741506271a5c6368cbe8ba9",
        ),
        (
            executable_directory.join("dxil.dll"),
            "0cf5fa4b90f74550166108b3870c883f4a85243976b246b1da3ea90ad5b92015",
        ),
    ] {
        anyhow::ensure!(
            file_sha256(&path)? == digest,
            "GPU component checksum mismatch: {}",
            path.display()
        );
    }
    anyhow::ensure!(
        file_sha256(model)? == expected.to_ascii_lowercase(),
        "Candidate checksum mismatch"
    );
    let mut header = [0; 8];
    std::fs::File::open(model)?.read_exact(&mut header)?;
    anyhow::ensure!(
        &header[4..] == b"TFL3",
        "Candidate is not a TFLite flatbuffer"
    );
    Ok(())
}

fn file_sha256(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        args.len() == 5
            && args[0] == "litert-worker"
            && args[1] == "--runtime"
            && args[3] == "--model",
        "Usage: candidate_worker litert-worker --runtime PINNED_DLL --model CANDIDATE_TFLITE"
    );
    let _ = probe_shape()?; // Reject invalid diagnostic bounds before loading DLLs.
    native::run(Path::new(&args[2]), Path::new(&args[4]))
}
