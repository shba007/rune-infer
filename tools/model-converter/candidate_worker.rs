//! Validation-only worker; never linked into Infer's server.
//! Build against Infer's existing sha2 and anyhow artifacts. The caller must
//! set RUNE_LITERT_CANDIDATE_SHA256 to the conversion report's candidate digest.
//! The native ABI implementation is shared with production to prevent drift.
use std::{io::Read, path::Path};

#[path = "../../src/inference/engines/litert/native.rs"]
mod native;

const PIXELS: usize = 518 * 686;
const RUNTIME_SHA: &str = "6fe42cfa518879dd7bae4500244cf371d949bf0c8559cb60be207d81112da16d";

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
    native::run(Path::new(&args[2]), Path::new(&args[4]))
}
