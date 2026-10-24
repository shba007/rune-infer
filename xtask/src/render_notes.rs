//! Fixed-version formatting boundary: all Markdown is produced by Cocogitto.
#[path = "cocogitto.rs"]
#[allow(dead_code)]
mod cocogitto;
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let input: serde_json::Value = serde_json::from_reader(std::io::stdin())?;
    let commits: Vec<cocogitto::Commit> = serde_json::from_value(input["commits"].clone())?;
    let text = cocogitto::render_at(
        input["base"].as_str(),
        input["version"].as_str().ok_or("Missing version")?,
        &commits,
        input["timestamp"].as_i64().ok_or("Missing timestamp")?,
        input["package"].as_str(),
    )?;
    print!("{text}");
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
