use anyhow::{Result, ensure};
use reqwest::{StatusCode, blocking::Client};
use std::io::Read;

pub(crate) const MODELS: &str = include_str!("../registry/models.v2.json");
pub(crate) const ENGINES: &str = include_str!("../registry/engines.v2.json");

pub(crate) fn supports_version(version: u32) -> bool {
    matches!(version, 1 | 2)
}

pub(crate) fn fetch(
    client: &Client,
    current: &str,
    legacy: &str,
    max_bytes: u64,
) -> Result<Vec<u8>> {
    let response = client.get(current).send()?;
    let response = if response.status() == StatusCode::NOT_FOUND {
        client.get(legacy).send()?
    } else {
        response
    };
    let mut body = Vec::new();
    response
        .error_for_status()?
        .take(max_bytes + 1)
        .read_to_end(&mut body)?;
    ensure!(
        body.len() as u64 <= max_bytes,
        "Registry exceeds size limit"
    );
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        time::Duration,
    };

    #[test]
    fn v1_documents_preserve_the_public_entries_from_v2() {
        for (name, legacy, current) in [
            ("models", include_str!("../registry/models.json"), MODELS),
            ("engines", include_str!("../registry/engines.json"), ENGINES),
        ] {
            let legacy: Value = serde_json::from_str(legacy).unwrap();
            let mut projected: Value = serde_json::from_str(current).unwrap();
            assert_eq!(projected["schema_version"], 2);
            assert!(projected["extensions"]["onnx"].is_object());
            projected.as_object_mut().unwrap().remove("extensions");
            projected["schema_version"] = json!(1);
            assert!(
                legacy == projected,
                "the v1 {name} projection is out of sync"
            );
        }
    }

    #[test]
    fn fetch_uses_legacy_only_when_current_is_missing() {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        for (status, expected) in [(200, Some("v2")), (404, Some("v1")), (500, None)] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let worker = std::thread::spawn(move || {
                let serve = |path: &str, status: u16, body: &str| {
                    let (mut stream, _) = listener.accept().unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut reader = BufReader::new(&mut stream);
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                    assert_eq!(line.split_whitespace().nth(1), Some(path));
                    loop {
                        line.clear();
                        assert!(reader.read_line(&mut line).unwrap() > 0);
                        if line == "\r\n" {
                            break;
                        }
                    }
                    write!(stream, "HTTP/1.1 {status} Response\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                };
                serve("/current", status, "v2");
                if status == 404 {
                    serve("/legacy", 200, "v1");
                }
            });
            let result = fetch(
                &client,
                &format!("{base}/current"),
                &format!("{base}/legacy"),
                16,
            );
            match expected {
                Some(body) => assert_eq!(result.unwrap(), body.as_bytes()),
                None => assert_eq!(
                    result
                        .unwrap_err()
                        .downcast_ref::<reqwest::Error>()
                        .unwrap()
                        .status(),
                    Some(StatusCode::INTERNAL_SERVER_ERROR)
                ),
            }
            worker.join().unwrap();
        }
    }
}
