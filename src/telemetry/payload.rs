//! Complete API body capture in a separate JSONL file. Bodies are copied as
//! they pass, without collecting a request or an SSE response in memory.
use axum::body::HttpBody;
use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use base64::Engine;
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use tokio_stream::StreamExt;

static LOG: OnceLock<PayloadLog> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct PayloadLog(Arc<Writer>);
struct Writer {
    file: Mutex<File>,
    session: String,
    failed: AtomicBool,
}
impl PayloadLog {
    fn open(path: &Path) -> io::Result<Self> {
        if std::fs::symlink_metadata(path).is_ok_and(|metadata| !metadata.file_type().is_file()) {
            return Err(io::Error::other("Payload log must be a regular file"));
        }
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        Ok(Self(Arc::new(Writer {
            file: Mutex::new(options.open(path)?),
            session: format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            failed: AtomicBool::new(false),
        })))
    }
    fn record(&self, id: u64, direction: &str, event: &str, fields: serde_json::Value) {
        let mut record = fields;
        record["session"] = self.0.session.clone().into();
        record["request_id"] = id.into();
        record["direction"] = direction.into();
        record["event"] = event.into();
        record["timestamp"] = chrono::Utc::now().to_rfc3339().into();
        let result = (|| -> io::Result<()> {
            let mut bytes = serde_json::to_vec(&record)?;
            bytes.push(b'\n');
            let mut file = self
                .0
                .file
                .lock()
                .map_err(|_| io::Error::other("Payload log lock poisoned"))?;
            file.write_all(&bytes)
        })();
        if let Err(error) = result {
            if !self.0.failed.swap(true, Ordering::Relaxed) {
                tracing::error!(%error, "Request/response payload log write failed; payload capture is incomplete");
            }
        }
    }
}

/// Enable complete payload logging for CLI HTTP routers. Embedders opt in by
/// calling this once; library users otherwise create no payload log files.
pub fn init(path: impl AsRef<Path>) -> io::Result<()> {
    let log = PayloadLog::open(path.as_ref())?;
    LOG.set(log).map_err(|_| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "Payload logging already initialized",
        )
    })
}

struct Capture {
    log: PayloadLog,
    id: u64,
    direction: &'static str,
    offset: u64,
    finished: bool,
}
impl Capture {
    fn finish(&mut self, complete: bool) {
        if !self.finished {
            self.finished = true;
            self.log.record(
                self.id,
                self.direction,
                "end",
                serde_json::json!({"bytes": self.offset, "complete": complete}),
            );
        }
    }
    fn body(mut self, body: Body) -> Body {
        if body.size_hint().exact() == Some(0) {
            self.finish(true);
            return body;
        }
        Body::from_stream(async_stream::stream! {
            let mut stream = body.into_data_stream();
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        // Bound each serialization allocation even for a large body frame.
                        for piece in bytes.chunks(64 * 1024) {
                            let (encoding, data) = match std::str::from_utf8(piece) {
                                Ok(text) => ("utf8", text.to_owned()),
                                Err(_) => ("base64", base64::engine::general_purpose::STANDARD.encode(piece)),
                            };
                            self.log.record(self.id, self.direction, "body", serde_json::json!({"offset":self.offset,"encoding":encoding,"data":data}));
                            self.offset += piece.len() as u64;
                        }
                        yield Ok::<_, axum::Error>(bytes);
                    }
                    Err(error) => { self.finish(false); yield Err(error); return; }
                }
            }
            self.finish(true);
        })
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        self.finish(false);
    }
}

pub(crate) async fn middleware(request: Request, next: Next) -> Response {
    match LOG.get() {
        Some(log) => capture(log.clone(), request, next).await,
        None => next.run(request).await,
    }
}

async fn capture(log: PayloadLog, request: Request, next: Next) -> Response {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    log.record(id, "request", "start", serde_json::json!({"method":request.method().as_str(),"path":request.uri().path(),"content_type":request.headers().get("content-type").and_then(|value|value.to_str().ok())}));
    let (parts, body) = request.into_parts();
    let request = Request::from_parts(
        parts,
        Capture {
            log: log.clone(),
            id,
            direction: "request",
            offset: 0,
            finished: false,
        }
        .body(body),
    );
    let response = next.run(request).await;
    log.record(id, "response", "start", serde_json::json!({"status":response.status().as_u16(),"content_type":response.headers().get("content-type").and_then(|value|value.to_str().ok())}));
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Capture {
            log,
            id,
            direction: "response",
            offset: 0,
            finished: false,
        }
        .body(body),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Bytes, to_bytes},
        routing::post,
    };
    use tower::ServiceExt;
    fn fixture() -> (std::path::PathBuf, PayloadLog) {
        let path = std::env::temp_dir().join(format!(
            "rune-payload-{}.log",
            crate::setup::generate_api_key()
        ));
        let log = PayloadLog::open(&path).unwrap();
        (path, log)
    }
    fn records(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    fn body(records: &[serde_json::Value], direction: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        for entry in records
            .iter()
            .filter(|r| r["direction"] == direction && r["event"] == "body")
        {
            assert_eq!(entry["offset"].as_u64().unwrap(), bytes.len() as u64);
            let data = entry["data"].as_str().unwrap();
            if entry["encoding"] == "utf8" {
                bytes.extend_from_slice(data.as_bytes());
            } else {
                bytes.extend(
                    base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .unwrap(),
                );
            }
        }
        bytes
    }
    #[tokio::test]
    async fn complete_json_is_logged_without_truncation_or_authentication_headers() {
        let (path, log) = fixture();
        let app = Router::new()
            .route("/chat", post(|bytes: Bytes| async move { bytes }))
            .layer(axum::middleware::from_fn(move |req, next| {
                let log = log.clone();
                async move { capture(log, req, next).await }
            }));
        let payload=serde_json::json!({"tools":[{"name":"plan_task_graph","arguments":{"text":"नेपाली text ".repeat(10000)}}]}).to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chat?api_key=hidden-query-secret")
                    .header("authorization", "Bearer hidden-header-secret")
                    .header("cookie", "hidden-cookie")
                    .body(Body::from(payload.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .as_ref(),
            payload.as_bytes()
        );
        let entries = records(&path);
        assert_eq!(body(&entries, "request"), payload.as_bytes());
        assert_eq!(body(&entries, "response"), payload.as_bytes());
        assert_eq!(
            entries
                .iter()
                .filter(|r| r["event"] == "end" && r["complete"] == true)
                .count(),
            2
        );
        assert!(
            entries
                .iter()
                .all(|r| r["request_id"] == entries[0]["request_id"])
        );
        assert!(!std::fs::read_to_string(&path).unwrap().contains("hidden-"));
        std::fs::remove_file(path).unwrap();
    }
    #[tokio::test]
    async fn sse_and_binary_bodies_are_forwarded_and_logged_byte_for_byte() {
        let (path, log) = fixture();
        let frames = vec![
            Bytes::from_static(b"data: {\"delta\":\"<tool"),
            Bytes::from_static(b"_call>\"}\n\ndata: [DONE]\n\n"),
        ];
        let expected = frames.concat();
        let app = Router::new()
            .route(
                "/stream",
                post(move |_: Bytes| {
                    let frames = frames.clone();
                    async move {
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from_stream(tokio_stream::iter(
                                frames.into_iter().map(Ok::<_, io::Error>),
                            )))
                            .unwrap()
                    }
                }),
            )
            .layer(axum::middleware::from_fn(move |req, next| {
                let log = log.clone();
                async move { capture(log, req, next).await }
            }));
        let input = vec![0, 255, 195, 128, 42];
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/stream")
                    .body(Body::from(input.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .as_ref(),
            expected
        );
        let entries = records(&path);
        assert_eq!(body(&entries, "request"), input);
        assert_eq!(body(&entries, "response"), expected);
        std::fs::remove_file(path).unwrap();
    }
    #[tokio::test]
    async fn concurrent_exchanges_keep_distinct_correlated_bodies() {
        let (path, log) = fixture();
        let app = Router::new()
            .route(
                "/echo",
                post(|bytes: Bytes| async move {
                    tokio::task::yield_now().await;
                    bytes
                }),
            )
            .layer(axum::middleware::from_fn(move |req, next| {
                let log = log.clone();
                async move { capture(log, req, next).await }
            }));
        let send = |app: Router, text: &'static str| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/echo")
                        .body(Body::from(text))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(to_bytes(response.into_body(), 1024).await.unwrap(), text);
        };
        tokio::join!(
            send(app.clone(), "first exchange"),
            send(app, "second exchange")
        );
        let entries = records(&path);
        let ids: std::collections::BTreeSet<_> = entries
            .iter()
            .map(|r| r["request_id"].as_u64().unwrap())
            .collect();
        assert_eq!(ids.len(), 2);
        for id in ids {
            let exchange: Vec<_> = entries
                .iter()
                .filter(|r| r["request_id"] == id)
                .cloned()
                .collect();
            assert_eq!(body(&exchange, "request"), body(&exchange, "response"));
            assert_eq!(
                exchange
                    .iter()
                    .filter(|r| r["event"] == "end" && r["complete"] == true)
                    .count(),
                2
            );
        }
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn dropped_stream_is_marked_incomplete_instead_of_claiming_a_full_response() {
        let (path, log) = fixture();
        let app = Router::new()
            .route(
                "/stream",
                post(|_: Bytes| async {
                    Response::new(Body::from_stream(async_stream::stream! {
                        yield Ok::<_,io::Error>(Bytes::from_static(b"first"));
                        std::future::pending::<()>().await;
                    }))
                }),
            )
            .layer(axum::middleware::from_fn(move |req, next| {
                let log = log.clone();
                async move { capture(log, req, next).await }
            }));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut stream = response.into_body().into_data_stream();
        assert_eq!(stream.next().await.unwrap().unwrap(), "first");
        drop(stream);
        let entries = records(&path);
        assert_eq!(body(&entries, "response"), b"first");
        assert!(entries.iter().any(|r| r["direction"] == "response"
            && r["event"] == "end"
            && r["complete"] == false));
        std::fs::remove_file(path).unwrap();
    }
}
