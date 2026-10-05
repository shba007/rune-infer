use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore, SemaphorePermit};

use crate::config::RateLimitConfig;
use crate::types::ApiError;

pub struct PermitGuard<'a> {
    _permit: SemaphorePermit<'a>,
}

pub struct RemoteModelController {
    pub model_id: String,
    semaphore: Arc<Semaphore>,
    rpm: u32,
    queue_timeout: Duration,
    request_timestamps: Mutex<VecDeque<Instant>>,
    cooldown_until: Mutex<Option<Instant>>,
}

impl RemoteModelController {
    pub fn new(model_id: String, config: Option<&RateLimitConfig>) -> Self {
        let default_cfg = RateLimitConfig::default();
        let cfg = config.unwrap_or(&default_cfg);

        Self {
            model_id,
            semaphore: Arc::new(Semaphore::new(cfg.max_concurrent.max(1))),
            rpm: cfg.requests_per_minute.max(1),
            queue_timeout: Duration::from_secs(cfg.queue_timeout_seconds.max(1)),
            request_timestamps: Mutex::new(VecDeque::new()),
            cooldown_until: Mutex::new(None),
        }
    }

    /// Acquires permission to execute a request.
    /// If capacity is constrained, queues asynchronously until a slot opens or timeout is reached.
    pub async fn acquire_permit(&self) -> Result<PermitGuard<'_>, ApiError> {
        let deadline = Instant::now() + self.queue_timeout;

        // 1. Reactive backoff check: wait if upstream enforced a 429 cooldown
        loop {
            let cooldown_wait = {
                let guard = self.cooldown_until.lock().await;
                guard.and_then(|target| {
                    if target > Instant::now() {
                        Some(target - Instant::now())
                    } else {
                        None
                    }
                })
            };

            if let Some(wait_dur) = cooldown_wait {
                if Instant::now() + wait_dur > deadline {
                    return Err(ApiError::new(format!(
                        "Model '{}' is in cooldown after upstream rate limit (429). Queue timeout exceeded.",
                        self.model_id
                    ))
                    .with_type("requests")
                    .with_code("rate_limit_exceeded"));
                }
                tokio::time::sleep(wait_dur).await;
            } else {
                break;
            }
        }

        // 2. Concurrency check: wait for a free semaphore permit
        let time_left = deadline.saturating_duration_since(Instant::now());
        if time_left.is_zero() {
            return Err(ApiError::new(format!(
                "Queue timeout exceeded waiting for concurrency permit on model '{}'",
                self.model_id
            ))
            .with_type("requests")
            .with_code("rate_limit_exceeded"));
        }

        let permit = match tokio::time::timeout(time_left, self.semaphore.acquire()).await {
            Ok(Ok(p)) => p,
            _ => {
                return Err(ApiError::new(format!(
                    "Model '{}' queue capacity reached. Max concurrent requests ({}) saturated.",
                    self.model_id,
                    self.semaphore.available_permits()
                ))
                .with_type("requests")
                .with_code("rate_limit_exceeded"));
            }
        };

        // 3. Sliding-window RPM check: ensure we do not exceed requests per minute
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(ApiError::new(format!(
                    "Queue timeout exceeded waiting for RPM slot on model '{}'",
                    self.model_id
                ))
                .with_type("requests")
                .with_code("rate_limit_exceeded"));
            }

            let sleep_delay = {
                let mut timestamps = self.request_timestamps.lock().await;
                // Evict timestamps older than 60 seconds
                while let Some(&oldest) = timestamps.front() {
                    if now.duration_since(oldest) >= Duration::from_secs(60) {
                        timestamps.pop_front();
                    } else {
                        break;
                    }
                }

                if timestamps.len() < self.rpm as usize {
                    timestamps.push_back(now);
                    None
                } else if let Some(&oldest) = timestamps.front() {
                    let elapsed = now.duration_since(oldest);
                    Some(
                        Duration::from_secs(60).saturating_sub(elapsed) + Duration::from_millis(10),
                    )
                } else {
                    None
                }
            };

            if let Some(delay) = sleep_delay {
                if Instant::now() + delay > deadline {
                    return Err(ApiError::new(format!(
                        "Model '{}' rate limit exceeded ({} req/min). Queue wait would exceed {}s timeout.",
                        self.model_id,
                        self.rpm,
                        self.queue_timeout.as_secs()
                    ))
                    .with_type("requests")
                    .with_code("rate_limit_exceeded"));
                }
                tokio::time::sleep(delay).await;
            } else {
                break;
            }
        }

        Ok(PermitGuard { _permit: permit })
    }

    /// Records an upstream 429 cooldown period.
    pub async fn record_rate_limit_backoff(&self, retry_after: Option<Duration>) {
        let backoff = retry_after.unwrap_or(Duration::from_secs(10));
        let mut guard = self.cooldown_until.lock().await;
        *guard = Some(Instant::now() + backoff);
        tracing::warn!(
            model = %self.model_id,
            backoff_secs = backoff.as_secs(),
            "Upstream rate limit (429) received; throttling queue"
        );
    }
}
