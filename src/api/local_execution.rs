//! Version 1 local execution constraint. Coverage is deliberately endpoint-specific.
use axum::{
    Json,
    extract::Request,
    http::{Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

pub(super) const HEADER: &str = "x-rune-local-only";

pub(super) fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(crate::types::ErrorResponse {
            error: crate::types::ApiError::new(message)
                .with_type("invalid_request_error")
                .with_code(code),
        }),
    )
        .into_response()
}

pub(super) async fn capabilities() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "local_execution": {
            "version": 1,
            "request_header": HEADER,
            "request_value": "1",
            "endpoints": [{"method": "POST", "path": "/v1/chat/completions"}]
        }
    }))
}

pub(super) async fn enforce(request: Request, next: Next) -> Response {
    let mut values = request.headers().get_all(HEADER).iter();
    if let Some(value) = values.next() {
        if value != "1" || values.next().is_some() {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_local_execution_constraint",
                "x-rune-local-only must occur once with value 1 (contract version 1).",
            );
        }
        if request.method() != Method::POST || request.uri().path() != "/v1/chat/completions" {
            return error(
                StatusCode::BAD_REQUEST,
                "local_execution_unsupported",
                "Local execution contract version 1 covers POST /v1/chat/completions only.",
            );
        }
    }
    next.run(request).await
}
