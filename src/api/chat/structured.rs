use crate::{
    inference::types::InferenceTaskResponse,
    types::{ApiError, ChatCompletionRequest, ErrorResponse},
};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

pub(super) fn validator(
    request: &ChatCompletionRequest,
) -> Result<Option<jsonschema::Validator>, String> {
    let Some(format) = &request.response_format else {
        return Ok(None);
    };
    let schema = match format.r#type.as_str() {
        "json_schema" => format
            .json_schema
            .as_ref()
            .and_then(|s| s.schema.clone())
            .ok_or("response_format.json_schema.schema is required")?,
        "json_object" => json!({"type": "object"}),
        _ => return Ok(None),
    };
    jsonschema::options()
        .should_validate_formats(true)
        .should_ignore_unknown_formats(false)
        .build(&schema)
        .map(Some)
        .map_err(|e| format!("Invalid or unsupported response schema: {e}"))
}

pub(super) fn validate_output(
    validator: &jsonschema::Validator,
    output: &InferenceTaskResponse,
) -> Result<(), String> {
    let InferenceTaskResponse::Text(text) = output else {
        return Err("Structured output requires JSON content; the engine returned a different response type.".into());
    };
    let value: Value = serde_json::from_str(text).map_err(|_| {
        "Engine output is not a complete JSON value matching the requested schema.".to_string()
    })?;
    validator.validate(&value).map_err(|e| {
        format!(
            "Engine output violates the requested schema at {}.",
            e.instance_path()
        )
    })
}

pub(super) async fn validate_remote(
    validator: &jsonschema::Validator,
    response: Response,
) -> Response {
    let (parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, 32 * 1024 * 1024).await else {
        return error(
            StatusCode::BAD_GATEWAY,
            "structured_output_invalid",
            "The provider reply could not be read for schema validation.".into(),
        );
    };
    let content = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|reply| {
            reply
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    let Some(content) = content else {
        return error(
            StatusCode::BAD_GATEWAY,
            "structured_output_invalid",
            "The provider reply has no text content to validate against the requested schema."
                .into(),
        );
    };
    match validate_output(validator, &InferenceTaskResponse::Text(content)) {
        Ok(()) => Response::from_parts(parts, axum::body::Body::from(bytes)),
        Err(message) => error(
            StatusCode::BAD_GATEWAY,
            "structured_output_invalid",
            message,
        ),
    }
}

pub(super) fn error(status: StatusCode, code: &str, message: String) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ApiError::new(message)
                .with_type(if status.is_client_error() {
                    "invalid_request_error"
                } else {
                    "invalid_response_error"
                })
                .with_code(code),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_complete_root_nullable_boxes_and_nested_constraints() {
        let schema = json!({"type":"object","additionalProperties":false,
        "required":["answer","findings"], "properties": {
            "answer":{"type":"string"},
            "findings":{"type":"array","maxItems":1,"items":{
                "type":"object","additionalProperties":false,"required":["box","kind"],
                "properties":{"box":{"type":["array","null"],"minItems":4,"maxItems":4,"items":{"type":"number"}},
                "kind":{"enum":["issue"]}}
            }}
        }});
        let validator = jsonschema::validator_for(&schema).unwrap();
        for value in [
            json!({"answer":"ok","findings":[{"box":null,"kind":"issue"}]}),
            json!({"answer":"ok","findings":[{"box":[0,1,2,3],"kind":"issue"}]}),
        ] {
            assert!(
                validate_output(&validator, &InferenceTaskResponse::Text(value.to_string()))
                    .is_ok()
            );
        }
        for value in [
            json!([]),
            json!([{"answer":"ok","findings":[]}]),
            json!({"findings":[]}),
            json!({"answer":"ok","findings":[{"box":null,"kind":"issue"},{"box":null,"kind":"issue"}]}),
            json!({"answer":"ok","findings":[{"box":[0,1,"two",3],"kind":"issue"}]}),
            json!({"answer":"ok","findings":[{"box":[1,2],"kind":"issue"}]}),
            json!({"answer":"ok","findings":[{"box":null,"kind":"bad"}]}),
            json!({"answer":"ok","findings":[{"kind":"issue"}]}),
            json!({"answer":"ok","findings":[],"extra":true}),
        ] {
            assert!(
                validate_output(&validator, &InferenceTaskResponse::Text(value.to_string()))
                    .is_err()
            );
        }
        for text in ["{", "```json\n{}\n```", "{} {}"] {
            assert!(
                validate_output(&validator, &InferenceTaskResponse::Text(text.into())).is_err()
            );
        }
    }
    #[test]
    fn rejects_external_references_without_fetching() {
        assert!(
            jsonschema::validator_for(&json!({"$ref":"https://example.invalid/schema"})).is_err()
        );
        assert!(jsonschema::validator_for(&json!({"$ref":"file:///secret"})).is_err());
    }
    #[test]
    fn compiles_local_references_and_rejects_bad_schemas_before_inference() {
        let request = |schema| {
            serde_json::from_value::<ChatCompletionRequest>(json!({
            "model":"fixture","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"fixture_schema","schema":schema}}
        })).unwrap()
        };
        let good = request(
            json!({"type":"object","$defs":{"name":{"type":"string","minLength":2}},
            "properties":{"name":{"$ref":"#/$defs/name"}},"required":["name"]}),
        );
        let validator = validator(&good).unwrap().unwrap();
        assert!(validator.is_valid(&json!({"name":"ok"})));
        assert!(!validator.is_valid(&json!({"name":"x"})));
        for schema in [
            json!({"type":"nonsense"}),
            json!({"required":true}),
            json!({"type":"string","format":"unsupported-format"}),
        ] {
            assert!(super::validator(&request(schema)).is_err());
        }
    }
    fn reply(content: &str) -> Response {
        Json(json!({"choices":[{"message":{"role":"assistant","content":content}}]}))
            .into_response()
    }
    async fn status_of(content: &str) -> StatusCode {
        let validator = jsonschema::options()
            .build(&json!({"type":"object","required":["score"],"properties":{"score":{"type":"integer"}}}))
            .unwrap();
        validate_remote(&validator, reply(content)).await.status()
    }
    #[tokio::test]
    async fn remote_replies_are_validated_against_the_requested_schema() {
        assert_eq!(status_of(r#"{"score":3}"#).await, StatusCode::OK);
        for bad in [
            r#"{"score":"x"}"#,
            "{}",
            "```json
{\"score\":3}
```",
            "plain",
        ] {
            assert_eq!(status_of(bad).await, StatusCode::BAD_GATEWAY);
        }
        let validator = jsonschema::options()
            .build(&json!({"type":"object"}))
            .unwrap();
        let empty = Json(json!({"choices":[]})).into_response();
        assert_eq!(
            validate_remote(&validator, empty).await.status(),
            StatusCode::BAD_GATEWAY
        );
    }
}
