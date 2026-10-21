use base64::Engine;

#[test]
fn off_center_fixture_matches_expected_pixels() {
    let image = image::load_from_memory(include_bytes!("fixtures/studio/off-center.png"))
        .unwrap()
        .to_rgb8();
    assert_eq!(image.dimensions(), (400, 240));
    assert_eq!(image.get_pixel(350, 36).0, [220, 30, 30]);
    assert_eq!(image.get_pixel(200, 120).0, [245, 245, 245]);
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/studio/expected.json")).unwrap();
    assert_eq!(expected["point_pixels"], serde_json::json!([350, 36]));
}

/// Opt-in acceptance probe, never a substitute for a labeled calibration dataset.
#[test]
#[ignore = "requires a running server and explicit RUNE_VISUAL_MODEL; no GPU required by CI"]
fn live_decision_and_grounding_acceptance() {
    let base = std::env::var("RUNE_VISUAL_BASE_URL")
        .expect("set RUNE_VISUAL_BASE_URL, e.g. http://127.0.0.1:8080");
    let model = std::env::var("RUNE_VISUAL_MODEL").expect("set RUNE_VISUAL_MODEL");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();
    let image = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD
            .encode(include_bytes!("fixtures/studio/off-center.png"))
    );
    let send = |path: &str, body: serde_json::Value| {
        let mut request = client
            .post(format!("{}{path}", base.trim_end_matches('/')))
            .json(&body);
        if let Ok(key) = std::env::var("RUNE_VISUAL_API_KEY") {
            request = request.bearer_auth(key);
        }
        let response = request.send().unwrap().error_for_status().unwrap();
        assert_eq!(response.headers()["x-model-locality"], "local");
        response.json::<serde_json::Value>().unwrap()
    };
    let decision = send(
        "/v1/decisions",
        serde_json::json!({"model":model,"state":"Inspect the image.","images":[image],"questions":[{"name":"red_visible","type":"noul","instruction":"Is a red rectangle visible in the image?"}]}),
    );
    assert_eq!(decision["answers"]["red_visible"]["decision"], true);
    let probability = decision["answers"]["red_visible"]["probability"]
        .as_f64()
        .unwrap();
    assert!(
        (0.8..=1.0).contains(&probability),
        "acceptance target only: {decision}"
    );
    let grounding = send(
        "/v1/images/grounding",
        serde_json::json!({"model":model,"image":image,"query":"red rectangle"}),
    );
    let point = grounding["point"].as_array().unwrap();
    assert!((800..950).contains(&point[0].as_u64().unwrap()));
    assert!((100..200).contains(&point[1].as_u64().unwrap()));
}
