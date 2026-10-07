/// Tool call parsing and extraction utilities for llama-server.
pub fn parse_xml_tool_call(block: &str) -> Option<serde_json::Value> {
    let fn_start = block.find("<function=")?;
    let after_fn = &block[fn_start + "<function=".len()..];
    let fn_end = after_fn.find('>')?;
    let fn_name = after_fn[..fn_end]
        .trim()
        .trim_matches('"')
        .trim_matches('\'');
    if fn_name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    let mut cursor = &after_fn[fn_end + 1..];

    while let Some(p_start) = cursor.find("<parameter=") {
        let after_p = &cursor[p_start + "<parameter=".len()..];
        let p_end = after_p.find('>')?;
        let param_name = after_p[..p_end]
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        let val_start = &after_p[p_end + 1..];

        let val_end = val_start
            .find("<parameter=")
            .or_else(|| val_start.find("</parameter>"))
            .or_else(|| val_start.find("</function>"))
            .unwrap_or(val_start.len());

        let raw_val = val_start[..val_end].trim();
        let val = serde_json::from_str::<serde_json::Value>(raw_val)
            .unwrap_or_else(|_| serde_json::Value::String(raw_val.to_string()));

        if !param_name.is_empty() {
            args.insert(param_name, val);
        }

        cursor = &val_start[val_end..];
        if let Some(stripped) = cursor.strip_prefix("</parameter>") {
            cursor = stripped;
        }
    }

    Some(serde_json::json!({
        "name": fn_name,
        "arguments": serde_json::Value::Object(args)
    }))
}

pub fn extract_tool_calls(raw: &str) -> Option<(Option<String>, serde_json::Value)> {
    if raw.contains("<tool_call>") {
        let mut calls = Vec::new();
        let mut remaining = raw;
        let mut text_parts = Vec::new();

        while let Some(start) = remaining.find("<tool_call>") {
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim());
            }
            let after_start = &remaining[start + "<tool_call>".len()..];
            let (block, next_rem) = match after_start.find("</tool_call>") {
                Some(end) => (
                    &after_start[..end],
                    &after_start[end + "</tool_call>".len()..],
                ),
                None => (after_start, ""),
            };
            remaining = next_rem;
            let block = block.trim();

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(block) {
                if let Some(arr) = parsed.as_array() {
                    calls.extend(arr.clone());
                } else if parsed.is_object() {
                    calls.push(parsed);
                }
                continue;
            }

            if let Some(call) = parse_xml_tool_call(block) {
                calls.push(call);
            }
        }

        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim());
        }

        if !calls.is_empty() {
            let clean_text = text_parts.join("\n\n").trim().to_string();
            let content_opt = if clean_text.is_empty() {
                None
            } else {
                Some(clean_text)
            };
            return Some((content_opt, serde_json::Value::Array(calls)));
        }
    }

    let mut cleaned = raw;
    if let Some(think_end) = raw.find("</think>") {
        cleaned = &raw[think_end + "</think>".len()..];
    }

    if let Some(val) = cleaned
        .find('[')
        .zip(cleaned.rfind(']'))
        .filter(|&(start, end)| start < end)
        .and_then(|(start, end)| {
            serde_json::from_str::<serde_json::Value>(&cleaned[start..=end]).ok()
        })
        .filter(|v| v.is_array())
    {
        return Some((None, val));
    }

    if let Some(val) = cleaned
        .find('{')
        .zip(cleaned.rfind('}'))
        .filter(|&(start, end)| start < end)
        .and_then(|(start, end)| {
            serde_json::from_str::<serde_json::Value>(&cleaned[start..=end]).ok()
        })
        .filter(|v| v.is_object())
    {
        return Some((None, serde_json::Value::Array(vec![val])));
    }

    None
}
