//! Q10 (audit_codex_20260428.md W'6): pure JSON parsing helpers extracted
//! from `mcts_server.rs` so they can be reviewed and tested in isolation.
//!
//! These functions implement a tiny string-scan JSON reader used by the
//! mcts_server REPL to pull `"key": value` fragments out of the QIPC
//! control-line frames. They are intentionally not a general-purpose
//! JSON parser — every consumer at the mcts_server boundary already knows
//! the shape it expects and only needs the named field's primitive value.
//!
//! All functions are pure (`&str -> Option<...>` / `Vec<...>`), depend on
//! no engine state, and are independently unit-tested below. Splitting
//! them out reduces the mcts_server.rs review surface by ~80 lines and
//! gives parser-bug regressions their own test home.

/// Extract the string value of `"key": "..."`. Returns the inner str
/// without the quotes; None if the key is missing or the value is not a
/// string literal.
pub(crate) fn jstr<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start(); // skip whitespace after colon
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(&inner[..end])
    } else {
        None
    }
}

/// Extract the integer value of `"key": NN` (signed). Stops at the first
/// non-digit / non-minus character; None if the key is missing or the
/// value is not a parseable integer.
pub(crate) fn jint(s: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract a flat int64 array `"key": [1, 2, 3]`. Returns `vec![]` when
/// the key is missing or the array is malformed; this is the historical
/// behavior the mcts_server callers depend on.
pub(crate) fn jarr(s: &str, key: &str) -> Vec<i64> {
    let pat = format!("\"{}\":[", key);
    if let Some(start) = s.find(&pat) {
        let rest = &s[start + pat.len()..];
        if let Some(end) = rest.find(']') {
            return rest[..end]
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect();
        }
    }
    vec![]
}

/// Extract the float value of `"key": NN.NN`. Stops at the first
/// non-digit / non-minus / non-dot character.
pub(crate) fn jfloat(s: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract the bool value of `"key": true|false`. None if the key is
/// missing or the value is not exactly the JSON literal `true` / `false`.
pub(crate) fn jbool(s: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q10_jstr_round_trips_simple_string_value() {
        assert_eq!(jstr(r#"{"k": "v"}"#, "k"), Some("v"));
        assert_eq!(jstr(r#"{"k":"v"}"#, "k"), Some("v"));
        assert_eq!(jstr(r#"{"a":1,"k":"v"}"#, "k"), Some("v"));
        // Missing key.
        assert_eq!(jstr(r#"{"a":"b"}"#, "k"), None);
        // Non-string value falls through to None (caller uses jint/jfloat/jbool).
        assert_eq!(jstr(r#"{"k":42}"#, "k"), None);
    }

    #[test]
    fn test_q10_jint_signed_and_missing() {
        assert_eq!(jint(r#"{"k": 42}"#, "k"), Some(42));
        assert_eq!(jint(r#"{"k":-7}"#, "k"), Some(-7));
        assert_eq!(jint(r#"{"k": 0,"a":"b"}"#, "k"), Some(0));
        assert_eq!(jint(r#"{"a":1}"#, "k"), None);
    }

    #[test]
    fn test_q10_jfloat_with_decimal_and_negative() {
        let parsed = jfloat(r#"{"k": 3.14}"#, "k").unwrap();
        assert!((parsed - 3.14).abs() < 1e-9);
        let parsed = jfloat(r#"{"k": -0.5}"#, "k").unwrap();
        assert!((parsed - (-0.5)).abs() < 1e-9);
        assert_eq!(jfloat(r#"{}"#, "k"), None);
    }

    #[test]
    fn test_q10_jbool_strict_literal_match() {
        assert_eq!(jbool(r#"{"k": true}"#, "k"), Some(true));
        assert_eq!(jbool(r#"{"k": false}"#, "k"), Some(false));
        // Anything other than the literal returns None — historical behavior.
        assert_eq!(jbool(r#"{"k": 1}"#, "k"), None);
        assert_eq!(jbool(r#"{}"#, "k"), None);
    }

    #[test]
    fn test_q10_jarr_returns_empty_for_missing_or_malformed() {
        assert_eq!(jarr(r#"{"k":[1,2,3]}"#, "k"), vec![1, 2, 3]);
        assert_eq!(jarr(r#"{"k":[ 7 ,  -1 ]}"#, "k"), vec![7, -1]);
        // Missing → empty.
        assert_eq!(jarr(r#"{"a":[1]}"#, "k"), Vec::<i64>::new());
        // Truncated bracket → empty.
        assert_eq!(jarr(r#"{"k":[1,2"#, "k"), Vec::<i64>::new());
    }
}
