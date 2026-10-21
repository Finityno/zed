/// Maximum accepted size of an external HTML clipboard representation.
#[doc(hidden)]
pub const MAX_CLIPBOARD_HTML_BYTES: usize = 32 * 1024 * 1024;

/// Encode a UTF-8 HTML fragment using the Windows HTML Format byte offsets.
#[doc(hidden)]
pub fn encode_windows_html_clipboard(fragment: &str) -> Vec<u8> {
    let prefix = "<html><body><!--StartFragment-->";
    let suffix = "<!--EndFragment--></body></html>";
    let header = |start_html, end_html, start_fragment, end_fragment| format!(
        "Version:1.0\r\nStartHTML:{start_html:010}\r\nEndHTML:{end_html:010}\r\nStartFragment:{start_fragment:010}\r\nEndFragment:{end_fragment:010}\r\n"
    );
    let start_html = header(0, 0, 0, 0).len();
    let start_fragment = start_html + prefix.len();
    let end_fragment = start_fragment + fragment.len();
    let end_html = end_fragment + suffix.len();
    let mut result = header(start_html, end_html, start_fragment, end_fragment).into_bytes();
    result.extend_from_slice(prefix.as_bytes());
    result.extend_from_slice(fragment.as_bytes());
    result.extend_from_slice(suffix.as_bytes());
    result.push(0);
    result
}

/// Decode the selected fragment, rejecting invalid offsets and UTF-8 boundaries.
#[doc(hidden)]
pub fn decode_windows_html_clipboard(bytes: &[u8]) -> Option<&str> {
    let header_end = bytes.iter().position(|byte| *byte == b'<').unwrap_or(bytes.len()).min(32 * 1024);
    let header = std::str::from_utf8(bytes.get(..header_end)?).ok()?.trim_start_matches('\u{feff}');
    let mut start_html = None;
    let mut end_html = None;
    let mut start_fragment = None;
    let mut end_fragment = None;
    for line in header.split(['\r', '\n']) {
        let Some((key, value)) = line.split_once(':') else { continue; };
        match key.trim() {
            "StartHTML" => start_html = Some(value.trim().parse::<i64>().ok()?),
            "EndHTML" => end_html = Some(value.trim().parse::<i64>().ok()?),
            "StartFragment" => start_fragment = Some(value.trim().parse::<i64>().ok()?),
            "EndFragment" => end_fragment = Some(value.trim().parse::<i64>().ok()?),
            _ => {},
        }
    }
    match (start_fragment, end_fragment) {
        (Some(start), Some(end)) if start >= 0 && end >= start => {
            let start = usize::try_from(start).ok()?;
            let end = usize::try_from(end).ok()?;
            if end.checked_sub(start)? > MAX_CLIPBOARD_HTML_BYTES { return None; }
            if let Some(context_start) = start_html.filter(|offset| *offset >= 0) {
                if start < usize::try_from(context_start).ok()? { return None; }
            }
            if let Some(context_end) = end_html.filter(|offset| *offset >= 0) {
                let context_end = usize::try_from(context_end).ok()?;
                if end > context_end || context_end > bytes.len() { return None; }
            }
            std::str::from_utf8(bytes.get(start..end)?).ok()
        }
        (None, None) | (Some(-1), Some(-1)) => {
            let source = std::str::from_utf8(bytes).ok()?.trim_end_matches('\0');
            let start = source.find("<!--StartFragment-->")? + "<!--StartFragment-->".len();
            let end = source.get(start..)?.find("<!--EndFragment-->")? + start;
            (end - start <= MAX_CLIPBOARD_HTML_BYTES).then(|| &source[start..end])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClipboardItem, ClipboardString};

    #[test]
    fn html_clipboard_offsets_count_utf8_bytes_and_exclude_context() {
        let fragment = "<p>Grüße 東京 🦀</p>";
        let encoded = encode_windows_html_clipboard(fragment);
        assert_eq!(decode_windows_html_clipboard(&encoded), Some(fragment));
        let header = std::str::from_utf8(&encoded).expect("encoded UTF-8");
        let offset = |name: &str| header.lines().find_map(|line| line.strip_prefix(name)).and_then(|value| value.parse::<usize>().ok()).expect("offset");
        assert_eq!(&encoded[offset("StartFragment:")..offset("EndFragment:")], fragment.as_bytes());
        assert_eq!(encoded[offset("EndHTML:")], 0);
    }

    #[test]
    fn html_clipboard_accepts_missing_context_cr_headers_and_marker_fallback() {
        let encoded = encode_windows_html_clipboard("<b>bold</b>");
        let source = std::str::from_utf8(&encoded).expect("UTF-8");
        let start = source.find("<b>").expect("fragment start");
        let end = start + "<b>bold</b>".len();
        let header = format!("Version:1.0\rStartHTML:-1\rEndHTML:-1\rStartFragment:{start:010}\rEndFragment:{end:010}\r");
        let shift = header.len() as i64 - source.find("<html>").expect("HTML start") as i64;
        let payload = format!("Version:1.0\rStartHTML:-1\rEndHTML:-1\rStartFragment:{:010}\rEndFragment:{:010}\r{}",start as i64 + shift,end as i64 + shift,&source[source.find("<html>").expect("HTML start")..]);
        assert_eq!(decode_windows_html_clipboard(payload.as_bytes()), Some("<b>bold</b>"));
        assert_eq!(decode_windows_html_clipboard(b"Version:0.9\n<html><!--StartFragment--><i>text</i><!--EndFragment--></html>\0"), Some("<i>text</i>"));
    }

    #[test]
    fn html_clipboard_rejects_malformed_or_non_boundary_ranges() {
        assert_eq!(decode_windows_html_clipboard(b"StartFragment:999999\nEndFragment:1000000\n<html></html>"), None);
        assert_eq!(decode_windows_html_clipboard(b"StartFragment:40\nEndFragment:30\n<html></html>"), None);
        assert_eq!(decode_windows_html_clipboard(b"StartFragment:invalid\nEndFragment:invalid\n<html><!--StartFragment-->text<!--EndFragment--></html>"), None);
        assert_eq!(decode_windows_html_clipboard(b"StartFragment:-1\nEndFragment:40\n<html></html>"), None);
        let mut encoded = encode_windows_html_clipboard("é");
        let original = std::str::from_utf8(&encoded).expect("UTF-8");
        let start = original.find('é').expect("fragment");
        let replacement = format!("StartFragment:{:010}",start+1);
        let position = original.find("StartFragment:").expect("header");
        encoded[position..position+replacement.len()].copy_from_slice(replacement.as_bytes());
        assert_eq!(decode_windows_html_clipboard(&encoded), None);
    }

    #[test]
    fn html_clipboard_keeps_plain_text_and_json_metadata() {
        let item = ClipboardItem::new_string_with_json_metadata("plain".into(), vec![1, 2]).with_html("<b>plain</b>");
        assert_eq!(item.text().as_deref(), Some("plain"));
        assert_eq!(item.html(), Some("<b>plain</b>"));
        assert_eq!(item.metadata().map(String::as_str), Some("[1,2]"));
        assert_eq!(item.entries.len(), 1);
        assert_eq!(ClipboardString::new("plain".into()).html, None);
        let replaced = item.with_html("<i>plain</i>");
        assert_eq!(replaced.text().as_deref(), Some("plain"));
        assert_eq!(replaced.metadata().map(String::as_str), Some("[1,2]"));
        assert_eq!(replaced.entries.len(), 1);
        assert_eq!(ClipboardItem { entries: Vec::new() }.with_html("<hr>").html(), Some("<hr>"));
    }
}
