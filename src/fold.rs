//! Case- and diacritic-insensitive text matching, mirroring the index analyzer
//! (lowercase + ASCII folding) so "blad" finds and highlights "Błąd".

/// Lowercase `c` and strip its diacritics. Always returns exactly one char, so
/// char positions in the folded text line up with the original.
pub fn fold_char(c: char) -> char {
    let lower = c.to_lowercase().next().unwrap_or(c);
    match lower {
        'ą' | 'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' | 'ā' => 'a',
        'ć' | 'č' | 'ç' => 'c',
        'ď' => 'd',
        'ę' | 'é' | 'è' | 'ê' | 'ë' | 'ě' | 'ē' => 'e',
        'í' | 'ì' | 'î' | 'ï' | 'ī' => 'i',
        'ł' | 'ľ' | 'ĺ' => 'l',
        'ń' | 'ň' | 'ñ' => 'n',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' | 'ō' => 'o',
        'ř' => 'r',
        'ś' | 'š' => 's',
        'ť' => 't',
        'ú' | 'ù' | 'û' | 'ü' | 'ů' | 'ū' => 'u',
        'ý' | 'ÿ' => 'y',
        'ź' | 'ż' | 'ž' => 'z',
        other => other,
    }
}

/// Fold a whole string with [`fold_char`]
pub fn fold(text: &str) -> String {
    text.chars().map(fold_char).collect()
}

/// Byte ranges in `text` where `needle` occurs, ignoring case and diacritics
pub fn find_folded(text: &str, needle: &str) -> Vec<(usize, usize)> {
    let needle: Vec<char> = needle.chars().map(fold_char).collect();
    if needle.is_empty() {
        return Vec::new();
    }
    let chars: Vec<(usize, char)> = text.char_indices().map(|(i, c)| (i, fold_char(c))).collect();
    let mut matches = Vec::new();
    let mut i = 0;
    while i + needle.len() <= chars.len() {
        if chars[i..i + needle.len()].iter().map(|&(_, c)| c).eq(needle.iter().copied()) {
            let start = chars[i].0;
            let end = chars.get(i + needle.len()).map_or(text.len(), |&(b, _)| b);
            matches.push((start, end));
            i += needle.len();
        } else {
            i += 1;
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_strips_polish_diacritics_and_case() {
        assert_eq!(fold("Błąd Żółć"), "blad zolc");
    }

    #[test]
    fn find_folded_returns_byte_ranges_in_the_original_text() {
        let text = "Mamy BŁĄD i blad";

        let ranges = find_folded(text, "błąd");

        let found: Vec<&str> = ranges.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(found, ["BŁĄD", "blad"]);
    }
}
