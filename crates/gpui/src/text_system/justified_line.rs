use std::{collections::{BTreeMap, BTreeSet}, ops::Deref, sync::Arc};

use crate::{App, Bounds, LineLayout, Pixels, Point, Result, TextAlign, Window, WrapBoundary, WrappedLine, WrappedLineLayout, point};

/// A wrapped line with shared glyph, decoration, caret and hit-test justification geometry.
#[derive(Clone, Debug)]
pub struct JustifiedLine {
    line: WrappedLine,
    rows: Arc<Vec<JustifiedRow>>,
}

#[derive(Clone, Debug)]
struct JustifiedRow {
    end_index: usize,
    width: Pixels,
    carets: Vec<(usize, Pixels)>,
}

impl WrappedLine {
    /// Expand ordinary inter-word spaces in soft-wrapped rows without reshaping text.
    ///
    /// Final rows, hard-break rows and nonbreaking spaces remain ragged. Glyph clusters,
    /// source byte indices, font choices, wrapping and decorations are retained.
    pub fn justify(&self, width: Pixels) -> JustifiedLine {
        let original = || JustifiedLine { line: self.clone(), rows: Arc::default() };
        if !width.0.is_finite() || width <= Pixels::ZERO || self.wrap_boundaries.is_empty() {
            return original();
        }
        let flat: Vec<_> = self.unwrapped_layout.runs.iter().enumerate().flat_map(|(run_index, run)| {
            run.glyphs.iter().enumerate().map(move |(glyph_index, glyph)| (WrapBoundary { run_ix: run_index, glyph_ix: glyph_index }, glyph))
        }).collect();
        let boundaries: BTreeMap<_, _> = flat.iter().enumerate().map(|(index, (boundary, _))| (*boundary, index)).collect();
        let mut starts = vec![0];
        for boundary in self.wrap_boundaries() {
            let Some(index) = boundaries.get(boundary).copied() else { return original(); };
            if index <= *starts.last().unwrap_or(&0) { return original(); }
            starts.push(index);
        }
        starts.push(flat.len());
        let indices: Vec<_> = flat.iter().map(|(_, glyph)| glyph.index).chain([self.len()]).collect::<BTreeSet<_>>().into_iter().collect();
        let ends: BTreeMap<_, _> = indices.windows(2).map(|pair| (pair[0], pair[1])).collect();
        let mut runs = self.unwrapped_layout.runs.clone();
        let mut shift = Pixels::ZERO;
        let mut changed = false;
        for (row, range) in starts.windows(2).enumerate() {
            let start_x = if row == 0 { Pixels::ZERO } else { flat[range[0]].1.position.x };
            let end_x = flat.get(range[1]).map_or(self.unwrapped_layout.width, |(_, glyph)| glyph.position.x);
            let mut clusters = BTreeMap::<usize, Pixels>::new();
            let mut hard_break = false;
            for (_, glyph) in &flat[range[0]..range[1]] {
                clusters.entry(glyph.index).or_insert(glyph.position.x);
                if let Some(cluster) = ends.get(&glyph.index).and_then(|end| self.text.get(glyph.index..*end)) {
                    hard_break |= cluster.contains(['\n', '\r', '\u{2028}', '\u{2029}']);
                }
            }
            let cluster_positions = clusters;
            let mut clusters: Vec<_> = cluster_positions.iter().map(|(&index, &position)| {
                let space = ends.get(&index).and_then(|end| self.text.get(index..*end)).is_some_and(|cluster| !cluster.is_empty() && cluster.bytes().all(|byte| byte == b' '))
                    && self.text.get(..index).and_then(|text| text.chars().next_back()) != Some('\u{2060}')
                    && ends.get(&index).and_then(|end| self.text.get(*end..)).and_then(|text| text.chars().next()) != Some('\u{2060}');
                (position, space)
            }).collect();
            clusters.sort_by(|left, right| left.0.0.total_cmp(&right.0.0));
            let first_word = clusters.iter().position(|(_, space)| !space);
            let last_word = clusters.iter().rposition(|(_, space)| !space);
            let mut opportunities = Vec::new();
            let mut visible_end = end_x;
            if row + 2 < starts.len() && !hard_break {
                if let (Some(first_word), Some(last_word)) = (first_word, last_word) {
                    if let Some((position, _)) = clusters.get(last_word + 1) { visible_end = *position; }
                    for pair in clusters[first_word..=last_word].windows(2) {
                        if pair[0].1 && pair[1].0 > pair[0].0 { opportunities.push(pair[1].0); }
                    }
                }
            }
            let extra = width - (visible_end - start_x);
            let expand = !opportunities.is_empty() && extra > Pixels::ZERO;
            for (boundary, glyph) in &flat[range[0]..range[1]] {
                let anchor = cluster_positions.get(&glyph.index).copied().unwrap_or(glyph.position.x);
                // Every glyph of a cluster receives the same shift, including overhanging marks.
                let addition = if !expand { Pixels::ZERO }
                    else if anchor >= visible_end { width - (anchor - start_x) }
                    else { extra * (opportunities.partition_point(|position| *position <= anchor) as f32 / opportunities.len() as f32) };
                runs[boundary.run_ix].glyphs[boundary.glyph_ix].position.x = glyph.position.x + shift + addition;
            }
            if expand {
                shift += width - (end_x - start_x);
                changed = true;
            }
        }
        if !changed { return original(); }
        let layout = Arc::new(WrappedLineLayout {
            unwrapped_layout: Arc::new(LineLayout {
                font_size: self.font_size(), width: self.unwrapped_layout.width + shift,
                ascent: self.ascent(), descent: self.descent(), runs, len: self.len(),
            }),
            wrap_boundaries: self.wrap_boundaries.clone(), wrap_width: Some(width),
        });
        let line = WrappedLine { layout, text: self.text.clone(), decoration_runs: self.decoration_runs.clone(), cache_key: None };
        let adjusted: Vec<_> = line.unwrapped_layout.runs.iter().flat_map(|run| run.glyphs.iter()).collect();
        let mut rows = Vec::new();
        for (row, range) in starts.windows(2).enumerate() {
            let start = if row == 0 { Pixels::ZERO } else { adjusted[range[0]].position.x };
            let end = adjusted.get(range[1]).map_or(line.unwrapped_layout.width, |glyph| glyph.position.x);
            let end_index = adjusted.get(range[1]).map_or(line.len(), |glyph| glyph.index);
            let mut carets = BTreeMap::<usize, Pixels>::new();
            for glyph in &adjusted[range[0]..range[1]] {
                carets.entry(glyph.index).or_insert(glyph.position.x - start);
            }
            carets.insert(end_index, end - start);
            let mut carets: Vec<_> = carets.into_iter().collect();
            carets.sort_by(|left, right| left.1.0.total_cmp(&right.1.0).then(left.0.cmp(&right.0)));
            rows.push(JustifiedRow { end_index, width: end - start, carets });
        }
        JustifiedLine { line, rows: Arc::new(rows) }
    }
}

impl Deref for JustifiedLine {
    type Target = WrappedLine;
    fn deref(&self) -> &Self::Target { &self.line }
}

impl JustifiedLine {
    /// Paint glyphs and text decorations using the justified geometry.
    pub fn paint(&self, origin: Point<Pixels>, line_height: Pixels, bounds: Option<Bounds<Pixels>>, window: &mut Window, cx: &mut App) -> Result<()> {
        self.line.paint(origin, line_height, TextAlign::Left, bounds, window, cx)
    }

    /// Paint highlight backgrounds using the same geometry as the glyphs.
    pub fn paint_background(&self, origin: Point<Pixels>, line_height: Pixels, bounds: Option<Bounds<Pixels>>, window: &mut Window, cx: &mut App) -> Result<()> {
        self.line.paint_background(origin, line_height, TextAlign::Left, bounds, window, cx)
    }

    /// Return a shaped-cluster caret position for a UTF-8 source byte index.
    /// Interior ligature bytes snap to the next shaped source boundary.
    pub fn position_for_index(&self, index: usize, line_height: Pixels) -> Option<Point<Pixels>> {
        if self.rows.is_empty() { return self.line.position_for_index(index, line_height); }
        if index > self.len() { return None; }
        self.rows.iter().enumerate().flat_map(|(row, geometry)| geometry.carets.iter().map(move |(boundary, x)| (*boundary, row, *x)))
            .filter(|(boundary, _, _)| *boundary >= index)
            .min_by_key(|(boundary, row, _)| (*boundary, *row))
            .map(|(_, row, x)| point(x, line_height * row as f32))
    }

    /// Return the source cluster whose visual interval contains this position.
    pub fn index_for_position(&self, position: Point<Pixels>, line_height: Pixels) -> std::result::Result<usize, usize> {
        self.index_at(position, line_height, false)
    }

    /// Return the nearest source-cluster boundary in visual coordinates.
    pub fn closest_index_for_position(&self, position: Point<Pixels>, line_height: Pixels) -> std::result::Result<usize, usize> {
        self.index_at(position, line_height, true)
    }

    fn index_at(&self, position: Point<Pixels>, line_height: Pixels, closest: bool) -> std::result::Result<usize, usize> {
        if self.rows.is_empty() {
            return if closest { self.line.closest_index_for_position(position, line_height) } else { self.line.index_for_position(position, line_height) };
        }
        if line_height <= Pixels::ZERO || !position.x.0.is_finite() || !position.y.0.is_finite() { return Err(0); }
        let row = (position.y / line_height).max(0.0) as usize;
        let Some(row) = self.rows.get(row) else { return Err(0); };
        if position.x >= row.width { return Err(row.end_index); }
        let Some(first) = row.carets.first() else { return Err(row.end_index); };
        if position.x < first.1 { return Err(first.0); }
        let right = row.carets.partition_point(|(_, x)| *x <= position.x);
        let left = right.saturating_sub(1);
        let Some(&(index, x)) = row.carets.get(left) else { return Err(row.end_index); };
        if closest {
            if let Some(&(next, next_x)) = row.carets.get(right) {
                if next_x - position.x < position.x - x { return Ok(next); }
            }
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecorationRun, FontId, GlyphId, ShapedGlyph, ShapedRun, UnderlineStyle, black, px};

    fn line(text: &str, glyphs: &[(usize, f32)], wraps: &[usize], width: f32) -> WrappedLine {
        WrappedLine {
            layout: Arc::new(WrappedLineLayout {
                unwrapped_layout: Arc::new(LineLayout {
                    font_size: px(16.0), width: px(width), ascent: px(12.0), descent: px(4.0), len: text.len(),
                    runs: vec![ShapedRun { font_id: FontId(0), glyphs: glyphs.iter().enumerate().map(|(index, &(source, x))| ShapedGlyph { id: GlyphId(index as u32), position: point(px(x), Pixels::ZERO), index: source, is_emoji: false }).collect() }],
                }),
                wrap_boundaries: wraps.iter().map(|&glyph_ix| WrapBoundary { run_ix: 0, glyph_ix }).collect(), wrap_width: Some(px(60.0)),
            }),
            text: text.to_owned().into(), decoration_runs: Vec::new(), cache_key: None,
        }
    }

    fn ascii(text: &str, wraps: &[usize]) -> WrappedLine {
        line(text, &text.char_indices().enumerate().map(|(glyph, (index, _))| (index, glyph as f32 * 10.0)).collect::<Vec<_>>(), wraps, text.chars().count() as f32 * 10.0)
    }

    #[test]
    fn justified_line_fills_soft_rows_and_keeps_final_row_ragged() {
        let original = ascii("a b c d", &[4]);
        let justified = original.justify(px(60.0));
        assert_eq!(justified.text, original.text);
        assert_eq!(justified.wrap_boundaries(), original.wrap_boundaries());
        assert_eq!(original.position_for_index(2, px(20.0)), Some(point(px(20.0), px(0.0))));
        assert_eq!(justified.position_for_index(2, px(20.0)), Some(point(px(50.0), px(0.0))));
        assert_eq!(justified.position_for_index(4, px(20.0)), Some(point(px(60.0), px(0.0))));
        assert_eq!(justified.position_for_index(6, px(20.0)), Some(point(px(20.0), px(20.0))));
        for index in [0, 1, 2, 4, 5, 6, 7] {
            let position = justified.position_for_index(index, px(20.0)).expect("caret position");
            assert_eq!(justified.closest_index_for_position(position, px(20.0)).unwrap_or_else(|index| index), index);
        }
        assert_eq!(justified.closest_index_for_position(point(px(48.0), px(0.0)), px(20.0)), Ok(2));
        assert_eq!(justified.index_for_position(point(px(48.0), px(0.0)), px(20.0)), Ok(1));
    }

    #[test]
    fn justified_line_preserves_ragged_hard_breaks_objects_and_invalid_widths() {
        for text in ["a\u{00a0}b c", "a\u{2002}\u{2060}b c", "a\u{fffc}b c", "a\nb c"] {
            let original = ascii(text, &[4]);
            let justified = original.justify(px(60.0));
            assert!(Arc::ptr_eq(&justified.layout, &original.layout), "{text:?}");
        }
        let original = ascii("a b  ", &[]);
        for width in [px(90.0), px(0.0), px(-1.0), px(f32::NAN)] {
            assert!(Arc::ptr_eq(&original.justify(width).layout, &original.layout));
        }
        let trailing = ascii("a b    c d", &[7]).justify(px(60.0));
        assert_eq!(trailing.position_for_index(2, px(20.0)), Some(point(px(50.0), px(0.0))));
        assert_eq!(trailing.position_for_index(7, px(20.0)), Some(point(px(60.0), px(0.0))));
    }

    #[test]
    fn justified_line_retains_ligatures_marks_decorations_and_visual_bidi_clusters() {
        let mut original = line("fi a\u{301} אב c", &[(0,0.0),(2,20.0),(3,30.0),(3,28.0),(6,40.0),(9,50.0),(7,60.0),(11,70.0),(12,80.0)], &[8], 90.0);
        original.decoration_runs.push(DecorationRun {len:original.len() as u32,color:black(),background_color:Some(black()),underline:Some(UnderlineStyle {thickness:px(1.0),color:Some(black()),wavy:false}),strikethrough:None});
        let justified = original.justify(px(100.0));
        assert_eq!(justified.decoration_runs.len(), original.decoration_runs.len());
        assert_eq!(justified.decoration_runs[0].len, original.decoration_runs[0].len);
        assert_eq!(justified.decoration_runs[0].underline, original.decoration_runs[0].underline);
        assert_eq!(justified.decoration_runs[0].background_color, original.decoration_runs[0].background_color);
        let before = &original.runs()[0].glyphs;
        let after = &justified.runs()[0].glyphs;
        assert_eq!(after[3].position.x-after[2].position.x, before[3].position.x-before[2].position.x);
        for (before, after) in before.iter().zip(after) {assert_eq!(before.id,after.id);assert_eq!(before.index,after.index);assert_eq!(before.position.y,after.position.y);}
        assert_eq!(justified.position_for_index(1,px(20.0)),justified.position_for_index(2,px(20.0)));
        for index in [0,2,3,6,7,9,12,13] {
            let position=justified.position_for_index(index,px(20.0)).expect("cluster position");
            assert_eq!(justified.closest_index_for_position(position,px(20.0)).unwrap_or_else(|index|index),index);
        }
        assert!(after[5].position.x<after[6].position.x);
        assert_eq!(original.text,justified.text);
    }
}
