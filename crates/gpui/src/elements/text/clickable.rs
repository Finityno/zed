//! Where an interactive text shows the hand: over hitboxes laid over its
//! clickable ranges as it prepaints, rather than where the pointer was as it
//! painted.
//!
//! Reading the pointer as the text paints makes the view painting it depend
//! on the pointer, so that view is rendered again in every frame drawn after
//! the pointer moved anywhere in the window, and its mouse-move listener had
//! to notify the view for every character the pointer crossed, for the paint
//! to set the hand again. A cursor style set for a hitbox is picked by the
//! window's hit test instead, which follows the pointer without a frame.
//!
//! The clicks still find their index with [`TextLayout::index_for_position`],
//! so the stretches follow its steps: from line to line there, from row to
//! row (with the row's alignment offset) in
//! `WrappedLineLayout::_index_for_position`, and along a row in
//! `LineLayout::index_for_x`. Where one line ends and the next begins,
//! `index_for_position` takes the y for the line above, below its last row,
//! and finds no index; a hitbox holds the y it begins at, so the line below
//! takes it.

use super::TextLayout;
use crate::{Bounds, Pixels, TextAlign, WrapBoundary, WrappedLineLayout, point, size};
use std::ops::Range;

/// Calls `stretch` with the bounds, in the window, of each stretch of
/// `layout`'s rows where [`TextLayout::index_for_position`] finds an index
/// in one of `ranges`, clipped to the text's bounds. Nothing for text that
/// was not prepainted.
pub(super) fn clickable_stretches(
    layout: &TextLayout,
    ranges: &[Range<usize>],
    mut stretch: impl FnMut(Bounds<Pixels>),
) {
    let ranges = in_order(ranges);
    if ranges.is_empty() {
        return;
    }
    let inner = layout.0.borrow();
    let Some(inner) = inner.as_ref() else {
        return;
    };
    let Some(bounds) = inner.bounds else {
        return;
    };
    let line_height = inner.line_height;
    let mut line_origin = bounds.origin;
    let mut line_start_ix = 0;
    for line in &inner.lines {
        let line_end_ix = line_start_ix + line.len();
        if touches(&ranges, line_start_ix..line_end_ix) {
            line_stretches(
                line,
                line_height,
                inner.text_align,
                bounds.size.width,
                |ix| contains(&ranges, line_start_ix + ix),
                |line_stretch| {
                    let clipped = Bounds::new(line_origin + line_stretch.origin, line_stretch.size)
                        .intersect(&bounds);
                    if !clipped.is_empty() {
                        stretch(clipped);
                    }
                },
            );
        }
        line_origin.y += line.size(line_height).height;
        line_start_ix = line_end_ix + 1;
    }
}

/// Calls `stretch` with each stretch of `line`'s rows, in the line's own space
/// (its first row's top at zero, rows offset as `align` places them in
/// `width`), where `WrappedLineLayout::index_for_position`, given the row's
/// alignment offset as [`TextLayout::index_for_position`] gives it, finds an
/// index that `clickable` takes. Each stretch is as tall as a row, and
/// stretches that meet are merged.
pub(super) fn line_stretches(
    line: &WrappedLineLayout,
    line_height: Pixels,
    align: TextAlign,
    width: Pixels,
    clickable: impl Fn(usize) -> bool,
    mut stretch: impl FnMut(Bounds<Pixels>),
) {
    let layout = &line.unwrapped_layout;
    let steps = index_steps(line);
    let boundary_x = |boundary: &WrapBoundary| {
        layout.runs[boundary.run_ix].glyphs[boundary.glyph_ix]
            .position
            .x
    };
    for row in 0..=line.wrap_boundaries.len() {
        let start_x = match row {
            0 => Pixels::ZERO,
            row => boundary_x(&line.wrap_boundaries[row - 1]),
        };
        // `index_for_x` finds no glyph from the line's width on, and the
        // shaper can place a zero-width wrap boundary past it.
        let end_x = line
            .wrap_boundaries
            .get(row)
            .map_or(layout.width, boundary_x)
            .min(layout.width);
        if start_x >= end_x {
            continue;
        }
        let left = line.alignment_offset(row, align, width) - start_x;
        let top = line_height * row as f32;
        let mut emit = |from: Pixels, to: Pixels| {
            stretch(Bounds::new(
                point(left + from, top),
                size(to - from, line_height),
            ))
        };
        let first = steps.partition_point(|&(x, _)| x <= start_x);
        let mut index = first.checked_sub(1).map_or(0, |step| steps[step].1);
        let mut from = start_x;
        let mut open = None;
        for &(x, next_index) in steps[first..].iter().take_while(|&&(x, _)| x < end_x) {
            note_piece(&clickable, index, from, &mut open, &mut emit);
            from = x;
            index = next_index;
        }
        note_piece(&clickable, index, from, &mut open, &mut emit);
        if let Some(open) = open {
            emit(open, end_x);
        }
    }
}

/// Opens a stretch at `from`, where a piece of a row with `index` begins, if
/// `clickable` takes it and none is open, or closes the open one there if it
/// does not.
fn note_piece(
    clickable: &impl Fn(usize) -> bool,
    index: usize,
    from: Pixels,
    open: &mut Option<Pixels>,
    emit: &mut impl FnMut(Pixels, Pixels),
) {
    if clickable(index) {
        open.get_or_insert(from);
    } else if let Some(open) = open.take() {
        emit(open, from);
    }
}

/// Where along `line`, unwrapped, the index `LineLayout::index_for_x` finds
/// changes, and the index from there on. `index_for_x` takes the last glyph,
/// in the order the runs hold them, at or before an x (or 0 before them all),
/// so the glyphs are taken by x and each step keeps the latest in the runs'
/// order seen so far, which also holds for text running right to left or in
/// both directions.
fn index_steps(line: &WrappedLineLayout) -> Vec<(Pixels, usize)> {
    let mut glyphs: Vec<_> = line
        .unwrapped_layout
        .runs
        .iter()
        .flat_map(|run| &run.glyphs)
        .enumerate()
        .map(|(order, glyph)| (glyph.position.x, order, glyph.index))
        .collect();
    // Stable, so glyphs at the same x stay in the runs' order.
    glyphs.sort_by(|a, b| a.0.0.total_cmp(&b.0.0));
    let mut steps: Vec<(Pixels, usize)> = Vec::with_capacity(glyphs.len());
    let mut latest: Option<usize> = None;
    let mut latest_index = 0;
    for (x, order, index) in glyphs {
        if latest.is_none_or(|latest| order > latest) {
            latest = Some(order);
            latest_index = index;
        }
        match steps.last_mut() {
            Some(step) if step.0 == x => step.1 = latest_index,
            _ => steps.push((x, latest_index)),
        }
    }
    steps
}

/// `ranges` in order, empty ones dropped and ones that meet merged, so that
/// an index is looked up in them by a binary search.
fn in_order(ranges: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut sorted: Vec<_> = ranges
        .iter()
        .filter(|range| range.start < range.end)
        .cloned()
        .collect();
    sorted.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(sorted.len());
    for range in sorted {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

/// The first of `ranges`, in order, that ends after `ix`.
fn first_ending_after(ranges: &[Range<usize>], ix: usize) -> Option<&Range<usize>> {
    ranges.get(ranges.partition_point(|range| range.end <= ix))
}

fn contains(ranges: &[Range<usize>], ix: usize) -> bool {
    first_ending_after(ranges, ix).is_some_and(|range| range.start <= ix)
}

/// Whether one of `ranges`, in order, holds an index in `span`.
fn touches(ranges: &[Range<usize>], span: Range<usize>) -> bool {
    first_ending_after(ranges, span.start).is_some_and(|range| range.start < span.end)
}

#[cfg(test)]
mod tests;
