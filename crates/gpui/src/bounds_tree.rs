use crate::{Bounds, Half, util::CapacityShrink};
use std::{
    fmt::Debug,
    ops::{Add, Sub},
};

/// How many recorded bounds a replay may compare against, summed over every
/// search it makes, before building the grid is the cheaper way on.
const REPLAY_SEARCH_BUDGET: usize = 1 << 15;

/// The side of a grid cell, in the units of the bounds. Most of a frame's
/// bounds are a few dozen scaled pixels across, so each lands in one cell or
/// a handful.
const CELL_SIZE: f64 = 64.;

/// The most cells the grid spans along either axis. Bounds reaching past the
/// last cell are kept in it.
const MAX_CELLS_PER_AXIS: usize = 256;

/// The end of a cell's list.
const NO_ENTRY: u32 = u32::MAX;

/// Hands out draw orders for bounds inserted one after another: each is one
/// greater than the greatest order among the bounds inserted before it that
/// it intersects, so primitives that do not overlap share an order and draw
/// in one batch.
///
/// The bounds live in a uniform grid over the plane. Each cell lists the
/// bounds that reach into it, newest first, except those that cover it whole:
/// every search entering the cell meets those, so the cell keeps only the
/// greatest of their orders. A frame's few thousand bounds, mostly a cell or
/// a few across, are searched and placed several times faster this way than
/// in a hierarchy of boxes.
///
/// Consecutive fills usually insert the same bounds in the same order (only
/// colours or opacities moved), and an order depends on nothing but the
/// bounds inserted before it. So after a clear the tree replays what the fill
/// before handed out, without building the grid, for as long as that stays
/// provably the same answer; see [`BoundsTree::replay`].
#[derive(Debug)]
pub(crate) struct BoundsTree<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    grid: Grid<U>,
    /// The bounds with the greatest order so far, and that order: a search
    /// meeting it has its answer at once.
    max: Option<(Bounds<U>, u32)>,
    /// The bounds inserted since the last clear, in order, each with the
    /// order it was given.
    recorded: Vec<(Bounds<U>, u32)>,
    /// What `recorded` held at the last clear.
    previous: Vec<(Bounds<U>, u32)>,
    /// Whether every order since the last clear was handed out without the
    /// grid, which is built from `recorded` once replaying stops paying.
    replaying: bool,
    /// While replaying, every bounds whose entry differs from the one at the
    /// same position in `previous`, in its bounds or its order, in both its
    /// old and new form.
    changed: ChangedBounds<U>,
    /// How many more recorded bounds replaying may compare against before the
    /// grid is built instead.
    replay_search_budget: usize,
    /// Shrinks `recorded` and `previous` after a run of lighter fills.
    recorded_shrink: CapacityShrink,
}

/// The grid a [`BoundsTree`] files its bounds in. Cell `(column, row)` spans
/// `column * CELL_SIZE` to `(column + 1) * CELL_SIZE` across, and likewise
/// down, except that the first and last column and row reach on without end,
/// so every point of the plane is in exactly one cell.
#[derive(Debug)]
struct Grid<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    columns: usize,
    rows: usize,
    /// Row by row.
    cells: Vec<Cell>,
    /// Every cell's list, linked through `next`.
    entries: Vec<Entry<U>>,
    entries_shrink: CapacityShrink,
    /// Bounds without a positive width and height, or with an origin that is
    /// not a number. They may still meet others, as `intersects` defines it,
    /// but have no cells.
    degenerate: Vec<(Bounds<U>, u32)>,
    /// Whether anything was filed since the grid was last emptied. A fill
    /// replayed from start to end never builds the grid, and emptying its
    /// cells again, up to 65,536 of them, would be for nothing.
    filled: bool,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    /// The newest entry of the cell's list.
    head: u32,
    /// The greatest order among the bounds that cover the cell whole.
    cover: u32,
    /// The greatest order among all the cell's bounds.
    max: u32,
}

impl Cell {
    const EMPTY: Cell = Cell {
        head: NO_ENTRY,
        cover: 0,
        max: 0,
    };
}

#[derive(Clone, Debug)]
struct Entry<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    bounds: Bounds<U>,
    order: u32,
    /// The greatest order of this entry and every one after it in its list,
    /// so a search stops once the rest cannot raise its result.
    rest_max: u32,
    next: u32,
}

/// A bounds' extent along one axis, in grid terms.
#[derive(Clone, Copy)]
struct Span {
    start: f64,
    end: f64,
    first_cell: usize,
    last_cell: usize,
}

impl Span {
    fn new(start: f64, end: f64, cells: usize) -> Self {
        Span {
            start,
            end,
            first_cell: cell_at(start, cells),
            last_cell: cell_at(end, cells),
        }
    }

    /// Whether this span reaches into the interior of `cell` rather than
    /// stopping at one of its edges.
    fn enters(&self, cell: usize, cells: usize) -> bool {
        self.start < cell_end(cell, cells) && self.end > cell_start(cell)
    }

    fn covers(&self, cell: usize, cells: usize) -> bool {
        self.start <= cell_start(cell) && self.end >= cell_end(cell, cells)
    }
}

fn cell_at(coordinate: f64, cells: usize) -> usize {
    // `as` saturates and maps NaN to 0, and truncating toward zero only
    // differs from flooring below zero, which clamps to the first cell.
    ((coordinate * (1. / CELL_SIZE)) as isize).clamp(0, cells as isize - 1) as usize
}

fn cell_start(cell: usize) -> f64 {
    if cell == 0 {
        f64::NEG_INFINITY
    } else {
        cell as f64 * CELL_SIZE
    }
}

fn cell_end(cell: usize, cells: usize) -> f64 {
    if cell + 1 == cells {
        f64::INFINITY
    } else {
        (cell + 1) as f64 * CELL_SIZE
    }
}

/// Columns and rows of the coarse grid [`ChangedBounds`] marks, each
/// [`CELL_SIZE`] across, the first and last reaching on without end.
const CHANGED_CELLS: usize = 64;

/// The bounds that changed during a replay, with the cells of a coarse grid
/// each reaches into marked, so an insert reaching into no marked cell is
/// known to meet none of them without comparing it with each.
///
/// Two intersecting bounds overlap along each axis, and so do the cells they
/// span. Bounds whose far edge is not past their near one, or is not a
/// number, span no cells; once such a bounds has changed, or for such a
/// query, every changed bounds is compared.
#[derive(Debug)]
struct ChangedBounds<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    bounds: Vec<Bounds<U>>,
    /// Row by row, a bit per column.
    rows: [u64; CHANGED_CELLS],
    /// Whether a changed bounds spans no cells, so none can be ruled out.
    unmarked: bool,
}

impl<U> ChangedBounds<U>
where
    U: Clone
        + Debug
        + PartialEq
        + PartialOrd
        + Add<U, Output = U>
        + Sub<Output = U>
        + Half
        + Default
        + Into<f64>,
{
    fn clear(&mut self) {
        self.bounds.clear();
        self.rows = [0; CHANGED_CELLS];
        self.unmarked = false;
    }

    /// The first and last column and row of the cells `bounds` spans, if any.
    fn cells(bounds: &Bounds<U>) -> Option<(usize, usize, usize, usize)> {
        let left: f64 = bounds.origin.x.clone().into();
        let top: f64 = bounds.origin.y.clone().into();
        let right: f64 = (bounds.origin.x.clone() + bounds.size.width.clone()).into();
        let bottom: f64 = (bounds.origin.y.clone() + bounds.size.height.clone()).into();
        // False when either end is NaN.
        (right >= left && bottom >= top).then(|| {
            (
                cell_at(left, CHANGED_CELLS),
                cell_at(right, CHANGED_CELLS),
                cell_at(top, CHANGED_CELLS),
                cell_at(bottom, CHANGED_CELLS),
            )
        })
    }

    fn column_bits(first: usize, last: usize) -> u64 {
        (u64::MAX >> (CHANGED_CELLS - 1 - last)) & (u64::MAX << first)
    }

    fn push(&mut self, bounds: &Bounds<U>) {
        match Self::cells(bounds) {
            Some((left, right, top, bottom)) => {
                let columns = Self::column_bits(left, right);
                for row in &mut self.rows[top..=bottom] {
                    *row |= columns;
                }
            }
            None => self.unmarked = true,
        }
        self.bounds.push(bounds.clone());
    }

    /// False only when `bounds` cannot meet any changed bounds.
    fn might_meet(&self, bounds: &Bounds<U>) -> bool {
        if self.bounds.is_empty() {
            return false;
        }
        if self.unmarked {
            return true;
        }
        match Self::cells(bounds) {
            Some((left, right, top, bottom)) => {
                let columns = Self::column_bits(left, right);
                self.rows[top..=bottom].iter().any(|row| row & columns != 0)
            }
            None => true,
        }
    }
}

impl<U> Grid<U>
where
    U: Clone
        + Debug
        + PartialEq
        + PartialOrd
        + Add<U, Output = U>
        + Sub<Output = U>
        + Half
        + Default
        + Into<f64>,
{
    fn clear(&mut self) {
        if self.filled {
            self.cells.fill(Cell::EMPTY);
            self.degenerate.clear();
            self.filled = false;
        }
        self.entries_shrink.clear_vec(&mut self.entries);
    }

    /// Whether `bounds` has a positive width and height and an origin that is
    /// a number, which is what filing it in cells takes.
    #[allow(clippy::eq_op)]
    fn has_area(bounds: &Bounds<U>) -> bool {
        bounds.size.width > U::default()
            && bounds.size.height > U::default()
            && bounds.origin.x == bounds.origin.x
            && bounds.origin.y == bounds.origin.y
    }

    fn spans(&self, bounds: &Bounds<U>) -> (Span, Span) {
        let right = bounds.origin.x.clone() + bounds.size.width.clone();
        let bottom = bounds.origin.y.clone() + bounds.size.height.clone();
        (
            Span::new(bounds.origin.x.clone().into(), right.into(), self.columns),
            Span::new(bounds.origin.y.clone().into(), bottom.into(), self.rows),
        )
    }

    /// The columns and rows the grid needs to hold `bounds` without filing it
    /// in its last column or row, when that is more than it has.
    fn needs(&self, bounds: &Bounds<U>) -> Option<(usize, usize)> {
        let right: f64 = (bounds.origin.x.clone() + bounds.size.width.clone()).into();
        let bottom: f64 = (bounds.origin.y.clone() + bounds.size.height.clone()).into();
        if right <= self.columns as f64 * CELL_SIZE && bottom <= self.rows as f64 * CELL_SIZE {
            return None;
        }
        let needed = |end: f64| {
            ((end / CELL_SIZE).ceil() as isize).clamp(1, MAX_CELLS_PER_AXIS as isize) as usize
        };
        let (columns, rows) = (
            needed(right).max(self.columns),
            needed(bottom).max(self.rows),
        );
        (columns > self.columns || rows > self.rows).then_some((columns, rows))
    }

    /// Makes the grid `columns` by `rows`, empty.
    fn resize(&mut self, columns: usize, rows: usize) {
        self.columns = columns;
        self.rows = rows;
        self.cells.clear();
        self.cells.resize(columns * rows, Cell::EMPTY);
        self.entries.clear();
        self.degenerate.clear();
        self.filled = false;
    }

    fn add(&mut self, bounds: &Bounds<U>, order: u32) {
        self.filled = true;
        if !Self::has_area(bounds) {
            self.degenerate.push((bounds.clone(), order));
            return;
        }
        let (x, y) = self.spans(bounds);
        for row in y.first_cell..=y.last_cell {
            let covers_row = y.covers(row, self.rows);
            for column in x.first_cell..=x.last_cell {
                let cell = &mut self.cells[row * self.columns + column];
                cell.max = cell.max.max(order);
                if covers_row && x.covers(column, self.columns) {
                    cell.cover = cell.cover.max(order);
                } else {
                    let rest_max = match self.entries.get(cell.head as usize) {
                        Some(next) => next.rest_max.max(order),
                        None => order,
                    };
                    let entry = self.entries.len() as u32;
                    self.entries.push(Entry {
                        bounds: bounds.clone(),
                        order,
                        rest_max,
                        next: cell.head,
                    });
                    cell.head = entry;
                }
            }
        }
    }

    /// The greatest order among the bounds that intersect `query`, which has
    /// a positive width and height, or 0.
    ///
    /// Two such bounds that intersect share a point inside both, and the cell
    /// holding that point either lists the other bounds or is covered by it
    /// whole, and the query reaches into that cell's interior. A covering
    /// bounds meets every query entering the cell, so the cell's cover counts
    /// there without looking at the bounds.
    fn max_intersecting(&self, query: &Bounds<U>) -> u32 {
        let mut max = self
            .degenerate
            .iter()
            .filter(|(bounds, _)| bounds.intersects(query))
            .map(|(_, order)| *order)
            .max()
            .unwrap_or(0);
        let (x, y) = self.spans(query);
        for row in y.first_cell..=y.last_cell {
            let enters_row = y.enters(row, self.rows);
            for column in x.first_cell..=x.last_cell {
                let cell = &self.cells[row * self.columns + column];
                if cell.max <= max {
                    continue;
                }
                if cell.cover > max && enters_row && x.enters(column, self.columns) {
                    max = cell.cover;
                }
                let mut entry_index = cell.head;
                while let Some(entry) = self.entries.get(entry_index as usize) {
                    if entry.rest_max <= max {
                        break;
                    }
                    if entry.order > max && entry.bounds.intersects(query) {
                        max = entry.order;
                    }
                    entry_index = entry.next;
                }
            }
        }
        max
    }
}

impl<U> BoundsTree<U>
where
    U: Clone
        + Debug
        + PartialEq
        + PartialOrd
        + Add<U, Output = U>
        + Sub<Output = U>
        + Half
        + Default
        + Into<f64>,
{
    /// Clears all bounds from the tree, keeping what was inserted aside so
    /// the next fill can replay it.
    pub fn clear(&mut self) {
        self.grid.clear();
        self.max = None;
        let shrink_to = self
            .recorded_shrink
            .record(self.recorded.len(), self.recorded.capacity());
        std::mem::swap(&mut self.previous, &mut self.recorded);
        self.recorded.clear();
        if let Some(capacity) = shrink_to {
            self.recorded.shrink_to(capacity);
            self.previous.shrink_to(capacity);
        }
        self.replaying = true;
        self.changed.clear();
        self.replay_search_budget = REPLAY_SEARCH_BUDGET;
    }

    /// Clears the tree and forgets the fill before, so the next one is
    /// ordered from scratch.
    #[cfg(test)]
    pub fn forget(&mut self) {
        self.clear();
        self.previous.clear();
        self.replaying = false;
    }

    /// Shrinks this cleared tree's storage to twice the fill of `rendered`,
    /// the tree still on screen, once the window has stopped drawing; see
    /// [`CapacityShrink::idle_target`].
    pub fn shrink_idle(&mut self, rendered: &Self) {
        // A fill replayed to its end filed nothing in its grid, so what it
        // recorded stands for what a built grid would have held.
        let rendered_entries = rendered.grid.entries.len().max(rendered.recorded.len());
        self.grid
            .entries_shrink
            .shrink_vec_idle(&mut self.grid.entries, rendered_entries);
        // The fill before this tree was cleared sits in `previous`, kept for
        // replay; a heavy fill there outlives a quiet frame on screen, so it
        // counts toward the storage to release, and is dropped when it holds
        // more than the target (the next fill then orders from scratch).
        let capacity = self.recorded.capacity().max(self.previous.capacity());
        if let Some(capacity) = self
            .recorded_shrink
            .idle_target(rendered.recorded.len(), capacity)
        {
            self.recorded.shrink_to(capacity);
            if self.previous.len() > capacity {
                self.previous.clear();
            }
            self.previous.shrink_to(capacity);
        }
    }

    /// Inserts bounds into the tree and returns its assigned ordering.
    ///
    /// The ordering is one greater than the maximum ordering of any
    /// existing bounds that intersect with the new bounds.
    pub fn insert(&mut self, new_bounds: Bounds<U>) -> u32 {
        if self.replaying {
            if let Some(ordering) = self.replay(&new_bounds) {
                self.recorded.push((new_bounds, ordering));
                return ordering;
            }
            self.replaying = false;
            self.build_from_recorded();
        }

        let ordering = self.find_max_ordering(&new_bounds) + 1;
        self.add(&new_bounds, ordering);
        self.recorded.push((new_bounds, ordering));
        ordering
    }

    /// The order `bounds` gets while the tree is replayed, or `None` once
    /// finding it without the grid would cost more than building the grid.
    ///
    /// If `bounds` is what the previous fill inserted at this position, and
    /// it meets no bounds that was, or is now, different from the previous
    /// fill, then everything it meets and every order among those is as it
    /// was, and so is its own. Otherwise its order is worked out from what was
    /// inserted so far, and when that differs from the previous fill the
    /// bounds joins the changed ones.
    fn replay(&mut self, bounds: &Bounds<U>) -> Option<u32> {
        let previous = self.previous.get(self.recorded.len());
        if let Some((previous_bounds, ordering)) = previous
            && previous_bounds == bounds
        {
            let meets_changed = self.changed.might_meet(bounds) && {
                self.replay_search_budget = self
                    .replay_search_budget
                    .checked_sub(self.changed.bounds.len())?;
                self.changed
                    .bounds
                    .iter()
                    .any(|changed| changed.intersects(bounds))
            };
            if !meets_changed {
                return Some(*ordering);
            }
        }

        self.replay_search_budget = self.replay_search_budget.checked_sub(self.recorded.len())?;
        let ordering = self
            .recorded
            .iter()
            .filter(|(other, _)| other.intersects(bounds))
            .map(|(_, ordering)| *ordering)
            .max()
            .unwrap_or(0)
            + 1;
        match previous {
            Some((previous_bounds, previous_ordering)) => {
                if previous_bounds != bounds {
                    self.changed.push(previous_bounds);
                    self.changed.push(bounds);
                } else if *previous_ordering != ordering {
                    self.changed.push(bounds);
                }
            }
            None => self.changed.push(bounds),
        }
        Some(ordering)
    }

    /// Adds `bounds` with `ordering` to the grid, growing the grid first when
    /// the bounds reach past it.
    fn add(&mut self, bounds: &Bounds<U>, ordering: u32) {
        if Grid::has_area(bounds)
            && let Some((columns, rows)) = self.grid.needs(bounds)
        {
            self.grid.resize(columns, rows);
            for (recorded, recorded_ordering) in &self.recorded {
                self.grid.add(recorded, *recorded_ordering);
            }
        }
        self.grid.add(bounds, ordering);
        if self.max.as_ref().is_none_or(|(_, max)| *max < ordering) {
            self.max = Some((bounds.clone(), ordering));
        }
    }

    /// Builds the grid from what was replayed since the last clear. Those
    /// orders are known, so there is nothing to search, only bounds to file.
    fn build_from_recorded(&mut self) {
        let mut size = (self.grid.columns, self.grid.rows);
        for (bounds, _) in &self.recorded {
            if Grid::has_area(bounds)
                && let Some((columns, rows)) = self.grid.needs(bounds)
            {
                size = (size.0.max(columns), size.1.max(rows));
            }
        }
        if size != (self.grid.columns, self.grid.rows) {
            self.grid.resize(size.0, size.1);
        }
        for (bounds, ordering) in &self.recorded {
            self.grid.add(bounds, *ordering);
            if self.max.as_ref().is_none_or(|(_, max)| max < ordering) {
                self.max = Some((bounds.clone(), *ordering));
            }
        }
    }

    fn find_max_ordering(&self, query: &Bounds<U>) -> u32 {
        if let Some((max_bounds, max)) = &self.max
            && query.intersects(max_bounds)
        {
            return *max;
        }
        if Grid::has_area(query) {
            self.grid.max_intersecting(query)
        } else {
            // A query without area of its own may miss the interior of every
            // cell it touches, so the covers cannot answer for it.
            self.recorded
                .iter()
                .filter(|(bounds, _)| bounds.intersects(query))
                .map(|(_, ordering)| *ordering)
                .max()
                .unwrap_or(0)
        }
    }
}

impl<U> Default for BoundsTree<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    fn default() -> Self {
        BoundsTree {
            grid: Grid {
                columns: 1,
                rows: 1,
                cells: vec![Cell::EMPTY],
                entries: Vec::new(),
                entries_shrink: CapacityShrink::default(),
                degenerate: Vec::new(),
                filled: false,
            },
            max: None,
            recorded: Vec::new(),
            previous: Vec::new(),
            replaying: false,
            changed: ChangedBounds {
                bounds: Vec::new(),
                rows: [0; CHANGED_CELLS],
                unmarked: false,
            },
            replay_search_budget: REPLAY_SEARCH_BUDGET,
            recorded_shrink: CapacityShrink::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, Point, Size};
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_insert() {
        let mut tree = BoundsTree::<f32>::default();
        let bounds1 = Bounds {
            origin: Point { x: 0.0, y: 0.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds2 = Bounds {
            origin: Point { x: 5.0, y: 5.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds3 = Bounds {
            origin: Point { x: 10.0, y: 10.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };

        // Insert the bounds into the tree and verify the order is correct
        assert_eq!(tree.insert(bounds1), 1);
        assert_eq!(tree.insert(bounds2), 2);
        assert_eq!(tree.insert(bounds3), 3);

        // Insert non-overlapping bounds and verify they can reuse orders
        let bounds4 = Bounds {
            origin: Point { x: 20.0, y: 20.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds5 = Bounds {
            origin: Point { x: 40.0, y: 40.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds6 = Bounds {
            origin: Point { x: 25.0, y: 25.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        assert_eq!(tree.insert(bounds4), 1); // bounds4 does not overlap with bounds1, bounds2, or bounds3
        assert_eq!(tree.insert(bounds5), 1); // bounds5 does not overlap with any other bounds
        assert_eq!(tree.insert(bounds6), 2); // bounds6 overlaps with bounds4, so it should have a different order
    }

    fn expected_ordering(inserted: &[(Bounds<f32>, u32)], bounds: &Bounds<f32>) -> u32 {
        inserted
            .iter()
            .filter_map(|(other, order)| other.intersects(bounds).then_some(*order))
            .max()
            .unwrap_or(0)
            + 1
    }

    #[test]
    fn test_random_iterations() {
        let max_bounds = 100;
        for seed in 1..=1000 {
            let mut tree = BoundsTree::default();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
            let mut expected_quads: Vec<(Bounds<f32>, u32)> = Vec::new();

            let num_bounds = rng.random_range(1..=max_bounds);
            for _ in 0..num_bounds {
                let min_x: f32 = rng.random_range(-100.0..100.0);
                let min_y: f32 = rng.random_range(-100.0..100.0);
                let width: f32 = rng.random_range(0.0..50.0);
                let height: f32 = rng.random_range(0.0..50.0);
                let bounds = Bounds {
                    origin: Point { x: min_x, y: min_y },
                    size: Size { width, height },
                };

                let expected = expected_ordering(&expected_quads, &bounds);
                expected_quads.push((bounds, expected));
                assert_eq!(tree.insert(bounds), expected);
            }
        }
    }

    /// Large and spread out enough that bounds span many cells and the grid
    /// grows while it is filled.
    #[test]
    fn many_bounds_over_a_growing_grid_are_ordered_as_comparing_with_all_would() {
        for seed in 1..=10 {
            let mut tree = BoundsTree::default();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
            let mut inserted: Vec<(Bounds<f32>, u32)> = Vec::new();
            for _ in 0..2000 {
                let bounds = Bounds {
                    origin: Point {
                        x: rng.random_range(-1000.0..3000.0),
                        y: rng.random_range(-1000.0..3000.0),
                    },
                    size: Size {
                        width: rng.random_range(0.0..300.0),
                        height: rng.random_range(0.0..300.0),
                    },
                };
                let expected = expected_ordering(&inserted, &bounds);
                inserted.push((bounds, expected));
                assert_eq!(tree.insert(bounds), expected);
            }
        }
    }

    fn random_bounds(rng: &mut rand::rngs::StdRng) -> Bounds<f32> {
        Bounds {
            origin: Point {
                x: rng.random_range(-100.0..100.0),
                y: rng.random_range(-100.0..100.0),
            },
            size: Size {
                width: rng.random_range(0.0..50.0),
                height: rng.random_range(0.0..50.0),
            },
        }
    }

    fn fill(tree: &mut BoundsTree<f32>, frame: &[Bounds<f32>]) {
        tree.clear();
        let mut inserted: Vec<(Bounds<f32>, u32)> = Vec::new();
        for bounds in frame {
            let expected = expected_ordering(&inserted, bounds);
            assert_eq!(tree.insert(*bounds), expected, "{bounds:?}");
            inserted.push((*bounds, expected));
        }
    }

    /// A fill that follows the one before for a while and then goes its own
    /// way, repeats it, or differs from the start, still gives every bounds
    /// the order comparing it with everything before it would.
    #[test]
    fn replaying_the_previous_fill_gives_what_inserting_it_would() {
        for seed in 1..=300 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut tree = BoundsTree::default();
            let count = rng.random_range(1..=120);
            let first: Vec<_> = (0..count).map(|_| random_bounds(&mut rng)).collect();
            fill(&mut tree, &first);

            let kept = rng.random_range(0..=count);
            let mut second: Vec<_> = first[..kept].to_vec();
            second.extend((0..rng.random_range(0..=60)).map(|_| random_bounds(&mut rng)));
            fill(&mut tree, &second);
            fill(&mut tree, &second);

            let third: Vec<_> = (0..count).map(|_| random_bounds(&mut rng)).collect();
            fill(&mut tree, &third);
        }
    }

    /// A fill repeating the one before but for a few scattered bounds (a
    /// label grown by a digit, a row inserted or removed) is replayed past
    /// each of them. Fills with changes enough to exhaust the replay budget
    /// build the grid from wherever that happens.
    #[test]
    fn replaying_past_scattered_changes_gives_what_inserting_it_would() {
        for seed in 1..=300 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut tree = BoundsTree::default();
            let count = rng.random_range(1..=if seed % 10 == 0 { 800 } else { 150 });
            let mut frame: Vec<_> = (0..count).map(|_| random_bounds(&mut rng)).collect();
            fill(&mut tree, &frame);
            for _ in 0..6 {
                for _ in 0..rng.random_range(0..=count / 8 + 1) {
                    let at = rng.random_range(0..frame.len().max(1));
                    match rng.random_range(0..10) {
                        0 if !frame.is_empty() => {
                            frame.remove(at);
                        }
                        1 => frame.insert(at.min(frame.len()), random_bounds(&mut rng)),
                        _ if !frame.is_empty() => {
                            frame[at].size.width += rng.random_range(-5.0..5.0);
                        }
                        _ => {}
                    }
                }
                fill(&mut tree, &frame);
            }
        }
    }

    fn awkward_coordinate(rng: &mut rand::rngs::StdRng, reach: i32) -> f32 {
        let edge = CELL_SIZE as f32;
        match rng.random_range(0..14) {
            0 => f32::INFINITY,
            1 => f32::NEG_INFINITY,
            2 => f32::NAN,
            3 => rng.random_range(-4..reach + 4) as f32 * edge,
            4..8 => rng.random_range(-3..8) as f32 * edge,
            8 => rng.random_range(0.0..(reach as f32 + 8.) * edge),
            _ => rng.random_range(-2.0 * edge..6.0 * edge),
        }
    }

    fn awkward_length(rng: &mut rand::rngs::StdRng) -> f32 {
        let edge = CELL_SIZE as f32;
        match rng.random_range(0..12) {
            0 => 0.,
            1 => -rng.random_range(0.0..edge),
            2 => f32::INFINITY,
            3 => f32::NAN,
            4..7 => rng.random_range(0..4) as f32 * edge,
            _ => rng.random_range(0.0..3.0 * edge),
        }
    }

    fn awkward_bounds(rng: &mut rand::rngs::StdRng, reach: i32) -> Bounds<f32> {
        Bounds {
            origin: Point {
                x: awkward_coordinate(rng, reach),
                y: awkward_coordinate(rng, reach),
            },
            size: Size {
                width: awkward_length(rng),
                height: awkward_length(rng),
            },
        }
    }

    /// Bounds on cell edges, covering cells whole, reaching past the grid,
    /// empty, negative, infinite or NaN each still get the order comparing
    /// them with every bounds before them gives.
    #[test]
    fn awkward_bounds_are_ordered_as_comparing_with_all_would() {
        for seed in 1..=400 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut tree = BoundsTree::default();
            for _ in 0..3 {
                let frame: Vec<_> = (0..rng.random_range(1..200))
                    .map(|_| awkward_bounds(&mut rng, MAX_CELLS_PER_AXIS as i32))
                    .collect();
                tree.forget();
                let mut inserted: Vec<(Bounds<f32>, u32)> = Vec::new();
                for bounds in &frame {
                    let expected = expected_ordering(&inserted, bounds);
                    assert_eq!(tree.insert(*bounds), expected, "seed {seed}: {bounds:?}");
                    inserted.push((*bounds, expected));
                }
            }
        }
    }

    /// The same awkward bounds coming and going between replayed fills,
    /// including past the cells the changed bounds are marked in.
    #[test]
    fn replaying_past_awkward_changes_gives_what_inserting_it_would() {
        for seed in 1..=400 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut tree = BoundsTree::default();
            let mut frame: Vec<_> = (0..rng.random_range(1..200))
                .map(|_| awkward_bounds(&mut rng, CHANGED_CELLS as i32))
                .collect();
            fill(&mut tree, &frame);
            for _ in 0..4 {
                for _ in 0..rng.random_range(0..6) {
                    let at = rng.random_range(0..frame.len());
                    frame[at] = awkward_bounds(&mut rng, CHANGED_CELLS as i32);
                }
                fill(&mut tree, &frame);
            }
        }
    }

    /// A fill replayed from start to end files nothing in the grid, and the
    /// clear after it leaves the grid's cells alone; one that diverges builds
    /// the grid and the clear after it empties it.
    #[test]
    fn only_a_built_grid_is_emptied() {
        // Enough bounds that ordering them without the grid runs out of
        // budget, when they are not the ones replayed.
        let frame: Vec<_> = (0..400).map(unit_bounds).collect();
        let mut tree = BoundsTree::default();
        fill(&mut tree, &frame);
        assert!(tree.grid.filled);
        fill(&mut tree, &frame);
        assert!(!tree.grid.filled, "a replayed fill never builds the grid");
        let shifted: Vec<_> = (1000..1400).map(unit_bounds).collect();
        fill(&mut tree, &shifted);
        assert!(tree.grid.filled);
        tree.clear();
        assert!(!tree.grid.filled);
        assert!(tree.grid.cells.iter().all(|cell| cell.head == NO_ENTRY));
    }

    fn unit_bounds(index: usize) -> Bounds<f32> {
        Bounds {
            origin: Point {
                x: (index % 100) as f32 * 2.0,
                y: (index / 100) as f32 * 2.0,
            },
            size: Size {
                width: 1.0,
                height: 1.0,
            },
        }
    }

    /// A tree cleared after a heavy fill keeps its storage; the idle release
    /// sizes it to the tree still on screen, and the tree still works.
    #[test]
    fn idle_release_sizes_storage_to_the_rendered_tree() {
        let mut retired = BoundsTree::<f32>::default();
        for index in 0..10_000 {
            retired.insert(unit_bounds(index));
        }
        assert!(retired.grid.entries.len() >= 10_000);
        retired.clear();
        retired.insert(unit_bounds(0));
        // The heavy fill's recording comes back around as the one to fill.
        retired.clear();
        assert!(retired.grid.entries.capacity() >= 10_000);
        assert!(retired.recorded.capacity() >= 10_000);

        let mut rendered = BoundsTree::<f32>::default();
        rendered.insert(unit_bounds(0));
        retired.shrink_idle(&rendered);
        assert!(retired.grid.entries.capacity() <= crate::util::MIN_RETAINED_CAPACITY);
        assert!(retired.recorded.capacity() <= crate::util::MIN_RETAINED_CAPACITY);

        assert_eq!(retired.insert(unit_bounds(0)), 1);
        assert_eq!(retired.insert(unit_bounds(0)), 2);
    }

    /// A heavy fill cleared once is kept for replay; going idle on a light
    /// frame releases it too.
    #[test]
    fn idle_release_drops_a_heavy_fill_kept_for_replay() {
        let mut retired = BoundsTree::<f32>::default();
        for index in 0..10_000 {
            retired.insert(unit_bounds(index));
        }
        retired.clear();
        assert!(retired.previous.capacity() >= 10_000);

        let mut rendered = BoundsTree::<f32>::default();
        rendered.insert(unit_bounds(0));
        retired.shrink_idle(&rendered);
        assert!(retired.previous.capacity() <= crate::util::MIN_RETAINED_CAPACITY);

        assert_eq!(retired.insert(unit_bounds(0)), 1);
        assert_eq!(retired.insert(unit_bounds(0)), 2);
    }
}
