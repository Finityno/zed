//! A cache for storing the results of layout computation

#![allow(clippy::unusual_byte_groupings)]

use crate::geometry::Size;
use crate::style::AvailableSpace;
use crate::tree::{CollapsibleMarginSet, LayoutInput, LayoutOutput, RunMode};
use crate::RequestedAxis;

/// The number of cache entries for each node in the tree
const CACHE_SIZE: usize = 9;

// Manually written-out results of float to u32 bit casts because
// `f32::to_bits` is not yet const at our MSRV.

/// `f32::INFINITY` as a u32
const INFINITY_BITS: u32 = 0b_0_11111111_00000000000000000000000_u32;
/// `f32::NEG_INFINITY` as a u32
const NEG_INFINITY_BITS: u32 = 0b_1_11111111_00000000000000000000000_u32;

// The `CacheKey` encodes two f32s as a u64. We know that the f32s will always be
// non-negative, so we pack two extra bits encoding the `RequestedAxis` into the
// sign bits of the f32s. These constants help to encode and decode those bits.

/// The sign bit of the first f32
const SIGN_BIT_1: u64 = 1u64 << 63;
/// The sign bit of the second f32
const SIGN_BIT_2: u64 = 1u64 << 31;
/// Mask of both sign bits (used to compute NON_SIGN_BITS_MASK)
const BOTH_SIGN_BITS_MASK: u64 = SIGN_BIT_1 | SIGN_BIT_2;
/// Mask of excluding the sign bits (used when setting/getting the size excluding the packed bits)
const NON_SIGN_BITS_MASK: u64 = !BOTH_SIGN_BITS_MASK;

/// Mask which includes only the bits which encode the x-axis value that we can use to ignore the
/// y-axis value when comparing a cache key.
const X_AXIS_VALUE_MASK: u64 = (u32::MAX as u64) << 32;

/// Pack `Option<f32>` into `u32`
#[inline(always)]
fn option_cache_key(input: Option<f32>) -> u32 {
    match input {
        Some(value) => value.to_bits(),
        None => INFINITY_BITS,
    }
}

/// Pack `Size<Option<f32>>` into `u64`
#[inline(always)]
fn size_option_cache_key(input: Size<Option<f32>>) -> u64 {
    (option_cache_key(input.width) as u64) << 32 | option_cache_key(input.height) as u64
}

/// Pack `AvailableSpace` into `u32`
#[inline(always)]
fn available_space_cache_key(input: AvailableSpace) -> u32 {
    match input {
        AvailableSpace::Definite(value) => (-value).to_bits(),
        AvailableSpace::MinContent => NEG_INFINITY_BITS,
        AvailableSpace::MaxContent => INFINITY_BITS,
    }
}

/// Pack `Size<AvailableSpace>` into `u64`
#[inline(always)]
#[allow(dead_code)]
fn size_available_space_cache_key(input: Size<AvailableSpace>) -> u64 {
    (available_space_cache_key(input.width) as u64) << 32 | available_space_cache_key(input.height) as u64
}

/// Encodes combination of a `known_dimension` (Option<f32>) and `AvailableSpace` in
/// a single dimension into a cache key in a single dimension.
#[inline(always)]
fn mixed_cache_key(kd: Option<f32>, avs: AvailableSpace) -> u32 {
    kd.map(|kd| kd.to_bits()).unwrap_or_else(|| available_space_cache_key(avs))
}

/// Encodes combination of a `known_dimension` (Option<f32>) and `AvailableSpace` in
/// two dimensions into a cache key in a single dimension.
#[inline(always)]
fn size_mixed_cache_key(kd: Size<Option<f32>>, avs: Size<AvailableSpace>) -> u64 {
    (mixed_cache_key(kd.width, avs.width) as u64) << 32 | mixed_cache_key(kd.height, avs.height) as u64
}

/// Space-optimised cache key that packs bits into as small a size as possible
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
struct CacheKey {
    /// The initial cached size of the node itself
    kd_available_space: u64,
    /// The initial cached size of the parent's node
    parent_size: u64,
}

impl CacheKey {
    #[inline(always)]
    #[allow(dead_code)]
    /// Return the parent size with the extra bits that encode the requested axis masked out
    fn parent_size(&self) -> u64 {
        self.parent_size & NON_SIGN_BITS_MASK
    }

    /// Return the parent size with the extra bits that encode the requested axis masked out
    /// And the y-axis value masked out
    fn x_axis_parent_size(&self) -> u64 {
        self.parent_size & (X_AXIS_VALUE_MASK & NON_SIGN_BITS_MASK)
    }
}

impl From<&LayoutInput> for CacheKey {
    fn from(input: &LayoutInput) -> Self {
        // Pack axis enum into spare bits in the known_dimensions and available_space values
        let extra_bits = match input.axis {
            RequestedAxis::Horizontal => SIGN_BIT_1,
            RequestedAxis::Vertical => SIGN_BIT_2,
            RequestedAxis::Both => SIGN_BIT_1 | SIGN_BIT_2,
        };

        Self {
            kd_available_space: size_mixed_cache_key(input.known_dimensions, input.available_space),
            parent_size: (size_option_cache_key(input.parent_size) & NON_SIGN_BITS_MASK) | extra_bits,
        }
    }
}

/// Cached intermediate layout results
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub(crate) struct CacheEntry<T> {
    /// The key for the cache entry
    key: CacheKey,
    /// The cached size and baselines of the item
    content: T,
}

/// A cache for caching the results of a sizing a Grid Item or Flexbox Item
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Cache {
    /// The cache entry for the node's final layout
    final_layout_entry: Option<CacheEntry<LayoutOutput>>,
    /// The cache entries for the node's preliminary size measurements
    measure_entries: [Option<CacheEntry<Size<f32>>>; CACHE_SIZE],
    next_measure_entry: u8,
    /// Tracks if all cache entries are empty
    is_empty: bool,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

impl Cache {
    /// Create a new empty cache
    pub const fn new() -> Self {
        Self { final_layout_entry: None, measure_entries: [None; CACHE_SIZE], next_measure_entry: 0, is_empty: true }
    }

    /// Try to retrieve a cached result from the cache
    #[inline]
    pub fn get(&self, input: &LayoutInput) -> Option<LayoutOutput> {
        let key = CacheKey::from(input);
        match input.run_mode {
            RunMode::PerformLayout => self.final_layout_entry.filter(|entry| entry.key == key).map(|e| e.content),
            RunMode::ComputeSize => {
                for entry in self.measure_entries.iter().flatten() {
                    if entry.key.kd_available_space == key.kd_available_space
                        && (entry.key.x_axis_parent_size() == key.x_axis_parent_size())
                        && (entry.key.parent_size & BOTH_SIGN_BITS_MASK) & (key.parent_size & BOTH_SIGN_BITS_MASK)
                            == key.parent_size & BOTH_SIGN_BITS_MASK
                    {
                        return Some(LayoutOutput::from_outer_size(entry.content));
                    }
                }

                None
            }
            RunMode::PerformHiddenLayout => None,
        }
    }

    /// Store a computed size in the cache
    pub fn store(&mut self, input: &LayoutInput, layout_output: LayoutOutput) {
        let key = CacheKey::from(input);
        match input.run_mode {
            RunMode::PerformLayout => {
                self.is_empty = false;
                self.final_layout_entry = Some(CacheEntry { key, content: layout_output })
            }
            RunMode::ComputeSize => {
                // A size-only entry cannot preserve margin-collapse metadata.
                if layout_output.margins_can_collapse_through
                    || layout_output.top_margin != CollapsibleMarginSet::ZERO
                    || layout_output.bottom_margin != CollapsibleMarginSet::ZERO
                {
                    return;
                }
                self.is_empty = false;
                for entry in self.measure_entries.iter_mut().flatten() {
                    if entry.key == key {
                        entry.content = layout_output.size;
                        return;
                    }
                }
                // Definite constraints of the same kind can recur within one layout;
                // giving each a slot avoids repeatedly evicting and measuring them.
                self.measure_entries[usize::from(self.next_measure_entry)] = Some(CacheEntry { key, content: layout_output.size });
                self.next_measure_entry += 1;
                if usize::from(self.next_measure_entry) == CACHE_SIZE {
                    self.next_measure_entry = 0;
                }
            }
            RunMode::PerformHiddenLayout => {}
        }
    }

    /// Clear all cache entries and reports clear operation outcome ([`ClearState`])
    pub fn clear(&mut self) -> ClearState {
        if self.is_empty {
            return ClearState::AlreadyEmpty;
        }
        self.is_empty = true;
        self.final_layout_entry = None;
        self.measure_entries = [None; CACHE_SIZE];
        self.next_measure_entry = 0;
        ClearState::Cleared
    }

    /// Returns true if all cache entries are None, else false
    pub fn is_empty(&self) -> bool {
        self.final_layout_entry.is_none() && !self.measure_entries.iter().any(|entry| entry.is_some())
    }
}

/// Clear operation outcome. See [`Cache::clear`]
pub enum ClearState {
    /// Cleared some values
    Cleared,
    /// Everything was already cleared
    AlreadyEmpty,
}

#[cfg(test)]
mod owned_constraint_tests {
    use super::*;
    use crate::tree::{CollapsibleMarginSet, SizingMode};

    fn input(width: f32) -> LayoutInput {
        LayoutInput {
            run_mode: RunMode::ComputeSize,
            sizing_mode: SizingMode::InherentSize,
            axis: RequestedAxis::Both,
            known_dimensions: Size::NONE,
            parent_size: Size::NONE,
            available_space: Size { width: AvailableSpace::Definite(width), height: AvailableSpace::MaxContent },
            vertical_margins_are_collapsible: crate::geometry::Line { start: false, end: false },
        }
    }

    fn output(width: f32) -> LayoutOutput {
        LayoutOutput::from_outer_size(Size { width, height: 1000.0 / width })
    }

    #[test]
    fn repeated_distinct_definite_constraints_retain_their_sizes() {
        let mut cache = Cache::new();
        cache.store(&input(80.0), output(80.0));
        cache.store(&input(120.0), output(120.0));
        assert_eq!(cache.get(&input(80.0)).map(|value| value.size), Some(output(80.0).size));
        assert_eq!(cache.get(&input(120.0)).map(|value| value.size), Some(output(120.0).size));
    }

    #[test]
    fn partial_axis_results_do_not_supply_another_axis() {
        let mut cache = Cache::new();
        let mut horizontal = input(80.0);
        horizontal.axis = RequestedAxis::Horizontal;
        cache.store(&horizontal, output(80.0));
        let mut vertical = horizontal;
        vertical.axis = RequestedAxis::Vertical;
        assert!(cache.get(&vertical).is_none());
        assert!(cache.get(&input(80.0)).is_none());
        cache.store(&input(80.0), output(80.0));
        assert_eq!(cache.get(&vertical).map(|value| value.size), Some(output(80.0).size));
    }

    #[test]
    fn size_only_entries_preserve_margin_metadata_by_recomputing() {
        let mut cache = Cache::new();
        for value in [
            LayoutOutput { top_margin: CollapsibleMarginSet::from_margin(12.0), ..output(80.0) },
            LayoutOutput { bottom_margin: CollapsibleMarginSet::from_margin(-6.0), ..output(80.0) },
            LayoutOutput { margins_can_collapse_through: true, ..output(80.0) },
        ] {
            cache.store(&input(80.0), value);
            assert!(cache.get(&input(80.0)).is_none());
            assert!(cache.is_empty());
        }
    }

    #[test]
    fn replacement_and_clear_preserve_layout_and_measurement_contracts() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Cache>();
        let mut cache = Cache::new();
        for width in 1..=200 {
            cache.store(&input(width as f32), output(width as f32));
        }
        cache.store(&input(200.0), output(999.0));
        assert_eq!(cache.get(&input(200.0)).map(|value| value.size), Some(output(999.0).size));
        let mut layout_input = input(200.0);
        layout_input.run_mode = RunMode::PerformLayout;
        cache.store(&layout_input, output(200.0));
        assert_eq!(cache.get(&layout_input), Some(output(200.0)));
        layout_input.parent_size.width = Some(500.0);
        assert!(cache.get(&layout_input).is_none());
        assert!(matches!(cache.clear(), ClearState::Cleared));
        assert!(cache.is_empty());
        assert!(matches!(cache.clear(), ClearState::AlreadyEmpty));
        assert!(cache.get(&input(200.0)).is_none());
    }
}
