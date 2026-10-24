//! Process-wide gauges of the device memory the platform renderers hold.

use std::sync::atomic::{AtomicU64, Ordering};

/// Device memory the process's renderers hold, in bytes: the sum over every
/// live [`RenderMemoryLedger`], as of each holder's last report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderMemoryGauges {
    /// Atlas pages holding glyphs and SVGs, one byte per pixel.
    pub atlas_monochrome_bytes: u64,
    /// Atlas pages holding images, emoji and subpixel glyphs, four bytes per
    /// pixel.
    pub atlas_polychrome_bytes: u64,
    /// Instance buffers waiting in the renderers' shared pool. Buffers in
    /// flight for a frame being drawn return to the pool when the frame
    /// completes and are not counted until then. The wgpu renderer keeps one
    /// instance buffer per window and reports its capacity.
    pub instance_buffer_bytes: u64,
    /// Drawable-sized depth attachments. One kept in tile memory (memoryless,
    /// on Apple GPUs) has no allocation and reports nothing.
    pub depth_texture_bytes: u64,
    /// Targets paths are rasterized and composited through: the drawable-sized
    /// intermediate, the tile paths rasterize into and its multisampled
    /// companion.
    pub path_texture_bytes: u64,
    /// Scratch textures backdrop blurs (frosted surfaces) run through.
    pub backdrop_texture_bytes: u64,
    /// The device's own count of everything the process has allocated on
    /// it, as of the last frame any renderer drew: the gauges above plus
    /// drawables-in-waiting, buffers and textures no gauge names. Read from
    /// `MTLDevice.currentAllocatedSize` on Metal; zero where the platform
    /// has no such counter.
    pub device_allocated_bytes: u64,
}

/// One of the gauges in [`RenderMemoryGauges`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMemoryGauge {
    /// [`RenderMemoryGauges::atlas_monochrome_bytes`].
    AtlasMonochrome,
    /// [`RenderMemoryGauges::atlas_polychrome_bytes`].
    AtlasPolychrome,
    /// [`RenderMemoryGauges::instance_buffer_bytes`].
    InstanceBuffers,
    /// [`RenderMemoryGauges::depth_texture_bytes`].
    DepthTextures,
    /// [`RenderMemoryGauges::path_texture_bytes`].
    PathTextures,
    /// [`RenderMemoryGauges::backdrop_texture_bytes`].
    BackdropTextures,
}

const ALL_GAUGES: [RenderMemoryGauge; 6] = [
    RenderMemoryGauge::AtlasMonochrome,
    RenderMemoryGauge::AtlasPolychrome,
    RenderMemoryGauge::InstanceBuffers,
    RenderMemoryGauge::DepthTextures,
    RenderMemoryGauge::PathTextures,
    RenderMemoryGauge::BackdropTextures,
];

static GAUGES: [AtomicU64; ALL_GAUGES.len()] = [const { AtomicU64::new(0) }; ALL_GAUGES.len()];
static DEVICE_ALLOCATED: AtomicU64 = AtomicU64::new(0);

/// Records the device's own count of allocated bytes, for
/// [`RenderMemoryGauges::device_allocated_bytes`]. Every renderer in a process
/// shares one device, so the latest reading replaces the last.
pub fn record_device_allocated_bytes(bytes: u64) {
    DEVICE_ALLOCATED.store(bytes, Ordering::Relaxed);
}

/// The bytes one holder of device memory — a renderer, an atlas, a buffer
/// pool — has reported into the process-wide gauges. A report replaces the
/// holder's previous one, so the gauges stay the sum over live holders however
/// often each reports, and dropping the ledger takes the holder's bytes back
/// out, so a closed window stops counting.
#[derive(Debug, Default)]
pub struct RenderMemoryLedger {
    published: [u64; ALL_GAUGES.len()],
}

impl RenderMemoryLedger {
    /// Reports this holder's current `bytes` for `gauge`.
    pub fn publish(&mut self, gauge: RenderMemoryGauge, bytes: u64) {
        let slot = &mut self.published[gauge as usize];
        let global = &GAUGES[gauge as usize];
        if bytes >= *slot {
            global.fetch_add(bytes - *slot, Ordering::Relaxed);
        } else {
            global.fetch_sub(*slot - bytes, Ordering::Relaxed);
        }
        *slot = bytes;
    }

    /// What this holder last reported for `gauge`.
    pub fn published(&self, gauge: RenderMemoryGauge) -> u64 {
        self.published[gauge as usize]
    }
}

impl Drop for RenderMemoryLedger {
    fn drop(&mut self) {
        for gauge in ALL_GAUGES {
            self.publish(gauge, 0);
        }
    }
}

/// The device memory every live renderer holds, summed across windows. A few
/// atomic loads; callable from any thread with no `App` at hand.
pub fn render_memory_gauges() -> RenderMemoryGauges {
    let load = |gauge: RenderMemoryGauge| GAUGES[gauge as usize].load(Ordering::Relaxed);
    RenderMemoryGauges {
        atlas_monochrome_bytes: load(RenderMemoryGauge::AtlasMonochrome),
        atlas_polychrome_bytes: load(RenderMemoryGauge::AtlasPolychrome),
        instance_buffer_bytes: load(RenderMemoryGauge::InstanceBuffers),
        depth_texture_bytes: load(RenderMemoryGauge::DepthTextures),
        path_texture_bytes: load(RenderMemoryGauge::PathTextures),
        backdrop_texture_bytes: load(RenderMemoryGauge::BackdropTextures),
        device_allocated_bytes: DEVICE_ALLOCATED.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn ledgers_sum_into_the_gauges_and_leave_when_dropped() {
        let before = render_memory_gauges();

        let mut atlas = RenderMemoryLedger::default();
        let mut renderer = RenderMemoryLedger::default();
        atlas.publish(RenderMemoryGauge::AtlasPolychrome, 4 * MIB);
        renderer.publish(RenderMemoryGauge::DepthTextures, 15 * MIB);
        // A smaller re-report takes the difference back out.
        renderer.publish(RenderMemoryGauge::DepthTextures, 10 * MIB);
        assert_eq!(atlas.published(RenderMemoryGauge::AtlasPolychrome), 4 * MIB);
        assert_eq!(renderer.published(RenderMemoryGauge::DepthTextures), 10 * MIB);

        let during = render_memory_gauges();
        assert_eq!(
            during.atlas_polychrome_bytes - before.atlas_polychrome_bytes,
            4 * MIB
        );
        assert_eq!(
            during.depth_texture_bytes - before.depth_texture_bytes,
            10 * MIB
        );
        assert_eq!(during.atlas_monochrome_bytes, before.atlas_monochrome_bytes);
        assert_eq!(during.instance_buffer_bytes, before.instance_buffer_bytes);
        assert_eq!(during.path_texture_bytes, before.path_texture_bytes);

        drop(renderer);
        let without_renderer = render_memory_gauges();
        assert_eq!(
            without_renderer.depth_texture_bytes,
            before.depth_texture_bytes
        );
        assert_eq!(
            without_renderer.atlas_polychrome_bytes - before.atlas_polychrome_bytes,
            4 * MIB
        );

        drop(atlas);
        // The device count is a reading another test may take meanwhile.
        assert_eq!(
            RenderMemoryGauges {
                device_allocated_bytes: before.device_allocated_bytes,
                ..render_memory_gauges()
            },
            before
        );
    }

    #[test]
    fn backdrop_scratch_and_the_device_count_are_reported() {
        let before = render_memory_gauges();
        let mut renderer = RenderMemoryLedger::default();
        renderer.publish(RenderMemoryGauge::BackdropTextures, 2 * MIB);
        assert_eq!(
            render_memory_gauges().backdrop_texture_bytes - before.backdrop_texture_bytes,
            2 * MIB
        );
        drop(renderer);
        assert_eq!(
            render_memory_gauges().backdrop_texture_bytes,
            before.backdrop_texture_bytes
        );

        // The device count is a reading, not a sum: the latest replaces it.
        record_device_allocated_bytes(40 * MIB);
        record_device_allocated_bytes(30 * MIB);
        assert_eq!(render_memory_gauges().device_allocated_bytes, 30 * MIB);
    }
}
