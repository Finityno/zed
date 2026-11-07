use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use gpui_util::ResultExt;
use itertools::Itertools;
use windows::Win32::{
    Foundation::HMODULE,
    Graphics::{
        Direct3D::{
            D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
        },
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG,
            D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS,
            D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
        },
        Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_DEBUG, DXGI_CREATE_FACTORY_FLAGS,
            IDXGIAdapter1, IDXGIFactory6,
        },
    },
};
use windows::core::Interface;

pub(crate) fn try_to_recover_from_device_lost<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    (0..5)
        .map(|i| {
            if i > 0 {
                // Add a small delay before retrying
                std::thread::sleep(std::time::Duration::from_millis(100 + i * 10));
            }
            f()
        })
        .find_or_last(Result::is_ok)
        .unwrap()
        .context("DirectXRenderer failed to recover from lost device after multiple attempts")
}

/// How many device recoveries in a row are tried straight after one another
/// before the next ones wait for [`SLOW_RECOVERY_INTERVAL`].
pub(crate) const FAST_RECOVERY_ATTEMPTS: u32 = 5;

/// How long a recovery that keeps failing waits before it is tried again.
/// Each attempt spends most of a second waiting on the driver, so a driver
/// that takes longer than the fast attempts to come back (an update, a long
/// TDR) would otherwise outlast them, and every window would stay blank
/// until the app restarts although the driver was fine seconds later.
pub(crate) const SLOW_RECOVERY_INTERVAL: Duration = Duration::from_secs(5);

/// What a window's message handler answers `WM_GPUI_GPU_DEVICE_LOST` with
/// when its renderer could not be rebuilt on the new devices, so that the
/// recovery that sent them counts as failed and is tried again.
pub(crate) const DEVICE_REBUILD_FAILED: isize = 1;

/// The device recoveries that failed since the last one that worked.
///
/// A driver still resetting can lose the devices it just handed out, so a
/// failed recovery is tried again at once a few times. One that keeps
/// failing is tried every [`SLOW_RECOVERY_INTERVAL`] from then on, for as
/// long as it takes: the clock thread neither spins on a driver that is not
/// back yet nor gives up on one that comes back.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum RecoveryAttempts {
    /// Nothing failed since the last recovery that worked (or ever).
    #[default]
    Settled,
    /// `failed` recoveries in a row failed; the next is tried at once.
    Retrying { failed: u32 },
    /// The fast attempts are used up; the next is tried at `next_attempt`.
    BackingOff { failed: u32, next_attempt: Instant },
}

/// What a failed recovery leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FailedRecovery {
    /// Within the fast attempts: the next is tried at once.
    RetryNow { failed: u32 },
    /// The fast attempts just ran out; the next waits until `next_attempt`.
    BackoffStarted { next_attempt: Instant },
    /// Failed again while backing off; the next waits until `next_attempt`.
    BackingOff { failed: u32, next_attempt: Instant },
}

impl RecoveryAttempts {
    /// Counts a failed recovery at `now` and says when the next is tried.
    pub(crate) fn failed(&mut self, now: Instant) -> FailedRecovery {
        let failed = self.failures().saturating_add(1);
        if failed < FAST_RECOVERY_ATTEMPTS {
            *self = Self::Retrying { failed };
            return FailedRecovery::RetryNow { failed };
        }
        let next_attempt = now + SLOW_RECOVERY_INTERVAL;
        let started = !matches!(self, Self::BackingOff { .. });
        *self = Self::BackingOff {
            failed,
            next_attempt,
        };
        if started {
            FailedRecovery::BackoffStarted { next_attempt }
        } else {
            FailedRecovery::BackingOff {
                failed,
                next_attempt,
            }
        }
    }

    /// Settles after a recovery that worked, returning how many failed
    /// before it, if any did.
    pub(crate) fn recovered(&mut self) -> Option<u32> {
        let failed = self.failures();
        *self = Self::Settled;
        (failed > 0).then_some(failed)
    }

    /// Whether a recovery is due at `now`. `needed` says whether one is
    /// (a lost device, a window asking for new devices) and is consulted only
    /// while nothing has failed: while backing off, a window that fails to
    /// resize on the lost devices asks at every resize, and the attempt
    /// already scheduled answers it.
    pub(crate) fn due(&self, now: Instant, needed: impl FnOnce() -> bool) -> bool {
        match *self {
            Self::Settled => needed(),
            Self::Retrying { .. } => true,
            Self::BackingOff { next_attempt, .. } => next_attempt <= now,
        }
    }

    /// When the next recovery is tried, while backing off.
    pub(crate) fn backoff_until(&self) -> Option<Instant> {
        match *self {
            Self::BackingOff { next_attempt, .. } => Some(next_attempt),
            Self::Settled | Self::Retrying { .. } => None,
        }
    }

    /// Brings a recovery that is backing off forward to `now`: the displays
    /// changing is a sign the driver may be back.
    pub(crate) fn expedite(&mut self, now: Instant) {
        if let Self::BackingOff { next_attempt, .. } = self {
            *next_attempt = (*next_attempt).min(now);
        }
    }

    pub(crate) fn failures(&self) -> u32 {
        match *self {
            Self::Settled => 0,
            Self::Retrying { failed } | Self::BackingOff { failed, .. } => failed,
        }
    }
}

#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl DirectXDevices {
    pub(crate) fn new() -> Result<Self> {
        let debug_layer_available = check_debug_layer_available();
        let dxgi_factory =
            get_dxgi_factory(debug_layer_available).context("Creating DXGI factory")?;
        let (adapter, device, device_context, feature_level) =
            get_adapter(&dxgi_factory, debug_layer_available).context("Getting DXGI adapter")?;
        match feature_level {
            D3D_FEATURE_LEVEL_11_1 => {
                log::info!("Created device with Direct3D 11.1 feature level.")
            }
            D3D_FEATURE_LEVEL_11_0 => {
                log::info!("Created device with Direct3D 11.0 feature level.")
            }
            D3D_FEATURE_LEVEL_10_1 => {
                log::info!("Created device with Direct3D 10.1 feature level.")
            }
            _ => unreachable!(),
        }

        Ok(Self {
            adapter,
            dxgi_factory,
            device,
            device_context,
        })
    }
}

#[inline]
fn check_debug_layer_available() -> bool {
    #[cfg(debug_assertions)]
    {
        use windows::Win32::Graphics::Dxgi::{DXGIGetDebugInterface1, IDXGIInfoQueue};

        unsafe { DXGIGetDebugInterface1::<IDXGIInfoQueue>(0) }
            .log_err()
            .is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

#[inline]
fn get_dxgi_factory(debug_layer_available: bool) -> Result<IDXGIFactory6> {
    let factory_flag = if debug_layer_available {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        #[cfg(debug_assertions)]
        log::warn!(
            "Failed to get DXGI debug interface. DirectX debugging features will be disabled."
        );
        DXGI_CREATE_FACTORY_FLAGS::default()
    };
    unsafe { Ok(CreateDXGIFactory2(factory_flag)?) }
}

#[inline]
fn get_adapter(
    dxgi_factory: &IDXGIFactory6,
    debug_layer_available: bool,
) -> Result<(
    IDXGIAdapter1,
    ID3D11Device,
    ID3D11DeviceContext,
    D3D_FEATURE_LEVEL,
)> {
    for adapter_index in 0.. {
        let adapter: IDXGIAdapter1 = unsafe { dxgi_factory.EnumAdapters(adapter_index)?.cast()? };
        if let Ok(desc) = unsafe { adapter.GetDesc1() } {
            let gpu_name = String::from_utf16_lossy(&desc.Description)
                .trim_matches(char::from(0))
                .to_string();
            log::info!("Using GPU: {}", gpu_name);
        }
        // Check to see whether the adapter supports Direct3D 11 and create
        // the device if it does.
        let mut context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL::default();
        if let Some(device) = get_device(
            &adapter,
            Some(&mut context),
            Some(&mut feature_level),
            debug_layer_available,
        )
        .log_err()
        {
            return Ok((adapter, device, context.unwrap(), feature_level));
        }
    }

    unreachable!()
}

#[inline]
fn get_device(
    adapter: &IDXGIAdapter1,
    context: Option<*mut Option<ID3D11DeviceContext>>,
    feature_level: Option<*mut D3D_FEATURE_LEVEL>,
    debug_layer_available: bool,
) -> Result<ID3D11Device> {
    let mut device: Option<ID3D11Device> = None;
    let device_flags = if debug_layer_available {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT
    };
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            device_flags,
            // 4x MSAA is required for Direct3D Feature Level 10.1 or better
            Some(&[
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
            ]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            feature_level,
            context,
        )?;
    }
    let device = device.unwrap();
    let mut data = D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS::default();
    unsafe {
        device
            .CheckFeatureSupport(
                D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
                &mut data as *mut _ as _,
                std::mem::size_of::<D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS>() as u32,
            )
            .context("Checking GPU device feature support")?;
    }
    if data
        .ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x
        .as_bool()
    {
        Ok(device)
    } else {
        Err(anyhow::anyhow!(
            "Required feature StructuredBuffer is not supported by GPU/driver"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{FAST_RECOVERY_ATTEMPTS, FailedRecovery, RecoveryAttempts, SLOW_RECOVERY_INTERVAL};
    use std::time::{Duration, Instant};

    fn backing_off(start: Instant) -> RecoveryAttempts {
        let mut attempts = RecoveryAttempts::default();
        for _ in 0..FAST_RECOVERY_ATTEMPTS {
            attempts.failed(start);
        }
        attempts
    }

    #[test]
    fn recovery_retries_at_once_then_backs_off() {
        let start = Instant::now();
        let mut attempts = RecoveryAttempts::default();
        assert!(!attempts.due(start, || false));
        assert!(attempts.due(start, || true));
        for failed in 1..FAST_RECOVERY_ATTEMPTS {
            assert_eq!(attempts.failed(start), FailedRecovery::RetryNow { failed });
            assert!(attempts.due(start, || false));
            assert_eq!(attempts.backoff_until(), None);
        }
        let next_attempt = start + SLOW_RECOVERY_INTERVAL;
        assert_eq!(attempts.failed(start), FailedRecovery::BackoffStarted { next_attempt });
        assert_eq!(attempts.backoff_until(), Some(next_attempt));
        assert_eq!(attempts.failures(), FAST_RECOVERY_ATTEMPTS);
    }

    #[test]
    fn recovery_backing_off_waits_for_its_interval_whatever_asks() {
        let start = Instant::now();
        let attempts = backing_off(start);
        assert!(!attempts.due(start, || true));
        let just_before = start + SLOW_RECOVERY_INTERVAL - Duration::from_millis(1);
        assert!(!attempts.due(just_before, || true));
        assert!(attempts.due(start + SLOW_RECOVERY_INTERVAL, || false));
    }

    #[test]
    fn recovery_backing_off_never_gives_up() {
        let start = Instant::now();
        let mut attempts = backing_off(start);
        let mut now = start;
        for round in 1..=1000 {
            now += SLOW_RECOVERY_INTERVAL;
            assert!(attempts.due(now, || false));
            let next_attempt = now + SLOW_RECOVERY_INTERVAL;
            assert_eq!(
                attempts.failed(now),
                FailedRecovery::BackingOff {
                    failed: FAST_RECOVERY_ATTEMPTS + round,
                    next_attempt,
                }
            );
            assert_eq!(attempts.backoff_until(), Some(next_attempt));
        }
    }

    #[test]
    fn recovery_that_works_resets_the_attempts() {
        let start = Instant::now();
        let mut attempts = RecoveryAttempts::default();
        assert_eq!(attempts.recovered(), None);
        attempts.failed(start);
        attempts.failed(start);
        assert_eq!(attempts.recovered(), Some(2));
        assert_eq!(attempts, RecoveryAttempts::Settled);

        let mut attempts = backing_off(start);
        assert_eq!(attempts.recovered(), Some(FAST_RECOVERY_ATTEMPTS));
        assert!(!attempts.due(start, || false));
        assert_eq!(attempts.backoff_until(), None);
        assert_eq!(attempts.failed(start), FailedRecovery::RetryNow { failed: 1 });
    }

    #[test]
    fn recovery_expedited_only_while_backing_off() {
        let start = Instant::now();
        let mut attempts = backing_off(start);
        let changed_at = start + Duration::from_secs(1);
        attempts.expedite(changed_at);
        assert_eq!(attempts.backoff_until(), Some(changed_at));
        assert!(attempts.due(changed_at, || false));
        attempts.expedite(start + Duration::from_secs(60));
        assert_eq!(attempts.backoff_until(), Some(changed_at));

        let mut attempts = RecoveryAttempts::default();
        attempts.expedite(start);
        assert_eq!(attempts, RecoveryAttempts::Settled);
        attempts.failed(start);
        attempts.expedite(start);
        assert_eq!(attempts, RecoveryAttempts::Retrying { failed: 1 });
    }
}
