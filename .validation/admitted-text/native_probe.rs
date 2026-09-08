use std::{ffi::{c_char, c_void, CString}, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};

#[repr(C)]
#[derive(Default)]
pub struct Zone { pub blocks: u64, pub live: u64, pub high: u64, pub allocated: u64 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Sentinel { pub family: u64, pub address: u64, pub size: u64, pub released: u64 }
#[repr(C)]
struct Identity { postscript: [c_char; 256], version: [c_char; 256], path: [c_char; 1024], coretext: [c_char; 1024] }
unsafe extern "C" {
    fn probe_pool_push() -> *mut c_void;
    fn probe_pool_pop(pool: *mut c_void);
    fn probe_zone_snapshot(output: *mut Zone);
    fn probe_phase_marker(ordinal: usize) -> *mut c_void;
    fn probe_release_marker(pointer: *mut c_void);
    fn probe_allocation_sentinels(events: *mut Sentinel, capacity: usize) -> i32;
    fn probe_font_identity(postscript: *const c_char, output: *mut Identity) -> i32;
}

pub struct Pool(*mut c_void);
impl Pool {
    pub fn new() -> Result<Self, String> {
        let pool = unsafe { probe_pool_push() };
        if pool.is_null() { Err("autorelease pool allocation failed".into()) } else { Ok(Self(pool)) }
    }
}
impl Drop for Pool { fn drop(&mut self) { unsafe { probe_pool_pop(self.0) }; } }

#[derive(Debug)]
pub struct ProbeState { markers: Mutex<[usize; 32]>, next: AtomicUsize, overflow: AtomicUsize, fonts: Mutex<([[u8; 256]; 16], [usize; 16], usize)> }
impl Default for ProbeState {
    fn default() -> Self { Self { markers: Mutex::new([0; 32]), next: AtomicUsize::new(0), overflow: AtomicUsize::new(0), fonts: Mutex::new(([[0; 256]; 16], [0; 16], 0)) } }
}
impl ProbeState {
    pub fn marker(&self, name: &str) {
        let ordinal = self.next.fetch_add(1, Ordering::SeqCst);
        if ordinal >= 32 { self.overflow.store(1, Ordering::SeqCst); return; }
        let pointer = unsafe { probe_phase_marker(ordinal) };
        if pointer.is_null() { self.overflow.store(1, Ordering::SeqCst); return; }
        match self.markers.lock() {
            Ok(mut markers) => markers[ordinal] = pointer as usize,
            Err(_) => { unsafe { probe_release_marker(pointer) }; self.overflow.store(1, Ordering::SeqCst); return; }
        }
        println!("MARKER {ordinal} {name} 0x{:x} {}", pointer as usize, 1009 + ordinal * 16);
    }
    pub fn record_font(&self, name: &str) {
        let Ok(mut fonts) = self.fonts.lock() else { self.overflow.store(1, Ordering::SeqCst); return; };
        if (0..fonts.2).any(|index| &fonts.0[index][..fonts.1[index]] == name.as_bytes()) { return; }
        let index = fonts.2;
        if index >= 16 || name.len() >= 256 { self.overflow.store(1, Ordering::SeqCst); return; }
        fonts.0[index][..name.len()].copy_from_slice(name.as_bytes());
        fonts.1[index] = name.len();
        fonts.2 += 1;
    }
    pub fn print_font_identities(&self) -> Result<(), String> {
        let fonts = self.fonts.lock().map_err(|_| "font directory poisoned")?;
        for index in 0..fonts.2 {
            let name = std::str::from_utf8(&fonts.0[index][..fonts.1[index]]).map_err(|error| error.to_string())?;
            font_identity(name)?;
        }
        Ok(())
    }
    pub fn is_valid(&self) -> bool { self.overflow.load(Ordering::SeqCst) == 0 }
}
impl Drop for ProbeState {
    fn drop(&mut self) {
        if let Ok(markers) = self.markers.get_mut() {
            for address in markers.iter().copied().filter(|address| *address != 0) {
                unsafe { probe_release_marker(address as *mut c_void) };
            }
        }
    }
}

pub fn zone() -> Zone { let mut zone = Zone::default(); unsafe { probe_zone_snapshot(&mut zone) }; zone }
pub fn sentinels() -> Result<(), String> {
    let mut events = [Sentinel::default(); 7];
    let count = unsafe { probe_allocation_sentinels(events.as_mut_ptr(), events.len()) };
    if count != 7 { return Err(format!("allocation sentinel failed with code {count}")); }
    for event in events { println!("SENTINEL {} 0x{:x} {} {}", event.family, event.address, event.size, event.released); }
    Ok(())
}

pub fn font_identity(postscript: &str) -> Result<(), String> {
    let name = CString::new(postscript).map_err(|error| error.to_string())?;
    let mut identity = Identity { postscript: [0; 256], version: [0; 256], path: [0; 1024], coretext: [0; 1024] };
    let status = unsafe { probe_font_identity(name.as_ptr(), &mut identity) };
    if status != 0 { return Err(format!("font identity lookup failed: {status}")); }
    for (field, bytes) in [("postscript", identity.postscript.as_slice()), ("version", identity.version.as_slice()),
        ("path", identity.path.as_slice()), ("coretext", identity.coretext.as_slice())] {
        let value: Vec<u8> = bytes.iter().take_while(|byte| **byte != 0).map(|byte| *byte as u8).collect();
        println!("IDENTITY {field} {:?}", String::from_utf8_lossy(&value));
    }
    Ok(())
}
