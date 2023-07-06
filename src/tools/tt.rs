use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::fmt::Display;
use std::mem;
use std::ptr::{self, NonNull};

use crate::consts::*;
use crate::moves::moves::*;

const CLUSTER_SIZE: usize = 3;

const BYTES_PER_KB: f64 = 1000.0;
const BYTES_PER_MB: f64 = BYTES_PER_KB * 1000.0;
const BYTES_PER_GB: f64 = BYTES_PER_MB * 1000.0;

pub static mut TT_SAFE_INSERTS: usize = 0;
pub static mut TT_UNSAFE_INSERTS: usize = 0;


/// Defines the bound for a node.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum NodeBound {
    ExactValue,
    UpperBound,
    LowerBound,

}

/// Structure that holds the data of a node.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64,
    pub score: f64,
    pub bestmove: TTMove,
    pub depth: i8,
    pub bound: NodeBound,
    pub used: bool

}

impl Entry {
    pub fn new(key: u64, score: f64, depth: i8, bestmove: TTMove, flag: NodeBound) -> Entry {
        Entry {
            key,
            score,
            bestmove,
            depth,
            bound: flag,
            used: true

        }

    }

    pub fn replace(&mut self, entry: Entry) {
        self.key = entry.key;
        self.score = entry.score;
        self.bestmove = entry.bestmove;
        self.depth = entry.depth;
        self.bound = entry.bound;
        self.used = entry.used;

    }

}

/// Structure that holds multiple entrys and is stored in the trasposition table.
#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub entrys: [Entry; CLUSTER_SIZE],

}

/// Structure for the transposition table.
pub struct TranspositionTable {
    pub clusters: UnsafeCell<NonNull<Cluster>>,
    pub cap: UnsafeCell<usize>,
    
}

impl TranspositionTable {
    pub fn new(size: usize) -> TranspositionTable {
        TranspositionTable {
            clusters: UnsafeCell::new(alloc_room(size)),
            cap: UnsafeCell::new(size),

        }

    }

    /// Gets the max size of the transposition table in kilobytes.
    pub fn size_kilobytes(&self) -> f64 {
        (mem::size_of::<Cluster>() * self.num_clusters()) as f64 / BYTES_PER_KB

    }
    
    /// Gets the max size of the transposition table in megabytes.
    pub fn size_megabytes(&self) -> f64 {
        (mem::size_of::<Cluster>() * self.num_clusters()) as f64 / BYTES_PER_MB

    }

    /// Gets the max size of the transposition table in gigabytes.
    pub fn size_gigabytes(&self) -> f64 {
        (mem::size_of::<Cluster>() * self.num_clusters()) as f64 / BYTES_PER_GB
        
    }

    /// Gets the max number of culusters in the transposition table.
    pub fn num_clusters(&self) -> usize {
        unsafe { *self.cap.get() }

    }

    /// Gets the max number of entrys in the transposition table.
    pub fn num_entrys(&self) -> usize {
        self.num_clusters() * CLUSTER_SIZE

    }

    /// Returns a raw pointer to a specific cluster in the table.
    fn get_cluster(&self, i: usize) -> *mut Cluster {
        unsafe{ (*self.clusters.get()).as_ptr().add(i) }

    }

    /// Probes the transposition table for the data corosponding to a specific key.
    pub unsafe fn probe(&self, key: u64) -> (bool, &mut Entry) {
        let index = key as usize % self.num_clusters();

        let cluster = self.get_cluster(index);
        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            let entry = &mut (*entry_ptr);

            if entry.key == key {
                return (true, entry);

            }

        }

        (false, &mut (*cluster_first_entry(cluster)))

    }

    /// Uses a key and inserts a entry into the table in the best available spot.
    pub unsafe fn insert(&self, new_entry: Entry) -> bool {
        let index = new_entry.key as usize % self.num_clusters();

        let cluster = self.get_cluster(index);

        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            let entry = &mut (*entry_ptr);

            if !entry.used {
                TT_SAFE_INSERTS += 1;
                entry.replace(new_entry);
                return true;

            }

            if entry.key == new_entry.key && new_entry.depth >= entry.depth {
                TT_SAFE_INSERTS += 1;
                entry.replace(new_entry);
                return true;

            }

        }

        let mut replacement_ptr = cluster_first_entry(cluster);
        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            if (*entry_ptr).depth <= (*replacement_ptr).depth {
                replacement_ptr = entry_ptr;

            }

        }

        let replacement_entry = &mut (*replacement_ptr);

        TT_UNSAFE_INSERTS += 1;
        replacement_entry.replace(new_entry);
        false

    }

}

impl Display for TranspositionTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            for cluster_idx in 0..self.num_clusters() {
                let cluster = self.get_cluster(cluster_idx);
                writeln!(f, "cluster {}", cluster_idx)?;

                for entry_idx in 0..CLUSTER_SIZE {
                    let entry = get_entry(cluster, entry_idx);

                    if (*entry).used {
                        writeln!(f, "  - {:?}", *entry)?;
                        
                    } else {
                        writeln!(f, "  - NONE")?;

                    }

                }

            }

        }

        Result::Ok(())

    }

}

unsafe impl Sync for TranspositionTable {}

/// Returns a raw pointer 
fn get_entry(cluster: *mut Cluster, i: usize) -> *mut Entry {
    unsafe{ ((*cluster).entrys).as_ptr().add(i) as *mut Entry }

}

unsafe fn cluster_first_entry(cluster: *mut Cluster) -> *mut Entry {
    (*cluster).entrys.get_unchecked_mut(0) as *mut Entry

}

/// Allocates empty memory as clusters and returns a pointer to it.
fn alloc_room(size: usize) -> NonNull<Cluster> {
    unsafe {
        let size = size * mem::size_of::<Cluster>();
        let layout = Layout::from_size_align(size, 8).unwrap();

        let ptr: *mut u8 = alloc::alloc_zeroed(layout);

        let new_ptr: *mut Cluster = ptr.cast();

        NonNull::new(new_ptr).unwrap()

    }

}

/// Returns acess to the global transposition table.
pub fn tt() -> &'static TranspositionTable {
    unsafe { &*(&mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable) }

}

/// Initalizes the global transposition table.
/// 
/// Size must be a power of 2
pub fn init_tt(size: usize) {
    unsafe {
        let tt = &mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable;
        ptr::write(tt, TranspositionTable::new(size));

    }

}
