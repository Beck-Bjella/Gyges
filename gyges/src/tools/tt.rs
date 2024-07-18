//! The transposition table and the other helper structures for it.
//! 
//! A transposition table is a type of HashTable that maps Zobrist keys to data about that position.
//! This is a very acommon technique in chess engines and other board game AI. 
//! Its primary purpose is reduce the number of nodes that need to be searched by storing the data about a position
//! so that it does not need to be re-calculted.
//! 
//! Specifically, this transposition table is lockless and can be accessed by multiple threads at the same time. 
//! This means that overwriting data is possible, but it is unlikely. It also uses clusers to store multiple entrys that share the same key. 
//!
//! Zobrist hashing is used to generate the keys for the transposition table, and the keys are generated in the board module. 
//! This hashing technique can lead to collisions in the table (multiple positions having the same key), but this is very unlikely.
//! 
//! Credit to [Pleco chess engine](https://github.com/pleco-rs/Pleco) for huge insperaration for this file
//!

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::fmt::Display;
use std::mem;
use std::ptr::NonNull;

use crate::moves::*;

/// The number of entrys in a cluster.
const CLUSTER_SIZE: usize = 3;

/// Bytes per kilobyte.
const BYTES_PER_KB: f64 = 1000.0;
/// Bytes per megabyte.
const BYTES_PER_MB: f64 = BYTES_PER_KB * 1000.0;
/// Bytes per gigabyte.
const BYTES_PER_GB: f64 = BYTES_PER_MB * 1000.0;

/// Counter for the number of safe inserts into the transposition table.
pub static mut TT_SAFE_INSERTS: usize = 0;
/// Counter for the number of unsafe inserts into the transposition table.
pub static mut TT_UNSAFE_INSERTS: usize = 0;


/// Defines the bound for a node.
/// This is the same as the chess concept of a node type. [Chess Wiki](https://www.chessprogramming.org/Node_Types).
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum NodeBound {
    ExactValue,
    UpperBound,
    LowerBound,

}

/// Structure that holds the data of a node.
/// Stores the key, score, best move, depth, and bound.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64,
    pub score: f64,
    pub bestmove: Move,
    pub depth: i8,
    pub bound: NodeBound,
    pub used: bool,

}

impl Entry {
    /// Creates a new entry from its components.
    pub fn new(key: u64, score: f64, depth: i8, bestmove: Move, flag: NodeBound) -> Entry {
        Entry {
            key,
            score,
            bestmove,
            depth,
            bound: flag,
            used: true

        }

    }

    /// Replaces the data with another entrys data.
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
/// Each of the entrys that are stored in the cluster are share the same key.
#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub entrys: [Entry; CLUSTER_SIZE],

}

/// Structure representing a transposition table.
/// A transposition table is a type of HashTable that maps Zobrist keys to the data about that position. 
/// The data that is stored in [entrys]
/// 
/// [entrys]:
pub struct TranspositionTable {
    pub clusters: UnsafeCell<NonNull<Cluster>>,
    pub cap: UnsafeCell<usize>,
    
}

impl TranspositionTable {
    /// Creates a new TranspostionTable with a specific size.
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

    /// De-allocates the current heap.
    pub unsafe fn de_alloc(&self) {
        let layout = Layout::from_size_align(*self.cap.get(), 2).unwrap();
        let ptr: *mut u8 = mem::transmute(*self.clusters.get());
    
        alloc::dealloc(ptr, layout);
    }

    /// Resets the transposition table.
    /// Completly realloctes to empty memory.
    pub unsafe fn reset(&self) {
        self.de_alloc();
        *self.clusters.get() = alloc_room(*self.cap.get());

    
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

/// Returns a raw pointer to a specific entry in a cluster.
fn get_entry(cluster: *mut Cluster, i: usize) -> *mut Entry {
    unsafe{ ((*cluster).entrys).as_ptr().add(i) as *mut Entry }

}

/// Returns a raw pointer to the first entry in a cluster.
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
