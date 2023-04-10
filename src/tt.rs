use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::f32::consts::E;
use std::fmt::Display;
use std::mem;
use std::ptr::NonNull;
use std::ptr;

use crate::move_gen::*;
use crate::consts::*;

const CLUSTER_SIZE: usize = 3;

const BYTES_PER_KB: f64 = 1000.0;
const BYTES_PER_MB: f64 = BYTES_PER_KB * 1000.0;
const BYTES_PER_GB: f64 = BYTES_PER_MB * 1000.0;

pub static mut TT_SAFE_INSERTS: usize = 0;
pub static mut TT_UNSAFE_INSERTS: usize = 0;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum TTEntryType {
    ExactValue,
    UpperBound,
    LowerBound,
    None,

}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64,
    pub score: f64,
    pub bestmove: Move,
    pub depth: i8,
    pub flag: TTEntryType,
    pub used: bool

}

impl Entry {
    pub fn new(key: u64, score: f64, depth: i8, bestmove: Move, flag: TTEntryType) -> Entry {
        return Entry {
            key,
            score,
            bestmove,
            depth,
            flag,
            used: true
            
        };

    }

    pub fn replace(&mut self, entry: Entry) {
        self.key = entry.key;
        self.score = entry.score;
        self.bestmove = entry.bestmove;
        self.depth = entry.depth;
        self.flag = entry.flag;
        self.used = entry.used;

    }

}

#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub entrys: [Entry; CLUSTER_SIZE],

}

pub struct TranspositionTable {
    pub clusters: UnsafeCell<NonNull<Cluster>>,
    pub cap: UnsafeCell<usize>,

}

impl TranspositionTable {
    pub fn new(size: usize) -> TranspositionTable {
        return TranspositionTable {
            clusters: UnsafeCell::new(alloc_room(size)),
            cap: UnsafeCell::new(size)

        };
        
    }
    
    pub fn size_kilobytes(&self) -> f64 {
        return (mem::size_of::<Cluster>() * self.num_clusters()) as f64 / BYTES_PER_KB;
        
    }

    pub fn size_megabytes(&self) -> f64 {
        return (mem::size_of::<Cluster>() * self.num_clusters()) as f64 / BYTES_PER_MB;

    }

    pub fn size_gigabytes(&self) -> f64 {
        return (mem::size_of::<Cluster>() * self.num_clusters()) as f64 / BYTES_PER_GB;

    }

    pub fn num_clusters(&self) -> usize {
        return unsafe{ *self.cap.get() };

    }

    pub unsafe fn num_entrys(&self) -> usize {
        return self.num_clusters() * CLUSTER_SIZE;

    }

    unsafe fn get_cluster(&self, i: usize) -> *mut Cluster {
        return (*self.clusters.get()).as_ptr().add(i);

    }

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

    pub unsafe fn insert(&self, new_entry: Entry) -> bool {
        let index = new_entry.key as usize % self.num_clusters();

        let cluster = self.get_cluster(index);

        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            let entry = &mut (*entry_ptr);

            if !entry.used  {
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
        return false;

    }

}

impl Display for TranspositionTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            for cluster_idx in 0..self.num_clusters() {
                let cluster = self.get_cluster(cluster_idx);
                println!("cluster {}", cluster_idx);
    
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
        
        return Result::Ok(());

    }

}

unsafe impl Sync for TranspositionTable {
}

unsafe fn get_entry(cluster: *mut Cluster, i: usize) -> *mut Entry {
    return  ((*cluster).entrys).as_ptr().add(i) as *mut Entry;

} 

unsafe fn cluster_first_entry(cluster: *mut Cluster) -> *mut Entry {
    (*cluster).entrys.get_unchecked_mut(0) as *mut Entry
}

fn alloc_room(size: usize) -> NonNull<Cluster> {
    unsafe {
        let size = size * mem::size_of::<Cluster>();
        let layout = Layout::from_size_align(size, 8).unwrap();

        let ptr: *mut u8 = alloc::alloc_zeroed(layout);

        let new_ptr: *mut Cluster = ptr.cast();

        return NonNull::new(new_ptr).unwrap();

    }

}

pub fn tt() -> &'static TranspositionTable {
    return unsafe { &*(&mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable) };

}

pub fn init_tt() {
    unsafe {
        let tt = &mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable;
        ptr::write(tt, TranspositionTable::new(2usize.pow(24)));

    }

}
