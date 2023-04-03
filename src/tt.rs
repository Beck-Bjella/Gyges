use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::fmt::Display;
use std::mem;
use std::ptr::NonNull;

const CLUSTER_SIZE: usize = 1;

const BYTES_PER_KB: f64 = 1000.0;
const BYTES_PER_MB: f64 = BYTES_PER_KB * 1000.0;
const BYTES_PER_GB: f64 = BYTES_PER_MB * 1000.0;

pub static mut TT_EMPTY_INSERTS: usize = 0;
pub static mut TT_UNSAFE_INSERTS: usize = 0;

// #[derive(PartialEq, Debug)]
// pub enum TTProbeResult {
//     EmptyInsert,
//     ReplaceInsert,
//     Lookup

// }

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
    pub depth: i8,
    pub flag: TTEntryType,
    pub used: bool

}

impl Entry {
    pub fn new(key: u64, score: f64, depth: i8, flag: TTEntryType) -> Entry {
        return Entry {
            key,
            score,
            depth,
            flag,
            used: true
            
        };

    }

    pub const fn empty() -> Entry {
        return Entry {
            key: 0,
            score: 0.0,
            depth: 0,
            flag: TTEntryType::None,
            used: false
            
        };

    }

    pub fn replace(&mut self, entry: Entry) {
        self.key = entry.key;
        self.score = entry.score;
        self.depth = entry.depth;
        self.flag = entry.flag;
        self.used = entry.used;

    }

}

#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub entrys: [Entry; CLUSTER_SIZE],

}

impl Cluster {
    pub const fn empty() -> Cluster {
        return Cluster {
            entrys: [Entry::empty(); CLUSTER_SIZE]

        };

    }

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

    // // Returns ("Lookup", entry) if a corosponding entry for that key is found. 
    // // Returns ("Insert", entry) if There is a open slot for that key.
    // // Returns ("Insert", Most irrelvent entry) if no open slot or corosponding entry is found.
    // pub unsafe fn probe(&self, key: u64) -> (TTProbeResult, &mut Entry) {
    //     let index = ((self.num_clusters() - 1) as u64 & key) as usize;

    //     let cluster = self.get_cluster(index);

    //     for entry_idx in 0..CLUSTER_SIZE {
    //         let entry_ptr = get_entry(cluster, entry_idx);

    //         let entry = &mut (*entry_ptr);

    //         if !entry.used {
    //             TT_EMPTY_INSERTS += 1;
    //             return (TTProbeResult::EmptyInsert, entry);
    
    //         } else if entry.key == key {
    //             TT_SAFE_INSERTS += 1;
    //             return (TTProbeResult::Lookup, entry);

    //         }

    //     }

    //     let mut replacement_ptr = get_entry(cluster, 0);
    //     for entry_idx in 0..CLUSTER_SIZE {
    //         let entry_ptr = get_entry(cluster, entry_idx);

    //         if (*entry_ptr).depth >= (*replacement_ptr).depth {
    //             replacement_ptr = entry_ptr;

    //         }

    //     }

    //     return (TTProbeResult::ReplaceInsert, &mut (*replacement_ptr));

    // }

    pub unsafe fn probe(&self, key: u64) -> (bool, &mut Entry) {
        let index = ((self.num_clusters() - 1) as u64 & key) as usize;

        let cluster = self.get_cluster(index);

        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            let entry = &mut (*entry_ptr);

            if entry.key == key {
                return (true, entry);

            }

        }

        (false, &mut (*get_entry(cluster, 0)))

    }

    pub unsafe fn insert(&self, key: u64) -> (bool, &mut Entry) {
        let index = ((self.num_clusters() - 1) as u64 & key) as usize;

        let cluster = self.get_cluster(index);

        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            let entry = &mut (*entry_ptr);

            if !entry.used {
                TT_EMPTY_INSERTS += 1;
                return (true, entry);
    
            }

        }

        let mut replacement_ptr = get_entry(cluster, 0);
        for entry_idx in 0..CLUSTER_SIZE {
            let entry_ptr = get_entry(cluster, entry_idx);

            if (*entry_ptr).depth >= (*replacement_ptr).depth {
                replacement_ptr = entry_ptr;

            }

        }

        TT_UNSAFE_INSERTS += 1;
        return (false, &mut (*replacement_ptr));

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

// TERRIBLY UNSAFE AND WILL MOST LIKELY CRASH CODE
fn alloc_room(size: usize) -> NonNull<Cluster> {
    unsafe {
        let size = size * mem::size_of::<Cluster>();
        let layout = Layout::from_size_align(size, 8).unwrap();

        let ptr: *mut u8 = alloc::alloc_zeroed(layout);

        let new_ptr: *mut Cluster = ptr.cast();

        return NonNull::new(new_ptr).unwrap();

    }

}
