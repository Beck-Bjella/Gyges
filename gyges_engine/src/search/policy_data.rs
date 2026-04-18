//! Policy training-data collection.
//!
//! Observational hook: after each iterative-deepening search completes,
//! appends one CSV row capturing the root position and every root move
//! that received a real score during search.
//!
//! Columns:
//!   pieces — 36 ints (raw board, root is always Player::One's turn)
//!   moves  — JSON array of [src, pickup_or_null, dest, score] tuples
//!
//! Opens the target path in append mode so many `go` commands (and many
//! engine sessions configured to the same path) accumulate into one file.
//! Header is written only when the file is new/empty. Parallel workers
//! should be pointed at distinct paths to avoid interleaved writes.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use gyges::board::BoardState;
use gyges::moves::{MoveType, RootMove};

use crate::search::LOSS_THRESHOLD;

pub struct PolicyDataLogger {
    file_path: PathBuf,
    write_lock: Mutex<()>,
}

impl PolicyDataLogger {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        if let Some(parent) = file_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        // Touch the file so it exists even if no rows get written.
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        Ok(Self {
            file_path,
            write_lock: Mutex::new(()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.file_path
    }

    pub fn record(&self, board: &BoardState, root_moves: &[RootMove]) -> io::Result<()> {
        let _guard = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)?;

        // pieces: 36 comma-separated ints (0 empty, else Piece as u8 + 1)
        f.write_all(b"\"")?;
        for sq in 0..36usize {
            if sq > 0 {
                f.write_all(b",")?;
            }
            let bit = 1u64 << sq;
            let v = if board.piece_bb.0 & bit != 0 {
                board.data[sq] as u8 + 1
            } else {
                0
            };
            write!(f, "{}", v)?;
        }
        f.write_all(b"\",\"[")?;

        // moves: [[src, pickup_or_null, dest, score], ...] — scored, non-losing moves only
        let mut first = true;
        for rm in root_moves {
            if rm.ply == 0 || rm.score <= LOSS_THRESHOLD {
                continue;
            }
            if !first {
                f.write_all(b",")?;
            }
            first = false;
            let mv = &rm.mv;
            match mv.flag {
                MoveType::Bounce => {
                    let src = mv.data[0].1 .0;
                    let dest = mv.data[1].1 .0;
                    write!(f, "[{},null,{},{}]", src, dest, rm.score)?;
                }
                MoveType::Drop => {
                    let src = mv.data[0].1 .0;
                    let pickup = mv.data[1].1 .0;
                    let dest = mv.data[2].1 .0;
                    write!(f, "[{},{},{},{}]", src, pickup, dest, rm.score)?;
                }
                MoveType::None => {}
            }
        }
        f.write_all(b"]\"\n")?;
        Ok(())
    }
}
