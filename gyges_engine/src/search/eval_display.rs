// eval_display.rs

use crate::search::{EvaluationContext, PieceData, Piece};

// ── Cell display config ───────────────────────────────────────────────────────
// Adjust these two constants to change cell size.
// CW must be large enough to fit your largest value + sign + spaces.
// A value like -0.123 needs at minimum: 1 (sign) + 1 (digit) + 1 (.) + 3 (decimals) = 6
// Plus 2 spaces padding = CW 8. Increase CW for larger numbers.
const CW: usize = 8;         // total cell width in chars
const DECIMALS: usize = 3;   // decimal places shown in each cell

// ── Public colour helpers ─────────────────────────────────────────────────────

pub fn heat_rgb(t: f64) -> (u8, u8, u8) {
    let r = (255.0 * t.powf(0.6)).round() as u8;
    let g = (200.0 * (1.0 - (2.0 * t - 1.0).powi(2))).round() as u8;
    let b = (255.0 * (1.0 - t).powf(0.6)).round() as u8;
    (r, g, b)
}

pub fn diverge_rgb(t: f64) -> (u8, u8, u8) {
    if t < 0.5 {
        let f = 1.0 - t * 2.0;
        ((200.0 + 55.0 * f) as u8, (80.0 * (1.0 - f)) as u8, (80.0 * (1.0 - f)) as u8)
    } else {
        let f = (t - 0.5) * 2.0;
        ((80.0 * (1.0 - f)) as u8, (130.0 + 80.0 * f) as u8, (80.0 * (1.0 - f)) as u8)
    }
}

pub fn normalise(vals: &[f64]) -> impl Fn(f64) -> f64 {
    let finite: Vec<f64> = vals.iter().cloned().filter(|v| v.is_finite()).collect();
    let vmin = finite.iter().cloned().fold(f64::INFINITY,     f64::min);
    let vmax = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (vmax - vmin).abs() < 1e-9 { 1.0 } else { vmax - vmin };
    move |v: f64| ((v - vmin) / range).clamp(0.0, 1.0)
}

// ── EvaluationContext ─────────────────────────────────────────────────────────

impl EvaluationContext {
    pub fn print_extra(&self) {
        println!("=======================================================================");
        println!("============================== HEAT MAPS ==============================");
        println!("=======================================================================");

        // ── helpers ───────────────────────────────────────────────────────────

        let heat = |f: fn(&PieceData) -> f64| -> Vec<(usize, f64, (u8,u8,u8))> {
            let vals: Vec<f64> = self.piece_data.iter().map(|pd| f(pd)).collect();
            let norm = normalise(&vals);
            self.piece_data.iter().map(|pd| {
                let v = f(pd);
                let c = if v.is_finite() { heat_rgb(norm(v)) } else { (40,40,40) };
                (pd.sq.0 as usize, v, c)
            }).collect()
        };

        let div = |f: fn(&PieceData) -> f64| -> Vec<(usize, f64, (u8,u8,u8))> {
            let vals: Vec<f64> = self.piece_data.iter().map(|pd| f(pd)).collect();
            let norm = normalise(&vals);
            self.piece_data.iter().map(|pd| {
                let v = f(pd);
                let c = if v.is_finite() { diverge_rgb(norm(v)) } else { (40,40,40) };
                (pd.sq.0 as usize, v, c)
            }).collect()
        };

        // ── heatmaps ──────────────────────────────────────────────────────────

        // control
        self.print_heatmap("Control Map", (0..36).map(|sq| {
            let bit = 1u64 << sq;
            let p1  = (self.unique_piece_control[0].0 & bit) != 0;
            let p2  = (self.unique_piece_control[1].0 & bit) != 0;
            let sh  = (self.shared_piece_control.0    & bit) != 0;
            let (v, c) = match (p1, p2, sh) {
                (_, _, true) => (3.0, (160, 100, 200)),
                (true, _, _) => (1.0, (92,  180, 130)),
                (_, true, _) => (2.0, (200, 90,  90 )),
                _            => (0.0, (40,  40,  40 )),
            };
            (sq, v, c)
        }).collect());

        // self.print_heatmap("Stop Power", heat(|pd| pd.stop_power[0]));
        // self.print_heatmap("Stop Power", heat(|pd| pd.stop_power[1]));
        // self.print_heatmap("FLOW P1", heat(|pd| pd.flow_percentage[0]));
        // self.print_heatmap("FLOW P2", heat(|pd| pd.flow_percentage[1]));

        // Ownership map — only for shared pieces, color = P1 vs P2 dominance, value = P1 ownership
        self.print_heatmap("Ownership (P1 share of shared pieces)", self.piece_data.iter().filter_map(|pd| {
            if !pd.shared { return None; }
            let d1 = pd.path_min_depths[0];
            let d2 = pd.path_min_depths[1];
            let a1 = if d1.is_finite() { 1.0 / d1.max(1.0) } else { 0.0 };
            let a2 = if d2.is_finite() { 1.0 / d2.max(1.0) } else { 0.0 };
            let total = a1 + a2;
            let ownership = if total == 0.0 { 0.5 } else { a1 / total };
            let c = diverge_rgb(ownership);
            Some((pd.sq.0 as usize, ownership, c))
        }).collect());
        
    }

    /// Generic heatmap.
    /// data: Vec<(square_index, value, (r,g,b))>
    /// Colour is fully decided before calling — build it however you like.
    pub fn print_heatmap(&self, label: &str, data: Vec<(usize, f64, (u8,u8,u8))>) {
        let size = 6usize;

        let mut grid: Vec<Option<(f64, (u8,u8,u8))>> = vec![None; size * size];
        for (sq, v, c) in data {
            if sq < size * size { grid[sq] = Some((v, c)); }
        }

        let vals: Vec<f64> = grid.iter()
            .filter_map(|g| g.map(|(v,_)| v))
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() { return; }

        let vmin = vals.iter().cloned().fold(f64::INFINITY,     f64::min);
        let vmax = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let grid_w   = size * (CW + 1) + 1;
        let legend_w = 32 * 2 + 6;
        let total_w  = legend_w.max(grid_w);
        let grid_off = (total_w - grid_w) / 2;
        let leg_off  = (total_w - legend_w) / 2;
        let indent   = 4usize;
        let goal_x   = indent + grid_off + grid_w / 2;

        let border = |left: &str, mid: &str, right: &str| {
            let pad = " ".repeat(indent + grid_off);
            let inner: String = (0..size)
                .map(|i| format!("{}{}", "─".repeat(CW), if i < size-1 { mid } else { right }))
                .collect();
            println!("\x1b[38;2;50;50;50m{}{}{}\x1b[0m", pad, left, inner);
        };

        // header
        println!();
        println!("{}\x1b[1m\x1b[38;2;210;210;210m{}\x1b[0m  \x1b[38;2;70;70;70m[{:.3} … {:.3}]\x1b[0m",
            " ".repeat(indent), label, vmin, vmax);
        println!();

        // P2 goal
        println!("{}\x1b[38;2;70;70;70m{}\x1b[0m", " ".repeat(goal_x), goal_str(self.board.data[37]));
        println!("{}\x1b[38;2;50;50;50m│\x1b[0m",  " ".repeat(goal_x));
        border("┌","┬","┐");

        for row in (0..size).rev() {
            for pass in 0..3usize {
                print!("{}\x1b[38;2;50;50;50m│\x1b[0m", " ".repeat(indent + grid_off));
                for col in 0..size {
                    let sq = row * size + col;
                    match grid[sq] {
                        Some((v, (r,g,b))) if v.is_finite() => {
                            if pass == 1 {
                                let luma = 0.299*r as f64 + 0.587*g as f64 + 0.114*b as f64;
                                let (fr,fg,fb) = if luma > 140.0 { (20u8,20,20) } else { (220u8,220,220) };
                                // {:>+width$.DECIMALS} includes sign in width — no overflow
                                let num = format!("{:>+width$.dec$}", v, width = CW - 2, dec = DECIMALS);
                                print!("\x1b[48;2;{r};{g};{b}m \x1b[38;2;{fr};{fg};{fb}m{num} \x1b[0m");
                            } else {
                                print!("\x1b[48;2;{r};{g};{b}m{}\x1b[0m", " ".repeat(CW));
                            }
                        }
                        Some(_) => {
                            if pass == 1 {
                                let num = format!("{:^width$}", "NaN", width = CW - 2);
                                print!("\x1b[48;2;22;22;22m \x1b[38;2;70;70;70m{num} \x1b[0m");
                            } else {
                                print!("\x1b[48;2;22;22;22m{}\x1b[0m", " ".repeat(CW));
                            }
                        }
                        None => print!("\x1b[48;2;22;22;22m{}\x1b[0m", " ".repeat(CW)),
                    }
                    print!("\x1b[38;2;50;50;50m│\x1b[0m");
                }
                println!();
            }
            if row > 0 { border("├","┼","┤"); }
        }

        border("└","┴","┘");

        // P1 goal
        println!("{}\x1b[38;2;50;50;50m│\x1b[0m",  " ".repeat(goal_x));
        println!("{}\x1b[38;2;70;70;70m{}\x1b[0m", " ".repeat(goal_x), goal_str(self.board.data[36]));
        println!();

    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn goal_str(piece: Piece) -> String {
    match piece {
        Piece::None  => "◇  empty".to_string(),
        Piece::One   => "\x1b[1m\x1b[38;2;200;169;110m1\x1b[0m\x1b[38;2;70;70;70m  1-ring".to_string(),
        Piece::Two   => "\x1b[1m\x1b[38;2;200;169;110m2\x1b[0m\x1b[38;2;70;70;70m  2-ring".to_string(),
        Piece::Three => "\x1b[1m\x1b[38;2;200;169;110m3\x1b[0m\x1b[38;2;70;70;70m  3-ring".to_string(),
    }
}