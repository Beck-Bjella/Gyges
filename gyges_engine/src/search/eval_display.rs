// eval_display.rs

use gyges::core::SQ;
use crate::search::{EvaluationContext, PieceData, Piece};

// ── Cell display config ───────────────────────────────────────────────────────
// Adjust these two constants to change cell size.
// CW must be large enough to fit your largest value + sign + spaces.
// A value like -0.123 needs at minimum: 1 (sign) + 1 (digit) + 1 (.) + 3 (decimals) = 6
// Plus 2 spaces padding = CW 8. Increase CW for larger numbers.
const CW: usize = 10;        // total cell width in chars
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
    /// Material heatmap: colored background per piece type, piece number + net value per cell.
    pub fn print_material_heatmap(&self) {
        const SIZE: usize = 6;
        const MCW: usize = 12; // cell: space + letter + space + +xxx.xxx + space = 12
        const INDENT: usize = 4;

        // Build a grid indexed by square: (piece, net_value, color)
        let mut grid: Vec<Option<(Piece, f64, (u8, u8, u8))>> = vec![None; SIZE * SIZE];
        for pd in self.piece_data.iter() {
            let sq = pd.sq.0 as usize;
            let v = (pd.material_score[0] - pd.material_score[1]) * 0.001;
            let bit = 1u64 << sq;
            let p1 = (self.unique_piece_control[0].0 & bit) != 0;
            let p2 = (self.unique_piece_control[1].0 & bit) != 0;
            let sh = (self.shared_piece_control.0    & bit) != 0;
            let c = match (p1, p2, sh) {
                (_, _, true) => (160u8, 100u8, 200u8), // shared — purple
                (true, _, _) => ( 92u8, 180u8, 130u8), // P1 unique — green
                (_, true, _) => (200u8,  90u8,  90u8), // P2 unique — red
                _            => ( 40u8,  40u8,  40u8), // uncontrolled
            };
            grid[sq] = Some((pd.piece, v, c));
        }

        let border = |left: &str, mid: &str, right: &str| {
            let pad = " ".repeat(INDENT);
            let inner: String = (0..SIZE)
                .map(|i| format!("{}{}", "─".repeat(MCW), if i < SIZE - 1 { mid } else { right }))
                .collect();
            println!("\x1b[38;2;50;50;50m{}{}{}\x1b[0m", pad, left, inner);
        };

        println!();
        println!("{}\x1b[1m\x1b[38;2;210;210;210mMaterial Score  (P1 + / P2 −)\x1b[0m", " ".repeat(INDENT));
        println!();

        border("┌", "┬", "┐");
        for row in (0..SIZE).rev() {
            // 4 passes: top padding, piece letter, value, bottom padding
            for pass in 0..4usize {
                print!("{}\x1b[38;2;50;50;50m│\x1b[0m", " ".repeat(INDENT));
                for col in 0..SIZE {
                    let sq = row * SIZE + col;
                    match grid[sq] {
                        Some((piece, v, (r, g, b))) => {
                            if pass == 1 {
                                let piece_str = match piece { Piece::One => "1", Piece::Two => "2", Piece::Three => "3", Piece::None => "?" };
                                print!("\x1b[48;2;{r};{g};{b}m\x1b[38;2;220;220;220m{:^MCW$}\x1b[0m", piece_str);
                            } else if pass == 2 {
                                let num = format!("{:+.3}", v);
                                print!("\x1b[48;2;{r};{g};{b}m\x1b[38;2;220;220;220m{:^MCW$}\x1b[0m", num);
                            } else {
                                print!("\x1b[48;2;{r};{g};{b}m{}\x1b[0m", " ".repeat(MCW));
                            }
                        }
                        None => {
                            print!("\x1b[48;2;22;22;22m{}\x1b[0m", " ".repeat(MCW));
                        }
                    }
                    print!("\x1b[38;2;50;50;50m│\x1b[0m");
                }
                println!();
            }
            if row > 0 { border("├", "┼", "┤"); }
        }
        border("└", "┴", "┘");
        println!();
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

// ── Material Breakdown  ──────────────────────────────────────────────────

impl EvaluationContext {
    pub fn print_material_breakdown(&self) {
        const INDENT: &str = "    ";

        println!();
        println!("{}\x1b[1m\x1b[38;2;210;210;210mMaterial Breakdown\x1b[0m", INDENT);
        println!();

        println!(
            "{}\x1b[38;2;100;100;100m{:>4}  {:4}  {:8}  {:>6}  {:>7}  {:>7}  {:>7}  {}\x1b[0m",
            INDENT, "sq", "type", "control", "own", "P1", "P2", "net", "flags"
        );
        println!("{}\x1b[38;2;50;50;50m{}\x1b[0m", INDENT, "─".repeat(62));

        let mut pieces: Vec<&PieceData> = self.piece_data.iter().collect();
        pieces.sort_by_key(|pd| pd.sq.0);

        for pd in pieces {
            let sq = pd.sq.0 as usize;
            let piece_str = match pd.piece { Piece::One => "1", Piece::Two => "2", Piece::Three => "3", Piece::None => "?" };

            let p1_unique = (self.unique_piece_control[0].0 & pd.sq.bit()) != 0;
            let p2_unique = (self.unique_piece_control[1].0 & pd.sq.bit()) != 0;
            let shared    = (self.shared_piece_control.0    & pd.sq.bit()) != 0;

            let (ctrl_str, ctrl_r, ctrl_g, ctrl_b) = match (p1_unique, p2_unique, shared) {
                (true, _, _) => ("P1 uniq",  92u8, 180u8, 130u8),
                (_, true, _) => ("P2 uniq", 200u8,  90u8,  90u8),
                (_, _, true) => ("shared ", 160u8, 100u8, 200u8),
                _            => ("none   ",  60u8,  60u8,  60u8),
            };

            let own = {
                let a1 = if pd.path_min_depths[0].is_finite() { 1.0 / pd.path_min_depths[0].max(1.0) } else { 0.0 };
                let a2 = if pd.path_min_depths[1].is_finite() { 1.0 / pd.path_min_depths[1].max(1.0) } else { 0.0 };
                let total = a1 + a2;
                if total == 0.0 { 0.0 } else { (a1 - a2) / total }
            };
            let (own_r, own_g, own_b) = if own > 0.05 { (92u8, 180u8, 130u8) }
                else if own < -0.05 { (200u8, 90u8, 90u8) }
                else { (140u8, 140u8, 140u8) };

            let p1_s = pd.material_score[0] * 0.001;
            let p2_s = pd.material_score[1] * 0.001;
            let net  = p1_s - p2_s;

            let mut flags = String::new();
            if pd.trapped[0]  { flags.push_str("T1 "); }
            if pd.trapped[1]  { flags.push_str("T2 "); }
            if pd.stranded[0] { flags.push_str("S1 "); }
            if pd.stranded[1] { flags.push_str("S2 "); }

            let (net_r, net_g, net_b) = if net > 0.05 { (92u8, 180u8, 130u8) }
                else if net < -0.05 { (200u8, 90u8, 90u8) }
                else { (140u8, 140u8, 140u8) };

            println!(
                "{}\x1b[38;2;140;140;140m{:>4}  \x1b[38;2;200;169;110m{:4}  \x1b[38;2;{};{};{}m{:8}\x1b[38;2;{};{};{}m  {:>+6.2}  \x1b[38;2;140;140;140m{:>7.3}  {:>7.3}  \x1b[38;2;{};{};{}m{:>+7.3}\x1b[38;2;80;80;80m  {}\x1b[0m",
                INDENT, sq, piece_str,
                ctrl_r, ctrl_g, ctrl_b, ctrl_str,
                own_r, own_g, own_b, own,
                p1_s, p2_s,
                net_r, net_g, net_b, net,
                flags
            );
        }

        println!("{}\x1b[38;2;50;50;50m{}\x1b[0m", INDENT, "─".repeat(62));
        println!();
        self.print_material_heatmap();
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
