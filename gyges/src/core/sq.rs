use std::{ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign}, fmt::Display};


#[derive(PartialEq, Debug, Clone, Copy)]
pub struct SQ(pub u8);

impl SQ {
    pub const NONE: SQ = SQ(100);

    pub const P1_GOAL: SQ = SQ(36);
    pub const P2_GOAL: SQ = SQ(37);

    pub const GOALS: [SQ; 2] = [SQ(36), SQ(37)];

    pub fn bit(&self) -> u64 {
        1 << self.0

    }

    pub fn in_bounds(&self) -> bool {
        self.0 < 38

    }

    fn on_top_edge(&self) -> bool {
        self.0 + 6 > 35 

    }

    fn on_bottom_edge(&self) -> bool {
        self.0 - 6 > 35 

    }

    fn on_right_edge(&self) -> bool {
        self.0 == 5 || self.0 == 11 || self.0 == 17 || self.0 == 23 || self.0 == 29 || self.0 == 35

    }

    fn on_left_edge(&self) -> bool {
        self.0 == 0 || self.0 == 6 || self.0 == 12 || self.0 == 18 || self.0 == 24 || self.0 == 30

    }

}

impl Display for SQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)

    }
    

}

// impl math operations
impl Add<usize> for SQ {
    type Output = SQ;

    fn add(self, rhs: usize) -> SQ {
        SQ(self.0 + rhs as u8)

    }

}

impl Add<SQ> for SQ {
    type Output = SQ;

    fn add(self, rhs: SQ) -> SQ {
        SQ(self.0 + rhs.0)

    }

}

impl AddAssign<usize> for SQ {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs as u8

    }

}

impl AddAssign<SQ> for SQ {
    fn add_assign(&mut self, rhs: SQ) {
        self.0 += rhs.0

    }

}

impl Sub<usize> for SQ {
    type Output = SQ;

    fn sub(self, rhs: usize) -> SQ {
        SQ(self.0 - rhs as u8)

    }

}

impl Sub<SQ> for SQ {
    type Output = SQ;

    fn sub(self, rhs: SQ) -> SQ {
        SQ(self.0 - rhs.0)

    }

}

impl SubAssign<usize> for SQ {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs as u8

    }

}

impl SubAssign<SQ> for SQ {
    fn sub_assign(&mut self, rhs: SQ) {
        self.0 -= rhs.0

    }

}

impl Mul<usize> for SQ {
    type Output = SQ;

    fn mul(self, rhs: usize) -> SQ {
        SQ(self.0 * rhs as u8)

    }

}

impl Mul<SQ> for SQ {
    type Output = SQ;

    fn mul(self, rhs: SQ) -> SQ {
        SQ(self.0 * rhs.0)

    }

}

impl MulAssign<usize> for SQ {
    fn mul_assign(&mut self, rhs: usize) {
        self.0 *= rhs as u8

    }

}

impl MulAssign<SQ> for SQ {
    fn mul_assign(&mut self, rhs: SQ) {
        self.0 *= rhs.0

    }

}

