use std::{
    cell::UnsafeCell,
    ops::{Add, Mul},
};

pub trait Holder<Data> {
    /// Update the holder with a new input sample at the given time.
    fn hold(&mut self, time: f64, input: &Data);

    /// Return the approximated input value at the given query time.
    fn get_output(&self, time: f64) -> Data;
}

/// Zero-Order Hold (ZOH)
pub struct ZeroOrderHold<Data> {
    last_time: f64,
    last_input: Data,
}

impl<Data> ZeroOrderHold<Data> {
    pub fn new() -> Self
    where
        Data: Default,
    {
        Self {
            last_time: 0.0,
            last_input: Data::default(),
        }
    }
}

impl<Data> Holder<Data> for ZeroOrderHold<Data>
where
    Data: Clone,
{
    fn hold(&mut self, time: f64, input: &Data) {
        self.last_time = time;
        self.last_input = input.clone();
    }

    fn get_output(&self, _query_time: f64) -> Data {
        self.last_input.clone()
    }
}

/// First-Order Hold (FOH)
pub struct FirstOrderHold<Data> {
    last_time: f64,
    last_input: Data,
    curr_time: f64,
    curr_input: Data,
    initialized: bool,
    /// scratch buffer used to store interpolated result returned by reference.
    output: UnsafeCell<Data>,
}

impl<Data> FirstOrderHold<Data> {
    pub fn new() -> Self
    where
        Data: Default,
    {
        Self {
            last_time: 0.0,
            last_input: Data::default(),
            curr_time: 0.0,
            curr_input: Data::default(),
            initialized: false,
            output: UnsafeCell::new(Data::default()),
        }
    }
}

impl<Data> Holder<Data> for FirstOrderHold<Data>
where
    Data: Clone + Mul<f64, Output = Data> + Add<Data, Output = Data>,
{
    fn hold(&mut self, time: f64, input: &Data) {
        if !self.initialized {
            self.curr_time = time;
            self.curr_input = input.clone();
            self.initialized = true;
        } else {
            self.last_time = self.curr_time;
            self.last_input = self.curr_input.clone();
            self.curr_time = time;
            self.curr_input = input.clone();
        }
    }

    fn get_output(&self, query_time: f64) -> Data {
        // not initialized -> return current input (which is zero-initialized until hold is called)
        if !self.initialized {
            return self.curr_input.clone();
        }

        // degenerate interval -> return current input
        if (self.curr_time - self.last_time).abs() < f64::EPSILON {
            return self.curr_input.clone();
        }

        // outside interval -> clamp to endpoints
        if query_time <= self.last_time {
            return self.last_input.clone();
        }
        if query_time >= self.curr_time {
            return self.curr_input.clone();
        }

        // interpolate into scratch buffer and return reference to it.
        let tau = (query_time - self.last_time) / (self.curr_time - self.last_time);

        // SAFETY: UnsafeCell gives interior mutability for single-threaded use.
        // We write the interpolated value into the cell and then return a reference to it
        // tied to &self. This is safe if caller does not alias mutable references concurrently.
        unsafe {
            let out = &mut *self.output.get();
            // compute via owned temporaries then copy into out to avoid per-element loops
            *out = self.last_input.clone() * (1.0 - tau) + self.curr_input.clone() * tau;
            (*self.output.get()).clone()
        }
    }
}

/// Impulse Hold: returns the input only at the update instant, zero otherwise.
pub struct ImpulseHold<Data> {
    last_time: f64,
    last_input: Data,
    zero: Data,
}

impl<Data> ImpulseHold<Data> {
    pub fn new() -> Self
    where
        Data: Default,
    {
        Self {
            last_time: 0.0,
            last_input: Data::default(),
            zero: Data::default(),
        }
    }
}

impl<Data> Holder<Data> for ImpulseHold<Data>
where
    Data: Clone,
{
    fn hold(&mut self, time: f64, input: &Data) {
        self.last_time = time;
        self.last_input = input.clone();
    }

    fn get_output(&self, query_time: f64) -> Data {
        if (query_time - self.last_time).abs() < f64::EPSILON {
            &self.last_input
        } else {
            &self.zero
        }
        .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, Owned, Vector};

    pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;

    #[test]
    fn zoh_keeps_last_input() {
        const N: usize = 3;
        let mut zoh = ZeroOrderHold::<VecN<N>>::new();

        // before any hold call -> zeros
        assert_eq!(zoh.get_output(0.0), VecN::<N>::zeros());

        let sample = VecN::<N>::from_row_slice(&[1.0, -2.0, 3.5]);
        zoh.hold(1.0, &sample);

        // at and after sample time -> same sample
        assert_eq!(zoh.get_output(1.0), sample);
        assert_eq!(zoh.get_output(1.5), sample);
        assert_eq!(zoh.get_output(0.9), sample); // ZOH returns last_input regardless of query_time
    }

    #[test]
    fn foh_interpolates_linearly_and_clamps() {
        const N: usize = 2;
        let mut foh = FirstOrderHold::<VecN<N>>::new();

        // initially uninitialized -> returns zero curr_input
        assert_eq!(foh.get_output(0.0), VecN::<N>::zeros());

        // first sample (initializes)
        let s0 = VecN::<N>::from_row_slice(&[0.0, 0.0]);
        foh.hold(0.0, &s0);

        // second sample -> forms an interval [0.0, 1.0] with values s0 and s1
        let s1 = VecN::<N>::from_row_slice(&[2.0, 4.0]);
        foh.hold(1.0, &s1);

        // midpoint should be average -> [1.0, 2.0]
        let mid_expected = VecN::<N>::from_row_slice(&[1.0, 2.0]);
        assert_eq!(foh.get_output(0.5), mid_expected);

        // clamp to endpoints
        assert_eq!(foh.get_output(0.0), s0);
        assert_eq!(foh.get_output(1.0), s1);
        assert_eq!(foh.get_output(-1.0), s0);
        assert_eq!(foh.get_output(2.0), s1);
    }

    #[test]
    fn impulse_hold_only_at_instant() {
        const N: usize = 2;
        let mut ih = ImpulseHold::<VecN<N>>::new();

        let sample = VecN::<N>::from_row_slice(&[3.0, -3.0]);
        ih.hold(2.0, &sample);

        // exactly at sample time -> sample
        assert_eq!(ih.get_output(2.0), sample);

        // slightly off -> zero
        let eps = 1e-12;
        assert_eq!(ih.get_output(2.0 + eps), VecN::<N>::zeros());
        assert_eq!(ih.get_output(1.999999999999), VecN::<N>::zeros());
    }
}
