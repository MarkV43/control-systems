use nalgebra::{Const, Storage};
use std::cell::UnsafeCell;

use crate::utils::VecN;

pub trait Holder<const N: usize> {
    /// Update the holder with a new input sample at the given time.
    fn hold<S>(&mut self, time: f64, input: &VecN<N, S>)
    where
        S: Storage<f64, Const<N>>;

    /// Return the approximated input value at the given query time.
    fn get_output(&self, time: f64) -> &VecN<N>;
}

/// Zero-Order Hold (ZOH)
pub struct ZeroOrderHold<const N: usize> {
    last_time: f64,
    last_input: VecN<N>,
}

impl<const N: usize> ZeroOrderHold<N> {
    pub fn new() -> Self {
        Self {
            last_time: 0.0,
            last_input: VecN::<N>::zeros(),
        }
    }
}

impl<const N: usize> Holder<N> for ZeroOrderHold<N> {
    fn hold<S>(&mut self, time: f64, input: &VecN<N, S>)
    where
        S: Storage<f64, Const<N>>,
    {
        self.last_time = time;
        self.last_input.copy_from(input);
    }

    fn get_output(&self, _query_time: f64) -> &VecN<N> {
        &self.last_input
    }
}

/// First-Order Hold (FOH)
pub struct FirstOrderHold<const N: usize> {
    last_time: f64,
    last_input: VecN<N>,
    curr_time: f64,
    curr_input: VecN<N>,
    initialized: bool,
    /// scratch buffer used to store interpolated result returned by reference.
    output: UnsafeCell<VecN<N>>,
}

impl<const N: usize> FirstOrderHold<N> {
    pub fn new() -> Self {
        Self {
            last_time: 0.0,
            last_input: VecN::<N>::zeros(),
            curr_time: 0.0,
            curr_input: VecN::<N>::zeros(),
            initialized: false,
            output: UnsafeCell::new(VecN::<N>::zeros()),
        }
    }
}

impl<const N: usize> Holder<N> for FirstOrderHold<N> {
    fn hold<S>(&mut self, time: f64, input: &VecN<N, S>)
    where
        S: Storage<f64, Const<N>>,
    {
        if !self.initialized {
            self.curr_time = time;
            self.curr_input.copy_from(input);
            self.initialized = true;
        } else {
            self.last_time = self.curr_time;
            self.last_input = self.curr_input.clone_owned();
            self.curr_time = time;
            self.curr_input.copy_from(input);
        }
    }

    fn get_output(&self, query_time: f64) -> &VecN<N> {
        // not initialized -> return current input (which is zero-initialized until hold is called)
        if !self.initialized {
            return &self.curr_input;
        }

        // degenerate interval -> return current input
        if (self.curr_time - self.last_time).abs() < f64::EPSILON {
            return &self.curr_input;
        }

        // outside interval -> clamp to endpoints
        if query_time <= self.last_time {
            return &self.last_input;
        }
        if query_time >= self.curr_time {
            return &self.curr_input;
        }

        // interpolate into scratch buffer and return reference to it.
        let tau = (query_time - self.last_time) / (self.curr_time - self.last_time);

        // SAFETY: UnsafeCell gives interior mutability for single-threaded use.
        // We write the interpolated value into the cell and then return a reference to it
        // tied to &self. This is safe if caller does not alias mutable references concurrently.
        unsafe {
            let out = &mut *self.output.get();
            // compute via owned temporaries then copy into out to avoid per-element loops
            *out =
                self.last_input.clone_owned() * (1.0 - tau) + self.curr_input.clone_owned() * tau;
            &*self.output.get()
        }
    }
}

/// Impulse Hold: returns the input only at the update instant, zero otherwise.
pub struct ImpulseHold<const N: usize> {
    last_time: f64,
    last_input: VecN<N>,
    zero: VecN<N>,
}

impl<const N: usize> ImpulseHold<N> {
    pub fn new() -> Self {
        Self {
            last_time: 0.0,
            last_input: VecN::<N>::zeros(),
            zero: VecN::<N>::zeros(),
        }
    }
}

impl<const N: usize> Holder<N> for ImpulseHold<N> {
    fn hold<S>(&mut self, time: f64, input: &VecN<N, S>)
    where
        S: Storage<f64, Const<N>>,
    {
        self.last_time = time;
        self.last_input.copy_from(input);
    }

    fn get_output(&self, query_time: f64) -> &VecN<N> {
        if (query_time - self.last_time).abs() < f64::EPSILON {
            &self.last_input
        } else {
            &self.zero
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::VecN;

    #[test]
    fn zoh_keeps_last_input() {
        const N: usize = 3;
        let mut zoh = ZeroOrderHold::<N>::new();

        // before any hold call -> zeros
        assert_eq!(zoh.get_output(0.0), &VecN::<N>::zeros());

        let sample = VecN::<N>::from_row_slice(&[1.0, -2.0, 3.5]);
        zoh.hold(1.0, &sample);

        // at and after sample time -> same sample
        assert_eq!(zoh.get_output(1.0), &sample);
        assert_eq!(zoh.get_output(1.5), &sample);
        assert_eq!(zoh.get_output(0.9), &sample); // ZOH returns last_input regardless of query_time
    }

    #[test]
    fn foh_interpolates_linearly_and_clamps() {
        const N: usize = 2;
        let mut foh = FirstOrderHold::<N>::new();

        // initially uninitialized -> returns zero curr_input
        assert_eq!(foh.get_output(0.0), &VecN::<N>::zeros());

        // first sample (initializes)
        let s0 = VecN::<N>::from_row_slice(&[0.0, 0.0]);
        foh.hold(0.0, &s0);

        // second sample -> forms an interval [0.0, 1.0] with values s0 and s1
        let s1 = VecN::<N>::from_row_slice(&[2.0, 4.0]);
        foh.hold(1.0, &s1);

        // midpoint should be average -> [1.0, 2.0]
        let mid_expected = VecN::<N>::from_row_slice(&[1.0, 2.0]);
        assert_eq!(foh.get_output(0.5), &mid_expected);

        // clamp to endpoints
        assert_eq!(foh.get_output(0.0), &s0);
        assert_eq!(foh.get_output(1.0), &s1);
        assert_eq!(foh.get_output(-1.0), &s0);
        assert_eq!(foh.get_output(2.0), &s1);
    }

    #[test]
    fn impulse_hold_only_at_instant() {
        const N: usize = 2;
        let mut ih = ImpulseHold::<N>::new();

        let sample = VecN::<N>::from_row_slice(&[3.0, -3.0]);
        ih.hold(2.0, &sample);

        // exactly at sample time -> sample
        assert_eq!(ih.get_output(2.0), &sample);

        // slightly off -> zero
        let eps = 1e-12;
        assert_eq!(ih.get_output(2.0 + eps), &VecN::<N>::zeros());
        assert_eq!(ih.get_output(1.999999999999), &VecN::<N>::zeros());
    }
}
