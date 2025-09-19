use std::f64;

pub mod cloop;
pub mod gain;
pub mod series;

use crate::utils::Param;

/// A model for any kind of system with `INPUTS` inputs and `OUTPUTS` outputs.
///
/// Instead of the normal system implementations, where the overall simulation
/// has a single timestep for all its components, making it hard for fast system
/// to work in tandem with slower system, this implementation uses an event
/// request system, where each system requests an instant for its next update.
///
/// This way, slower system will be updated less often, while faster systems can
/// have different frequencies, and still work without issue.
pub trait System {
    type Input;
    type Output;

    /// Updates the system as if it were on instant `time` receiving inputs `input`.
    /// Returns the next instant the system will be updated at.
    fn update(&mut self, time: f64, input: &Self::Input) -> f64;

    /// Returns the system's current output. Should be called after `update`ing the system.
    fn get_output(&self, time: f64) -> Self::Output;

    /// Simulates the system for a full `total_time` time units.
    fn simulate<F>(
        &mut self,
        total_time: f64,
        max_timestep: f64,
        mut input: Param<Self::Input>,
        mut callback: F,
    ) where
        F: FnMut(Sample<Self::Input, Self::Output>) -> (),
        Self::Input: Clone,
        Self::Output: Clone,
    {
        let mut time = 0.0;

        while time < total_time {
            input.update(time);
            let next_time = self.update(time, &*input);

            let sample = Sample {
                instant: time,
                input: (*input).clone(),
                output: self.get_output(time).clone(),
            };
            callback(sample);

            time = next_time.min(time + max_timestep);
        }
    }
}

pub struct Sample<Input, Output> {
    pub instant: f64,
    pub input: Input,
    pub output: Output,
}

/// A simple system that directly transfer the input to the output.
/// Its transfer function is represented by $F(s) = 1$.
pub struct UnitSystem<Data> {
    output: Data,
}

impl<Data> System for UnitSystem<Data>
where
    Data: Clone,
{
    type Input = Data;
    type Output = Data;
    
    fn update(&mut self, _: f64, input: &Data) -> f64 {
        self.output = input.clone();
        f64::INFINITY
    }

    fn get_output(&self, _time: f64) -> Data {
        self.output.clone()
    }
}

impl<Data> Default for UnitSystem<Data>
where
    Data: Default,
{
    fn default() -> Self {
        Self {
            output: Data::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Param;
    use nalgebra::{Const, Owned, Vector};

    pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;

    #[test]
    fn test_max_timestep() {
        let mut sys = UnitSystem::<VecN<1>>::default();

        let input = Param::new(VecN::<1>::from_column_slice(&[3.]));

        let mut count = 0;
        sys.simulate(0.4, 0.1, input, |_| count += 1);

        assert_eq!(count, 4);
    }
}
