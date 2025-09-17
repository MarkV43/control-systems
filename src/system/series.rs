use nalgebra::{Const, OVector, Storage, Vector};

use crate::system::System;

/// Describes a couple of systems where the first's output is the second's input.
///
///           +---------+              +----------+
///           |         |              |          |
///  INPUT ---+  first  +--- MIDDLE ---+  second  +--- OUTPUT
///           |         |              |          |
///           +---------+              +----------+
///
pub struct SeriesSystem<
    const INPUTS: usize,
    const MIDDLE: usize,
    const OUTPUTS: usize,
    First,
    Second,
> where
    First: System<INPUTS, MIDDLE>,
    Second: System<MIDDLE, OUTPUTS>,
{
    first: First,
    second: Second,
}

impl<const INPUTS: usize, const MIDDLE: usize, const OUTPUTS: usize, First, Second>
    SeriesSystem<INPUTS, MIDDLE, OUTPUTS, First, Second>
where
    First: System<INPUTS, MIDDLE>,
    Second: System<MIDDLE, OUTPUTS>,
{
    pub fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }
}

impl<const INPUTS: usize, const MIDDLE: usize, const OUTPUTS: usize, First, Second>
    System<INPUTS, OUTPUTS> for SeriesSystem<INPUTS, MIDDLE, OUTPUTS, First, Second>
where
    First: System<INPUTS, MIDDLE>,
    Second: System<MIDDLE, OUTPUTS>,
{
    fn update<S>(&mut self, time: f64, input: &Vector<f64, Const<INPUTS>, S>) -> f64
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let next1 = self.first.update(time, input);
        let next2 = self.second.update(time, self.first.get_output());

        next1.min(next2)
    }

    fn get_output(&self) -> &OVector<f64, Const<OUTPUTS>> {
        self.second.get_output()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::vector;

    use crate::{prelude::Gain, utils::Param};

    use super::*;

    #[test]
    fn test_cloop() {
        let first = Gain::<1>::new(0.5);
        let second = Gain::new(4.0);

        let mut cloop = SeriesSystem::new(first, second);
        let input = Param::new(vector![1.0]);
        let mut out = vec![];

        cloop.simulate(1.0, 0.2, input, |x| out.push(x.output[0]));

        assert_eq!(out, &[2.0, 2.0, 2.0, 2.0, 2.0])
    }
}
