use std::marker::PhantomData;

use crate::system::System;

/// Describes a couple of systems where the first's output is the second's input.
///
///           +---------+              +----------+
///           |         |              |          |
///  INPUT ---+  first  +--- MIDDLE ---+  second  +--- OUTPUT
///           |         |              |          |
///           +---------+              +----------+
///
pub struct SeriesSystem<Input, Middle, Output, First, Second>
where
    First: System<Input, Middle>,
    Second: System<Middle, Output>,
{
    first: First,
    second: Second,
    _dummy: PhantomData<(Input, Middle, Output)>,
}

impl<Input, Middle, Output, First, Second> SeriesSystem<Input, Middle, Output, First, Second>
where
    First: System<Input, Middle>,
    Second: System<Middle, Output>,
{
    pub fn new(first: First, second: Second) -> Self {
        Self {
            first,
            second,
            _dummy: PhantomData,
        }
    }
}

impl<Input, Middle, Output, First, Second> System<Input, Output>
    for SeriesSystem<Input, Middle, Output, First, Second>
where
    First: System<Input, Middle>,
    Second: System<Middle, Output>,
{
    fn update(&mut self, time: f64, input: &Input) -> f64 {
        let next1 = self.first.update(time, input);
        let next2 = self.second.update(time, &self.first.get_output(time));

        next1.min(next2)
    }

    fn get_output(&self, time: f64) -> Output {
        self.second.get_output(time)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{vector, Const, Owned, Vector};

    use crate::{prelude::Gain, utils::Param};

    pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;

    use super::*;

    #[test]
    fn test_cloop() {
        let first = Gain::<VecN<1>>::new(0.5);
        let second = Gain::new(4.0);

        let mut cloop = SeriesSystem::new(first, second);
        let input = Param::new(vector![1.0]);
        let mut out = vec![];

        cloop.simulate(1.0, 0.2, input, |x| out.push(x.output[0]));

        assert_eq!(out, &[2.0, 2.0, 2.0, 2.0, 2.0])
    }
}
