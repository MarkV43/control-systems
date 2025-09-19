use std::{marker::PhantomData, ops::Sub};

use crate::{system::System};

pub struct ClosedLoop<Input, Output, SysFw, SysFb>
where
    SysFw: System<Input = Input, Output = Output>,
    SysFb: System<Input = Output, Output = Input>,
{
    forward: SysFw,
    feedback: SysFb,
    _dummy: PhantomData<(Input, Output)>,
}

impl<Input, Output, SysFw, SysFb> ClosedLoop<Input, Output, SysFw, SysFb>
where
    SysFw: System<Input = Input, Output = Output>,
    SysFb: System<Input = Output, Output = Input>,
{
    pub fn new(forward: SysFw, feedback: SysFb) -> Self {
        Self {
            forward,
            feedback,
            _dummy: PhantomData,
        }
    }
}

impl<Input, Output, SysFw, SysFb> System for ClosedLoop<Input, Output, SysFw, SysFb>
where
    SysFw: System<Input = Input, Output = Output>,
    SysFb: System<Input = Output, Output = Input>,
    for<'a> &'a Input: Sub<Input, Output = Input>,
{
    type Input = Input;
    type Output = Output;

    fn update(&mut self, time: f64, input: &Input) -> f64 {
        let error = input - self.feedback.get_output(time);

        let next1 = self.forward.update(time, &error);
        let next2 = self.feedback.update(time, &self.forward.get_output(time));

        next1.min(next2)
    }

    fn get_output(&self, time: f64) -> Output {
        self.forward.get_output(time)
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Gain, system::UnitSystem, utils::Param};
    use super::*;
    use nalgebra::{vector, Const, Owned, Vector};
    
    pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;

    #[test]
    fn test_cloop() {
        let fw = Gain::<VecN<1>>::new(0.5);
        let fb = UnitSystem::default();

        let mut cloop = ClosedLoop::new(fw, fb);
        let input = Param::new(vector![1.0]);
        let mut out = vec![];

        cloop.simulate(1.0, 0.2, input, &mut |x| out.push(x.output[0]));

        assert_eq!(out, &[0.5, 0.25, 0.375, 0.3125, 0.34375])
    }
}
