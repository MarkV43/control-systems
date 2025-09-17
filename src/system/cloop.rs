use nalgebra::{Const, Storage, Vector};

use crate::system::System;

pub struct ClosedLoop<const INPUTS: usize, const OUTPUTS: usize, SysFw, SysFb>
where
    SysFw: System<INPUTS, OUTPUTS>,
    SysFb: System<OUTPUTS, INPUTS>,
{
    forward: SysFw,
    feedback: SysFb,
}

impl<const INPUTS: usize, const OUTPUTS: usize, SysFw, SysFb>
    ClosedLoop<INPUTS, OUTPUTS, SysFw, SysFb>
where
    SysFw: System<INPUTS, OUTPUTS>,
    SysFb: System<OUTPUTS, INPUTS>,
{
    pub fn new(forward: SysFw, feedback: SysFb) -> Self {
        Self { forward, feedback }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, SysFw, SysFb> System<INPUTS, OUTPUTS>
    for ClosedLoop<INPUTS, OUTPUTS, SysFw, SysFb>
where
    SysFw: System<INPUTS, OUTPUTS>,
    SysFb: System<OUTPUTS, INPUTS>,
{
    fn update<S>(&mut self, time: f64, input: &Vector<f64, Const<INPUTS>, S>) -> f64
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let error = input - self.feedback.get_output();

        let next1 = self.forward.update(time, &error);
        let next2 = self.feedback.update(time, self.forward.get_output());

        next1.min(next2)
    }

    fn get_output(&self) -> &nalgebra::OVector<f64, nalgebra::Const<OUTPUTS>> {
        self.forward.get_output()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::vector;

    use crate::{prelude::Gain, system::UnitSystem, utils::Param};

    use super::*;

    #[test]
    fn test_cloop() {
        let fw = Gain::<1>::new(0.5);
        let fb = UnitSystem::default();

        let mut cloop = ClosedLoop::new(fw, fb);
        let input = Param::new(vector![1.0]);
        let mut out = vec![];

        cloop.simulate(1.0, 0.2, input, |x| out.push(x.output[0]));

        assert_eq!(out, &[0.5, 0.25, 0.375, 0.3125, 0.34375])
    }
}
