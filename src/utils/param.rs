use std::ops::{Deref, DerefMut};

use num::traits::Inv;

#[derive(Clone)]
pub struct Param<V> {
    pub value: V,
    steps: Vec<(f64, V)>,
    next_step: usize,
}

#[derive(Clone)]
pub struct ParamWith<V, F>
where
    F: FnMut(V) -> V,
{
    param: Param<V>,
    callback: F,
    pub with: V,
}

impl<V> Param<V> {
    pub fn new(value: impl Into<V>) -> Self {
        Self {
            value: value.into(),
            steps: Vec::new(),
            next_step: 0,
        }
    }

    #[inline]
    pub fn step(mut self, new_value: impl Into<V>, instant: f64) -> Self {
        let pos = self
            .steps
            .iter()
            .position(|(t, _)| {
                matches!(t.partial_cmp(&instant), Some(core::cmp::Ordering::Greater))
            })
            .unwrap_or(self.steps.len());
        self.steps.insert(pos, (instant, new_value.into()));
        self
    }

    // TODO use this
    #[allow(dead_code)]
    pub fn delay(mut self, delay_amount: f64) -> Self {
        self.steps.iter_mut().for_each(|(t, _)| *t += delay_amount);
        self
    }

    pub fn update(&mut self, current_time: f64) -> bool
    where
        V: Clone,
    {
        if let Some((time, value)) = self.steps.get(self.next_step)
            && current_time >= *time
        {
            self.next_step += 1;
            self.value = value.clone();
            true
        } else {
            false
        }
    }

    pub fn with<F>(self, mut callback: F) -> ParamWith<V, F>
    where
        V: Clone,
        F: FnMut(V) -> V,
    {
        let with = (callback)((*self).clone());
        ParamWith {
            with,
            callback,
            param: self,
        }
    }
}

impl<V, F> ParamWith<V, F>
where
    V: Inv<Output = V>,
    F: FnMut(V) -> V,
{
    #[inline]
    pub fn update(&mut self, current_time: f64) -> bool
    where
        V: Clone,
    {
        if !self.param.update(current_time) {
            return false;
        }

        let val = (*self.param).clone();

        self.with = (self.callback)(val);
        true
    }
}

impl<V> From<V> for Param<V> {
    fn from(value: V) -> Self {
        Self {
            value,
            steps: Vec::new(),
            next_step: 0,
        }
    }
}

impl<V> Deref for Param<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<V> DerefMut for Param<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<V, F> Deref for ParamWith<V, F>
where
    V: Inv,
    F: FnMut(V) -> V,
{
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.param
    }
}

impl<V, F> DerefMut for ParamWith<V, F>
where
    V: Inv,
    F: FnMut(V) -> V,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.param
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param() {
        let mut param = Param::<f64>::new(0.0).step(1.0, 5.0);

        assert_eq!(*param, 0.0);
        param.update(1.0);
        assert_eq!(*param, 0.0);
        param.update(2.0);
        assert_eq!(*param, 0.0);
        param.update(3.0);
        assert_eq!(*param, 0.0);
        param.update(4.0);
        assert_eq!(*param, 0.0);
        param.update(5.0);
        assert_eq!(*param, 1.0);
        param.update(6.0);
        assert_eq!(*param, 1.0);
    }

    #[test]
    fn test_param_with() {
        let mut param = Param::<f64>::new(2.0).step(1.0, 5.0).with(|x| 1. / x);

        assert_eq!(*param, 2.0);
        assert_eq!(param.with, 0.5);
        param.update(1.0);
        assert_eq!(*param, 2.0);
        assert_eq!(param.with, 0.5);
        param.update(2.0);
        assert_eq!(*param, 2.0);
        assert_eq!(param.with, 0.5);
        param.update(3.0);
        assert_eq!(*param, 2.0);
        assert_eq!(param.with, 0.5);
        param.update(4.0);
        assert_eq!(*param, 2.0);
        assert_eq!(param.with, 0.5);
        param.update(5.0);
        assert_eq!(*param, 1.0);
        assert_eq!(param.with, 1.0);
        param.update(6.0);
        assert_eq!(*param, 1.0);
        assert_eq!(param.with, 1.0);
    }
}
