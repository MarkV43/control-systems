mod param;

use nalgebra::{Const, Matrix, Owned, Vector};

pub use self::param::{Param, ParamWith};

pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;
pub type MatN<const R: usize, const C: usize, S = Owned<f64, Const<R>, Const<C>>> =
    Matrix<f64, Const<R>, Const<C>, S>;
