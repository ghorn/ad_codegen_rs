use sx_core::{SX, SXMatrix, SxError};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VectorizeLayoutError {
    #[error("flat layout length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },
}

pub trait ScalarLeaf: Clone {}

impl ScalarLeaf for SX {}
impl ScalarLeaf for f64 {}

pub trait Vectorize<T: ScalarLeaf>: Sized {
    type Rebind<U: ScalarLeaf>;
    type View<'a>
    where
        Self: 'a,
        T: 'a;

    const LEN: usize;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>);

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U>;

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        Self: 'a,
        T: 'a;

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a;

    fn flatten_cloned(&self) -> Vec<T> {
        let mut refs = Vec::with_capacity(Self::LEN);
        self.flatten_refs(&mut refs);
        refs.into_iter().cloned().collect()
    }

    fn from_flat_slice(values: &[T]) -> Result<Self::Rebind<T>, VectorizeLayoutError>
    where
        T: Clone,
    {
        if values.len() != Self::LEN {
            return Err(VectorizeLayoutError::LengthMismatch {
                expected: Self::LEN,
                got: values.len(),
            });
        }
        let mut index = 0usize;
        Ok(Self::from_flat_fn(&mut || {
            let value = values[index].clone();
            index += 1;
            value
        }))
    }
}

impl<T: ScalarLeaf> Vectorize<T> for T {
    type Rebind<U: ScalarLeaf> = U;
    type View<'a>
        = &'a T
    where
        T: 'a;

    const LEN: usize = 1;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        out.push(self);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        f()
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
        self
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
        let value = &slice[*index];
        *index += 1;
        value
    }
}

impl<T: ScalarLeaf> Vectorize<T> for () {
    type Rebind<U: ScalarLeaf> = ();
    type View<'a>
        = ()
    where
        T: 'a;

    const LEN: usize = 0;

    fn flatten_refs<'a>(&'a self, _out: &mut Vec<&'a T>) {}

    fn from_flat_fn<U: ScalarLeaf>(_f: &mut impl FnMut() -> U) -> Self::Rebind<U> {}

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
    }

    fn view_from_flat_slice<'a>(_slice: &'a [T], _index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
    }
}

impl<T, V, const N: usize> Vectorize<T> for [V; N]
where
    T: ScalarLeaf,
    V: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = [V::Rebind<U>; N];
    type View<'a>
        = [V::View<'a>; N]
    where
        T: 'a,
        V: 'a;

    const LEN: usize = N * V::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        for value in self {
            value.flatten_refs(out);
        }
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        std::array::from_fn(|_| V::from_flat_fn::<U>(f))
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        V: 'a,
    {
        std::array::from_fn(|idx| self[idx].view())
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        V: 'a,
    {
        std::array::from_fn(|_| V::view_from_flat_slice(slice, index))
    }
}

pub fn symbolic_value<T>(prefix: &str) -> Result<T, SxError>
where
    T: Vectorize<SX, Rebind<SX> = T>,
{
    let mut index = 0usize;
    Ok(T::from_flat_fn(&mut || {
        let name = if T::LEN == 1 {
            prefix.to_string()
        } else {
            let current = index;
            index += 1;
            format!("{prefix}_{current}")
        };
        SX::sym(name)
    }))
}

pub fn symbolic_column<T>(value: &T) -> Result<SXMatrix, SxError>
where
    T: Vectorize<SX>,
{
    SXMatrix::dense_column(value.flatten_cloned())
}

pub fn flatten_value<T>(value: &T) -> Vec<f64>
where
    T: Vectorize<f64>,
{
    value.flatten_cloned()
}

pub fn unflatten_value<S, T>(values: &[T]) -> Result<S::Rebind<T>, VectorizeLayoutError>
where
    S: Vectorize<T>,
    T: ScalarLeaf + Clone,
{
    S::from_flat_slice(values)
}

pub fn flat_view<'a, S, T>(values: &'a [T]) -> Result<S::View<'a>, VectorizeLayoutError>
where
    S: Vectorize<T>,
    T: ScalarLeaf,
{
    if values.len() != S::LEN {
        return Err(VectorizeLayoutError::LengthMismatch {
            expected: S::LEN,
            got: values.len(),
        });
    }
    let mut index = 0usize;
    Ok(S::view_from_flat_slice(values, &mut index))
}
