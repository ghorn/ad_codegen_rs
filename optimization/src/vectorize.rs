use sx_core::{SX, SXMatrix, SxError};

pub trait ScalarLeaf: Clone {}

impl ScalarLeaf for SX {}
impl ScalarLeaf for f64 {}

pub trait Vectorize<T: ScalarLeaf>: Sized {
    type Rebind<U: ScalarLeaf>;

    const LEN: usize;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>);

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U>;

    fn flatten_cloned(&self) -> Vec<T> {
        let mut refs = Vec::with_capacity(Self::LEN);
        self.flatten_refs(&mut refs);
        refs.into_iter().cloned().collect()
    }
}

impl<T: ScalarLeaf> Vectorize<T> for T {
    type Rebind<U: ScalarLeaf> = U;

    const LEN: usize = 1;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        out.push(self);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        f()
    }
}

impl<T: ScalarLeaf> Vectorize<T> for () {
    type Rebind<U: ScalarLeaf> = ();

    const LEN: usize = 0;

    fn flatten_refs<'a>(&'a self, _out: &mut Vec<&'a T>) {}

    fn from_flat_fn<U: ScalarLeaf>(_f: &mut impl FnMut() -> U) -> Self::Rebind<U> {}
}

impl<T, V, const N: usize> Vectorize<T> for [V; N]
where
    T: ScalarLeaf,
    V: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = [V::Rebind<U>; N];

    const LEN: usize = N * V::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        for value in self {
            value.flatten_refs(out);
        }
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        std::array::from_fn(|_| V::from_flat_fn::<U>(f))
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
