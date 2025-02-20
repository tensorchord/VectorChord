pub trait Pipe {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T
    where
        Self: Sized;
}

impl<S: ?Sized> Pipe for S {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T
    where
        Self: Sized,
    {
        f(self)
    }
}
