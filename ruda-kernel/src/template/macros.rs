/// Generates kernel source code by replacing some information using templating.
#[macro_export]
macro_rules! kernel_source {
    (
        $struct:ident,
        $file:expr
    ) => {
        /// Generated kernel from a source file.
        #[derive(new)]
        pub struct $struct;

        impl $struct {
            fn source(&self) -> $crate::template::SourceTemplate {
                $crate::template::SourceTemplate::new(include_str!($file))
            }
        }
    };
}
