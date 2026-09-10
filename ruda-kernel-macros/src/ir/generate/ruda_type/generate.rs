use proc_macro2::TokenStream;

use crate::ir::parse::ruda_type::RudaType;

impl RudaType {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        match self {
            RudaType::Enum(data) => data.generate(with_launch),
            RudaType::Struct(data) => data.generate(with_launch),
        }
    }
}
