use tracel_llvm::mlir_rs::{
    Context,
    dialect::{
        func,
        llvm::{
            self,
            attributes::{Linkage, linkage},
        },
    },
    ir::{
        BlockLike, Identifier, Location, Region,
        attribute::{StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
    },
};

pub fn add_external_function_to_module<'a>(
    context: &'a Context,
    module: &tracel_llvm::mlir_rs::ir::Module<'a>,
) {
    let integer_type = IntegerType::new(context, 32).into();
    let func_type = TypeAttribute::new(llvm::r#type::function(
        integer_type,
        &[llvm::r#type::pointer(context, 0)],
        true,
    ));
    module.body().append_operation(llvm::func(
        context,
        StringAttribute::new(context, "printf"),
        func_type,
        Region::new(),
        &[(
            Identifier::new(context, "linkage"),
            linkage(context, Linkage::External),
        )],
        Location::unknown(context),
    ));
    let func_name = StringAttribute::new(context, "sync_ruda");
    let func_type = TypeAttribute::new(FunctionType::new(context, &[], &[]).into());
    module.body().append_operation(func::func(
        context,
        func_name,
        func_type,
        Region::new(),
        &[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )],
        Location::unknown(context),
    ));
}
