#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Major { K, Mn }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind { Dense, Batched, MGrouped }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output { Bf16, F32 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub kind: Kind,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub output: Output,
}
impl Spec {
    pub fn all() -> Vec<Self> {
        let mut out = Vec::new();
        for kind in [Kind::Dense, Kind::Batched] {
            for a in [false, true] { for b in [false, true] {
                for output in [Output::Bf16, Output::F32] {
                    out.push(Self { kind, transpose_a:a, transpose_b:b, output });
                }
            }}
        }
        for output in [Output::Bf16, Output::F32] {
            out.push(Self {kind:Kind::MGrouped,transpose_a:false,transpose_b:true,output});
        }
        out
    }
    pub fn check(self) -> Result<(), String> {
        if self.kind == Kind::MGrouped && (self.transpose_a || !self.transpose_b) {
            return Err("grouped source currently accepts NT storage only".into());
        }
        crate::layout::TileLayout::new(self).map(|_| ())
    }
    pub fn key(self) -> String {
        let k = match self.kind {Kind::Dense=>"dense",Kind::Batched=>"batched",Kind::MGrouped=>"mgrouped"};
        format!("bf16_{k}_{}{}_{}", if self.transpose_a {"t"} else {"n"},
            if self.transpose_b {"t"} else {"n"}, if self.output==Output::F32 {"f32"} else {"bf16"})
    }
    pub fn entry(self) -> String { format!("ruda_rust_{}", self.key()) }
    pub fn major_a(self) -> Major { if self.transpose_a {Major::Mn} else {Major::K} }
    // The kernel treats B as the logical [N,K] operand of A * B^T.
    pub fn major_b(self) -> Major { if self.transpose_b {Major::K} else {Major::Mn} }
    pub fn block_m(self) -> u64 { if self.kind==Kind::MGrouped {256} else {64} }
    pub fn output_bytes(self) -> u64 {if self.output==Output::F32 {4} else {2}}
}
