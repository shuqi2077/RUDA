//! Compiled code is trusted input. Digests prevent stale/corrupted artifacts,
//! not malicious binaries. The expected contract comes from a real IR compile.
use std::{collections::BTreeMap,path::Path};
use sha2::{Digest,Sha256};
use ruda_compiler::ascend::AscendKernel;
use crate::{CannError,tensor::layout::invalid};
fn parse(text:&str)->Result<BTreeMap<String,String>,CannError>{
    if text.len()>131072{return Err(invalid("common-IR manifest too large"));}
    let mut map=BTreeMap::new();for line in text.lines(){let(k,v)=line.split_once('=').ok_or_else(||invalid("invalid common-IR manifest line"))?;if k.is_empty()||v.is_empty()||map.insert(k.into(),v.into()).is_some(){return Err(invalid("empty or duplicate common-IR manifest field"));}}Ok(map)
}
pub(super) fn verify_manifest(text:&str,expected:&AscendKernel,image:&[u8])->Result<(),CannError>{
    let mut actual=parse(text)?;
    for(k,v)in parse(&expected.build_contract())?{if actual.remove(&k).as_deref()!=Some(v.as_str()){return Err(invalid(format!("common-IR contract mismatch: {k}")));}}
    if actual.remove("object").as_deref()!=Some("kernel.o"){return Err(invalid("artifact object must be kernel.o"));}
    let source_hash=format!("{:x}",Sha256::digest(expected.source().as_bytes()));
    if actual.remove("source_sha256").as_deref()!=Some(source_hash.as_str()){return Err(invalid("artifact was built from a different generated device program"));}
    let object_hash=format!("{:x}",Sha256::digest(image));
    if actual.remove("object_sha256").as_deref()!=Some(object_hash.as_str()){return Err(invalid("artifact checksum mismatch"));}
    let compiler=actual.remove("compiler_sha256").ok_or_else(||invalid("missing compiler identity"))?;
    if compiler.len()!=64||!compiler.bytes().all(|c|c.is_ascii_hexdigit()){return Err(invalid("invalid compiler identity"));}
    if !actual.is_empty(){return Err(invalid("unknown artifact manifest fields"));}
    if image.len()<64||image.len()>128*1024*1024||image.get(..6)!=Some(&b"\x7fELF\x02\x01"[..])||image.get(16..18)!=Some(&[2,0][..]){return Err(invalid("expected a linked little-endian ELF64 device object"));}
    Ok(())
}
pub(super) fn read(dir:&Path,kernel:&AscendKernel)->Result<Vec<u8>,CannError>{
    let manifest=dir.join("kernel.ruda");let file=dir.join("kernel.o");
    if std::fs::metadata(&manifest).map_err(|e|invalid(e.to_string()))?.len()>131072{return Err(invalid("oversized common-IR manifest"));}
    let size=std::fs::metadata(&file).map_err(|e|invalid(e.to_string()))?.len();if !(64..=128*1024*1024).contains(&size){return Err(invalid("invalid common-IR object size"));}
    let text=std::fs::read_to_string(manifest).map_err(|e|invalid(e.to_string()))?;
    let image=std::fs::read(file).map_err(|e|invalid(e.to_string()))?;verify_manifest(&text,kernel,&image)?;Ok(image)
}
#[cfg(test)]mod tests{
    use super::*;use ruda_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,programs::{MapProgram,definition}};use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode};
    fn fixture()->(AscendKernel,Vec<u8>,String){let k=AscendCompiler.compile(definition(MapProgram::Add),&AscendOptions{target:Some(AscendTarget::Ascend950DT),elements:17,..Default::default()},ExecutionMode::Checked,UIntKind::U64.into()).unwrap();let mut image=vec![0;64];image[..6].copy_from_slice(b"\x7fELF\x02\x01");image[16]=2;let m=format!("{}object=kernel.o\nsource_sha256={:x}\nobject_sha256={:x}\ncompiler_sha256={}\n",k.build_contract(),Sha256::digest(k.source().as_bytes()),Sha256::digest(&image),"0".repeat(64));(k,image,m)}
    #[test]fn exact_compiler_contract_required(){let(k,b,m)=fixture();assert!(verify_manifest(&m,&k,&b).is_ok());for changed in [m.replace("elements=17","elements=18"),m.replace("Ascend950DT","OtherSoc"),m.replace("binding_2=2,w,68","binding_2=2,r,68"),m.replace("kernel.o","../kernel.o"),format!("{m}unknown=yes\n")]{assert!(verify_manifest(&changed,&k,&b).is_err());}}
    #[test]fn corrupted_object_is_rejected(){let(k,mut b,m)=fixture();b[33]=1;assert!(verify_manifest(&m,&k,&b).is_err());}
    #[test]fn non_binary_or_emit_only_is_rejected(){let(k,_,m)=fixture();assert!(verify_manifest(&m,&k,b"source, not binary").is_err());assert!(verify_manifest(&k.build_contract(),&k,&[0;64]).is_err());}
}

#[cfg(test)] mod row_contract_tests {
    use super::*;
    use ruda_compiler::ascend::{AscendCompiler, AscendOptions, AscendTarget,
        row_programs::{self, RowProgram}};
    use ruda_core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode};
    #[test] fn row_manifest_rejects_wrong_stat_shape_or_plane() {
        let k = AscendCompiler.compile(row_programs::definition(RowProgram::RmsNorm, 64, 1e-5).unwrap(),
            &AscendOptions { target: Some(AscendTarget::Ascend950DT), elements: 192,
                row_width: Some(64), ..Default::default() }, ExecutionMode::Checked, UIntKind::U64.into()).unwrap();
        // Deliberately synthetic parser fixture, not an executable CANN artifact.
        let mut image = vec![0; 64]; image[..6].copy_from_slice(b"\x7fELF\x02\x01"); image[16] = 2;
        let m = format!("{}object=kernel.o\nsource_sha256={:x}\nobject_sha256={:x}\ncompiler_sha256={}\n",
            k.build_contract(), Sha256::digest(k.source().as_bytes()), Sha256::digest(&image), "0".repeat(64));
        assert!(verify_manifest(&m, &k, &image).is_ok());
        assert!(verify_manifest(&m.replace("binding_3=3,w,12", "binding_3=3,w,768"), &k, &image).is_err());
        assert!(verify_manifest(&m.replace("logical_plane=32", "logical_plane=64"), &k, &image).is_err());
        assert!(verify_manifest(&m.replace("common-row.v1", "common-map.v1"), &k, &image).is_err());
    }
}
