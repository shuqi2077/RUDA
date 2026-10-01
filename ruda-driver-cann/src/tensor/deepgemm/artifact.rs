use crate::CannError;
use super::super::layout::invalid;
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::Path};

pub(super) fn read(root:&Path,key:&str)->Result<(Vec<u8>,String),CannError> {
    let dir=root.join(key);
    let manifest=dir.join("kernel.ruda");
    if std::fs::metadata(&manifest).map_err(|e|invalid(e.to_string()))?.len()>131072 {return Err(invalid("oversized artifact manifest"));}
    let text=std::fs::read_to_string(&manifest).map_err(|e|invalid(format!("cannot read compiled artifact {key}: {e}")))?;
    let entries=parse(&text)?;
    for (k,v) in [("schema","ruda.ascend.rust.bf16.v2"),("source_language","rust-device-ir"),("lowering","cann-c-intrinsics"),("arch","dav-c310"),("soc","Ascend950DT"),("cores","32"),("key",key),("object","kernel.o")] {
        if entries.get(k).map(String::as_str)!=Some(v){return Err(invalid(format!("artifact contract mismatch: {k}")));}
    }
    let name=entries.get("kernel_name").ok_or_else(||invalid("missing kernel_name"))?.clone();
    if name.is_empty()||name.contains('\0')||name.len()>65536{return Err(invalid("invalid kernel_name"));}
    let file=dir.join("kernel.o");
    let size=std::fs::metadata(&file).map_err(|e|invalid(e.to_string()))?.len();
    if !(64..=128*1024*1024).contains(&size) {return Err(invalid("invalid object size"));}
    let data=std::fs::read(&file).map_err(|e|invalid(e.to_string()))?;
    if data.len()<64 || data.len()>128*1024*1024 || data.get(..6)!=Some(&b"\x7fELF\x02\x01"[..]) || data.get(16..18)!=Some(&[2,0][..]){return Err(invalid("expected a linked little-endian ELF64 Ascend object"));}
    let digest=format!("{:x}",Sha256::digest(&data));
    if entries.get("sha256")!=Some(&digest){return Err(invalid("kernel object checksum mismatch"));}
    // Hashes detect corruption, not malicious binary code. load() requires a trusted root.
    Ok((data,name))
}
fn parse(text:&str)->Result<BTreeMap<String,String>,CannError> {
    if text.len()>131072{return Err(invalid("oversized artifact manifest"));}
    let allowed=["schema","arch","soc","cores","key","object","kernel_name","sha256","source_sha256","compiler_sha256","source_language","lowering"];
    let mut out=BTreeMap::new();
    for line in text.lines() {
        let (k,v)=line.split_once('=').ok_or_else(||invalid("malformed artifact manifest"))?;
        if !allowed.contains(&k)||out.insert(k.to_string(),v.to_string()).is_some(){return Err(invalid("unknown or duplicate manifest key"));}
    }
    if out.len()!=allowed.len(){return Err(invalid("incomplete artifact manifest"));}
    Ok(out)
}
#[cfg(test)]mod tests{use super::*;
#[test]fn malformed_manifest_rejected(){for s in ["key=x\nkey=y","unknown=x","key"]{assert!(parse(s).is_err());}}
}
