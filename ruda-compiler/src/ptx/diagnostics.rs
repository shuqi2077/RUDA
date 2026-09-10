use super::{Result, emit::Emitter, invalid};
use ruda_core::{compiler::CompilationError, ir::{OpCode, SourceLoc}};

pub(super) fn context(mut error: CompilationError, context: impl std::fmt::Display) -> CompilationError {
    let reason = match &mut error {
        CompilationError::UnsupportedInstruction { reason, .. } |
        CompilationError::Validation { reason, .. } |
        CompilationError::Generic { reason, .. } => reason,
    };
    reason.push_str(&format!("\n  in {context}"));
    error
}

pub(super) fn instruction_context(error: CompilationError, opcode: OpCode, source: Option<&SourceLoc>) -> CompilationError {
    if let Some(source) = source {
        context(error, format!("{opcode:?} at {:?}:{}:{} ({})", source.source.file, source.line, source.column, source.source.function_name))
    } else { context(error, format!("{opcode:?} (IR source location unavailable)")) }
}

fn filename_literal(filename: &str) -> String {
    let mut result = String::from("\"");
    for byte in filename.bytes() {
        match byte {
            b'"' => result.push_str("\\\""),
            b'\\' => result.push_str("\\\\"),
            0x20..=0x7e => result.push(char::from(byte)),
            _ => result.push_str(&format!("\\{byte:03o}")),
        }
    }
    result.push('"');
    result
}

impl Emitter {
    pub fn source_location(&mut self, source: Option<&SourceLoc>) -> Result<()> {
        let Some(source) = source else { return Ok(()); };
        if source.source.file.is_empty() || source.line == 0 { return Ok(()); }
        let file = source.source.file.as_ref();
        let index = if let Some(index) = self.source_files.get(file) { *index }
            else {
                let index = u32::try_from(self.source_files.len()).ok().and_then(|index| index.checked_add(1))
                    .ok_or_else(|| invalid("PTX source file index overflow"))?;
                self.module_declarations += &format!(".file {index} {}\n", filename_literal(file));
                self.source_files.insert(file.to_owned(), index);
                index
            };
        let location = (index, source.line, source.column);
        if self.current_source != Some(location) {
            self.line(format!(".loc {index} {} {}", source.line, source.column));
            self.current_source = Some(location);
        }
        Ok(())
    }
}
