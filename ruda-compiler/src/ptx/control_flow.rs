use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{RangeLoop, Switch, VariableKind};
use std::collections::HashSet;

impl Emitter {
    pub fn range_loop(&mut self, branch: RangeLoop) -> Result<()> {
        let ty = Scalar::of(branch.i.ty)?;
        if !ty.integer() || branch.start.ty != branch.i.ty || branch.end.ty != branch.i.ty {
            return Err(invalid("range loop requires matching integer index and bounds"));
        }
        let step = match branch.step {
            Some(step) => {
                let step_ty = Scalar::of(step.ty)?;
                if !step_ty.integer() {
                    return Err(invalid("range loop step must be an integer"));
                }
                Some((step_ty, self.value(step)?))
            }
            None => None,
        };
        let first = self.value(branch.start)?;
        let bound = self.value(branch.end)?;
        let index = self.destination(branch.i)?;
        let start = self.label();
        let end = self.label();
        let predicate = self.reg(Scalar::Pred);
        self.line(format!("mov.{} {index}, {first};", ty.suffix()));
        self.line(format!("{start}:"));
        let comparison = if branch.inclusive { "gt" } else { "ge" };
        self.line(format!("setp.{comparison}.{} {predicate}, {index}, {bound};", ty.suffix()));
        self.line(format!("@{predicate} bra {end};"));
        self.loops.push(end.clone());
        self.scope(branch.scope)?;
        self.loops.pop();
        let increment = match step {
            Some((step_ty, value)) if step_ty != ty => {
                let converted = self.reg(ty);
                self.line(format!("cvt.{}.{} {converted}, {value};", ty.suffix(), step_ty.suffix()));
                converted
            }
            Some((_, value)) => value,
            None => "1".into(),
        };
        self.line(format!("add.{} {index}, {index}, {increment};", ty.suffix()));
        self.normalize_integer(ty, &index);
        self.line(format!("bra {start};"));
        self.line(format!("{end}:"));
        Ok(())
    }

    pub fn switch(&mut self, branch: Switch) -> Result<()> {
        let ty = Scalar::of(branch.value.ty)?;
        if !ty.integer() {
            return Err(invalid("switch requires an integer selector"));
        }
        let value = self.value(branch.value)?;
        let end = self.label();
        let default = self.label();
        let matched = self.reg(Scalar::Pred);
        let mut cases = Vec::with_capacity(branch.cases.len());
        let mut seen = HashSet::new();
        for (case, scope) in branch.cases {
            if case.ty != branch.value.ty || !matches!(case.kind, VariableKind::Constant(_)) {
                return Err(invalid("switch cases must be integer constants matching the selector"));
            }
            if !seen.insert(case) {
                return Err(invalid("duplicate switch case"));
            }
            let constant = self.value(case)?;
            let label = self.label();
            self.line(format!("setp.eq.{} {matched}, {value}, {constant};", ty.suffix()));
            self.line(format!("@{matched} bra {label};"));
            cases.push((label, scope));
        }
        self.line(format!("bra {default};"));
        // Match the existing C++ backend's case scopes and non-fallthrough exits.
        self.loops.push(end.clone());
        for (label, scope) in cases {
            self.line(format!("{label}:"));
            self.scope(scope)?;
            self.line(format!("bra {end};"));
        }
        self.line(format!("{default}:"));
        self.scope(branch.scope_default)?;
        self.loops.pop();
        self.line(format!("{end}:"));
        Ok(())
    }
}
