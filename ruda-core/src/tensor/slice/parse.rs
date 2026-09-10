use super::Slice;
use alloc::format;
use core::fmt::{Display, Formatter};
use core::str::FromStr;

impl Display for Slice {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        if self.step == 1
            && let Some(end) = self.end
            && self.start == end - 1
        {
            f.write_fmt(format_args!("{}", self.start))
        } else {
            if self.start != 0 {
                f.write_fmt(format_args!("{}", self.start))?;
            }
            f.write_str("..")?;
            if let Some(end) = self.end {
                f.write_fmt(format_args!("{}", end))?;
            }
            if self.step != 1 {
                f.write_fmt(format_args!(";{}", self.step))?;
            }
            Ok(())
        }
    }
}

impl FromStr for Slice {
    type Err = crate::tensor::errors::ExpressionError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut s = source.trim();

        let parse_int = |v: &str| -> Result<isize, Self::Err> {
            v.parse::<isize>().map_err(|e| {
                crate::tensor::errors::ExpressionError::parse_error(
                    format!("Invalid integer: '{v}': {}", e),
                    source,
                )
            })
        };

        let mut start: isize = 0;
        let mut end: Option<isize> = None;
        let mut step: isize = 1;

        if let Some((head, tail)) = s.split_once(";") {
            step = parse_int(tail)?;
            s = head;
        }

        if s.is_empty() {
            return Err(crate::tensor::errors::ExpressionError::parse_error(
                "Empty expression",
                source,
            ));
        }

        if let Some((start_s, end_s)) = s.split_once("..") {
            if !start_s.is_empty() {
                start = parse_int(start_s)?;
            }
            if !end_s.is_empty() {
                if let Some(end_s) = end_s.strip_prefix('=') {
                    end = Some(parse_int(end_s)? + 1);
                } else {
                    end = Some(parse_int(end_s)?);
                }
            }
        } else {
            start = parse_int(s)?;
            end = Some(start + 1);
        }

        if step == 0 {
            return Err(crate::tensor::errors::ExpressionError::invalid_expression(
                "Step cannot be zero",
                source,
            ));
        }

        Ok(Slice::new(start, end, step))
    }
}
