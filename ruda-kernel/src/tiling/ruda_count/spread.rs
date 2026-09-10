use crate::dsl::{RudaCount, Runtime, client::ComputeClient};

pub fn ruda_count_spread_with_total<R: Runtime>(
    client: &ComputeClient<R>,
    num_rudas: usize,
) -> (RudaCount, usize) {
    let ruda_count = ruda_count_spread(&client.properties().hardware.max_ruda_count, num_rudas);

    (
        RudaCount::Static(
            ruda_count[0] as u32,
            ruda_count[1] as u32,
            ruda_count[2] as u32,
        ),
        ruda_count[0] * ruda_count[1] * ruda_count[2],
    )
}

fn ruda_count_spread(max_ruda_count: &(u32, u32, u32), num_rudas: usize) -> [usize; 3] {
    let max_ruda_count = [max_ruda_count.0, max_ruda_count.1, max_ruda_count.2];
    let mut num_rudas = [num_rudas, 1, 1];
    let base = 2;

    let mut reduce_count = |i: usize| {
        if num_rudas[i] <= max_ruda_count[i] as usize {
            return true;
        }

        loop {
            num_rudas[i] = num_rudas[i].div_ceil(base);
            num_rudas[i + 1] *= base;

            if num_rudas[i] <= max_ruda_count[i] as usize {
                return false;
            }
        }
    };

    for i in 0..2 {
        if reduce_count(i) {
            break;
        }
    }

    num_rudas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_num_rudas_even() {
        let max = (32, 32, 32);
        let required = 2048;

        let actual = ruda_count_spread(&max, required);
        let expected = [32, 32, 2];
        assert_eq!(actual, expected);
    }

    #[test]
    fn safe_num_rudas_odd() {
        let max = (48, 32, 16);
        let required = 3177;

        let actual = ruda_count_spread(&max, required);
        let expected = [25, 32, 4];
        assert_eq!(actual, expected);
    }
}
