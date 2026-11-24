// Four exclusive targets, each (quantized cost, signed expected utility).
pub type Options = [(usize, f64);4];
pub fn dp(rows:&[Options],_k:usize)->Vec<usize>{vec![0;rows.len()]}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ratio_counterexample_and_stop(){
        let rows=[[(0,0.),(3,5.),(9,-1.),(9,-2.)],[(0,0.),(2,3.),(9,-1.),(9,-2.)],[(0,0.),(2,3.),(9,-1.),(9,-2.)]];
        assert_eq!(dp(&rows,4),vec![0,1,1]);
        assert_eq!(dp(&rows,0),vec![0,0,0]);
    }
}
