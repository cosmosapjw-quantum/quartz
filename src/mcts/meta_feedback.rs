// Research-only, engine-independent two-level metacognitive controller.
pub fn update_memory(_m:f64,_y:f64)->f64 { 0. }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_prediction_error_changes_memory() {
        assert_eq!(update_memory(0.,0.5),0.25);
        assert_eq!(update_memory(0.25,-0.75),-0.25);
        assert_eq!(update_memory(1.,-1.),0.);
    }
}
