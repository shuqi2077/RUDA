use super::*;

#[test]
fn test_reshape_analysis_is_contiguous() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 1, 1, 1].into()),
        &[1, 1, 32, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
}

#[test]
fn test_reshape_analysis_is_contiguous_2() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 8].into(),
        Some(&[8, 8, 8, 1].into()),
        &[1, 1, 32, 1, 1, 8].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
}

#[test]
fn test_reshape_analysis_broadcasted_batch() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 32, 32, 32].into()),
        &[1, 1, 32, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::Broadcasted)
}

#[test]
fn test_reshape_analysis_unsqueeze_split() {
    // Unsqueeze
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 32, 32, 32].into()),
        &[32, 1, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::Split)
}

#[test]
fn test_reshape_analysis_split() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 32, 32, 32].into()),
        &[4, 8, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::Split)
}
