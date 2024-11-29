use handwritingrecognition::helper::sigmoid;
use handwritingrecognition::helper::initialize_with_zeros;
use ndarray::Array2;

#[test]
fn test_sigmoid() {
    assert_eq!(sigmoid(0.0), 0.5);
}

#[test]
fn test_initialize_with_zeros() {
    let (w, b) = initialize_with_zeros(2);
    assert_eq!(w, Array2::zeros((2, 1)));
    assert_eq!(b, 0.0);
}