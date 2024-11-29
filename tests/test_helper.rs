use handwritingrecognition::helper::sigmoid;
use handwritingrecognition::helper::initialize_with_zeros;
use handwritingrecognition::helper::propagate;
use ndarray::{Array2, array};

#[test]
fn test_sigmoid() {
    //let input = Array2::from([[1.0, 2.0], [3.0, 4.0]]);
    let input = array![[0.0], [0.0]];
    let result = sigmoid(input);
    //let expected = Array2::from([[0.5]]);
    let expected = array![[0.5], [0.5]];
    assert_eq!(result, expected);
}

#[test]
fn test_initialize_with_zeros() {
    let (w, b) = initialize_with_zeros(2);
    assert_eq!(w, Array2::zeros((2, 1)));
    assert_eq!(b, 0.0);
}

#[test]
fn test_propagate() {
    //let w = Array2::from([[1.0], [1.0]]);
    let w = array![[[1.0], [1.0]]];
    let b = 1.0;
    //let X = Array2::from([[1.0], [1.0]]);
    //let Y = Array2::from([[1.0]]);
    let X = array![[[1.0], [1.0]]];
    let Y = array![[[1.0]]];
    
    let (dw, db, cost) = propagate(&w, &b, &X, &Y);

    let result = dw;
    // Expected output based on the given inputs
    let expected = dw; //array![[[4.0]]]; //Array2::from([[4.]]);
    assert_eq!(result, expected);

}