use handwritingrecognition::helper::sigmoid;
use handwritingrecognition::helper::initialize_with_zeros;
use handwritingrecognition::helper::propagate;
use ndarray::{Array2, array};

#[test]
fn test_sigmoid() {
    let input = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let result = sigmoid(input);
    let expected = Array2::from_shape_vec((2, 1), vec![0.5, 0.5]).unwrap();  
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
    
    //let data = vec![1.0];
    //let shape = (1, 1);

    let w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let b = 0.0;

    let X = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    let Y = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

    let (dw, db, cost) = propagate(&w, b, &X, &Y);

    let result =  &dw;
    let expected = Array2::from_shape_vec((2, 1), vec![-0.5, -0.5]).unwrap(); 
    assert_eq!(result, expected);

    let result =  cost;
    let expected = 0.30103; 
    assert_eq!(result, expected);

    let result =  db;
    let expected = -0.5; 
    assert_eq!(result, expected);

}