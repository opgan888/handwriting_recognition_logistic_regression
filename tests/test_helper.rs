use handwritingrecognition::helper::initialize_with_zeros;
use handwritingrecognition::helper::model;
use handwritingrecognition::helper::optimize;
use handwritingrecognition::helper::predict;
use handwritingrecognition::helper::propagate;
use handwritingrecognition::helper::sigmoid;

use ndarray::{Array2};

#[test]
fn test_sigmoid() {
    let input = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let result = sigmoid(input);
    let expected = Array2::from_shape_vec((2, 1), vec![0.5, 0.5]).unwrap();
    assert_eq!(result, expected, "sigmoid computation algo failed");
}

#[test]
fn test_initialize_with_zeros() {
    let (w, b) = initialize_with_zeros(2);
    assert_eq!(w, Array2::zeros((2, 1)), "w computation algo failed");
    assert_eq!(b, 0.0, "b computation algo failed");
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

    let result = &dw;
    let expected = Array2::from_shape_vec((2, 1), vec![-0.5, -0.5]).unwrap();
    assert_eq!(result, expected, "dw computation algo failed");

    let result = cost;
    let expected = 0.30103;
    assert_eq!(result, expected, "cost computation algo failed");

    let result = db;
    let expected = -0.5;
    assert_eq!(result, expected, "db computation algo failed");
}

#[test]
fn test_optimize() {
    //let data = vec![1.0];
    //let shape = (1, 1);

    let w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let b = 0.0;

    let X = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    let Y = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let num_iterations = 101;
    let learning_rate = 0.005;
    let print_cost = true;

    let (w, b, dw, db, costs) = optimize(&w, b, &X, &Y, num_iterations, learning_rate, print_cost);

    let result = &dw;
    let expected = Array2::from_shape_vec((2, 1), vec![-0.5, -0.5]).unwrap();
    assert_eq!(result, expected, "dw computation algo failed");

    if !costs.is_empty() {
        let result = costs[costs.len() - 1]; // Access the last element using its index
        let expected = 0.30103;
        assert_eq!(result, expected, "costs computation algo failed");
    } else {
        assert!(false, "costs vector empty!");
    }

    let result = db;
    let expected = -0.5;
    assert_eq!(result, expected, "db computation algo failed");
}

#[test]
fn test_predict() {
    //let data = vec![1.0];
    //let shape = (1, 1);

    let w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let b = 0.0;

    let X = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    let mut Y_prediction = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

    Y_prediction = predict(&w, b, &X);

    let result = &Y_prediction;
    let expected = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
    assert_eq!(result, expected, "Y prediction algo failed");
}

#[test]
fn test_model() {
    //let data = vec![1.0];
    //let shape = (1, 1);

    let mut w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let mut b = 0.0;

    let X_train = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    let X_test = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    let Y_train = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let Y_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let mut num_iterations = 101;
    let mut learning_rate = 0.005;
    let print_cost = true;
    let mut costs = Vec::new(); // Create an empty vector
    let mut Y_prediction_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let mut Y_prediction_train = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

    (
        costs,
        Y_prediction_test,
        Y_prediction_train,
        w,
        b,
        learning_rate,
        num_iterations,
    ) = model(
        &X_train,
        &Y_train,
        &X_test,
        &Y_test,
        num_iterations,
        learning_rate,
        print_cost,
    );

    let result = b;
    let expected = 0.0;
    assert_eq!(result, expected, "b computation algo failed");
}
