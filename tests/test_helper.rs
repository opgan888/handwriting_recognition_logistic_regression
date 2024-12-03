use handwritingrecognition::helper::element_log;
use handwritingrecognition::helper::element_product;
use handwritingrecognition::helper::element_sum;
use handwritingrecognition::helper::initialize_with_zeros;
use handwritingrecognition::helper::matrixmultiply;
use handwritingrecognition::helper::model;
use handwritingrecognition::helper::optimize;
use handwritingrecognition::helper::predict;
use handwritingrecognition::helper::propagate;
use handwritingrecognition::helper::sigmoid;
use handwritingrecognition::helper::cost_cal;

use ndarray::Array2;
use ndarray_npy::write_npy;
use ndarray_npy::NpzReader;
use std::fs::File;

#[test]
fn test_cost_cal() {
    let a = Array2::from_shape_vec((1, 2), vec![0.9, 0.9]).unwrap();
    let result = cost_cal(&a);
    let expected = 0.0;
    assert_eq!(result, expected, "test_cost_cal algo failed");
}

#[test]
fn test_element_log() {
    let a = Array2::from_shape_vec((2, 1), vec![1.0, 0.5]).unwrap();
    let result = element_log(&a);
    let expected = Array2::from_shape_vec((2, 1), vec![0.0, -0.30103]).unwrap();

    assert_eq!(result, expected, "test_element_log algo failed");
}

#[test]
fn test_element_product() {
    let y = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    let a = Array2::from_shape_vec((2, 1), vec![5.0, 5.0]).unwrap();
    let result = element_product(&a, &y);
    let expected = Array2::from_shape_vec((2, 1), vec![5.0, 10.0]).unwrap();

    assert_eq!(result, expected, "test_element_product algo failed");
}

#[test]
fn test_element_sum() {
    let y = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let a = Array2::from_shape_vec((2, 1), vec![5.0, 5.0]).unwrap();
    let result = element_sum(&a, &y);
    let expected = 10.0;

    assert_eq!(result, expected, "test_element_sum algo failed");
}

#[test]
fn test_matrixmultiply() {
    let w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let w = Array2::from_shape_vec((2, 1), vec![5.0, 5.0]).unwrap();
    let b = 5.0;

    let x = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    let result = matrixmultiply(&w, b, &x);
    let result = sigmoid(result);
    let expected = Array2::from_shape_vec((1, 1), vec![0.99999964]).unwrap();

    assert_eq!(result, expected, "test_matrixmultiply algo failed");
}

#[test]
fn test_sigmoid() {
    let input = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap(); // test 0.0
    let input = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(); // test 1.0
    let result = sigmoid(input);
    let expected = Array2::from_shape_vec((2, 1), vec![0.5, 0.5]).unwrap();
    let expected = Array2::from_shape_vec((2, 1), vec![0.7310586, 0.7310586]).unwrap();

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

    let Ok((w, b, dw, db, costs)) =
        optimize(&w, b, &X, &Y, num_iterations, learning_rate, print_cost)
    else {
        todo!()
    };

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

    //let _ = write_npy("array.npy", &w);
    // let mut npz = NpzReader::new(File::open("arrays.npz")?)?;

    let result = b;
    let expected = 0.0;
    assert_eq!(result, expected, "b computation algo failed");
}
