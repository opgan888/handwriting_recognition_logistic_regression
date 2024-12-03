use log::{debug, error, info};
use ndarray::Array2;
use std::f32::consts::E;
//use handwritingrecognition::data::find_indices_filter;
use crate::data::find_indices_filter;

pub fn cost_cal(a: &Array2<f32>) -> f32 {
    let m=2.0;
    let y = &Array2::from_shape_vec((1, 2),  vec![0.0, 0.0]).unwrap();
    let result = (-1.0 / m) * ((y * (a.mapv(|e| e.log10())) + (1.0 - y) * ((1.0 - a).mapv(|d| d.log10())))
    .iter()
    .sum::<f32>());

    result
}




pub fn element_log(a: &Array2<f32>) -> Array2<f32> {
    let result: Array2<f32> = a.mapv(|e| e.log10());
    result
}

pub fn element_product(a: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    let result: Array2<f32> = y * a;
    result
}

pub fn element_sum(a: &Array2<f32>, _y: &Array2<f32>) -> f32 {
    let result: f32 = a.iter().sum();
    result
}

pub fn matrixmultiply(w: &Array2<f32>, b: f32, x: &Array2<f32>) -> Array2<f32> {
    (w.t()).dot(x) + b
}

/*

fn safe_powf(base: f64, exponent: f64) -> Result<f64, String> {
    if base < 0.0 && exponent.fract() != 0.0 {
        return Err("Negative base with non-integer exponent".to_string());
    }

    let result = base.powf(exponent);

    if result.is_nan() || result.is_infinite() {
        return Err("Numerical instability".to_string());
    }

    Ok(result)
}

*/

//pub fn sigmoid(z: f32) -> f32 {
pub fn sigmoid(z: Array2<f32>) -> Array2<f32> {
    /*
    Compute the sigmoid of z as 1 / (1 + np.exp(-z))
    Apply the exponential function to each element

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z), a probability between 0 and 1
    */

    // 1.0 / (1.0 + (-z).mapv(|x| E.powf(x)))

    1.0 / (1.0 +  z.mapv(|x| (-x).exp()) )

}

// try Result error next time
pub fn initialize_with_zeros(dim: usize) -> (Array2<f32>, f32) {
    //pub fn initialize_with_zeros(dim: usize) -> Result<(Array2<f32>, f32), String> {
    /*
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    */

    /*
    if dim == 0 {
        return Err("0 dimension array not allowed".to_string());
    }
    */

    let w: Array2<f32> = Array2::zeros((dim, 1));
    let owned_w = w.to_owned();
    let b = 0.0;
    (owned_w, b)

    // Ok((owned_w, b))
}

// use dot matrix slower than python
pub fn propagate(
    w: &Array2<f32>,
    b: f32,
    x: &Array2<f32>,
    y: &Array2<f32>,
    //dw: &Array2<f32>, // remove if not working
) -> (Array2<f32>, f32, f32) { // remove & if not working
    /*
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    */

    // m = X.shape[1] // (num_px * num_px * 3, number of examples)
    let m: f32 = x.shape()[1] as f32; // cast a `usize` to an `f32`

    /*
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    # compute cost by using np.dot to perform multiplication.
    # And don't use loops for the sum.
    */


    //let w: Array2<f32> = Array2::zeros((784, 1));
    //let x1: &Array2<f32> = &Array2::zeros((784, 60000));

    //let A = sigmoid(np.dot(np.transpose(w), X) + b)
    // let z = (w.t()).dot(x) + b;
    //let wt = w.t();
    //println!("w shape: {:?}", w.shape());
    //println!("x shape: {:?}", x.shape());


    //println!("Bef dot");
    //let z1 = wt.dot(x);
    //println!("After dot");

    //let z = z1 + b;
  
    // let z = w.t().dot(x) + b;

    let a = sigmoid(w.t().dot(x) + b);
    //let a = sigmoid(z);

    // log (python) and ln (Rust) refers to natural log
    // cost = -(1 / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    /*
    let cost = (-1.0 / m)
        * ((y * (&a.mapv(|e| e.ln())) + (1.0 - y) * ((1.0 - &a).mapv(|d| d.ln())))
            .iter()
            .sum::<f32>());
    */

    let cost: f32 = -(y * (&a.mapv(|e| e.ln())) + (1.0 - y) * ((1.0 - &a).mapv(|d| d.ln()))).sum()/m;
    
    //println!("cost {:?} ", cost);

    /*
    println!(
        "y * (&a.mapv(|e| e.log())) {:?} ", y * (&a.mapv(|e| e.ln()))
    );

    println!(
        "m {:?} ", m
    );

    println!(
        "a {:?} ", a
    );

    println!(
        "cost {:?} ", cost
    ); */

    //# BACKWARD PROPAGATION (TO FIND GRAD)

    // dw = (1 / m) * np.dot(X, np.transpose(A - Y))
    // let dw = (1.0 / m) * x.dot(&((&a - y).t())); // // Negate each element

    // let dw = x.dot(&((&a - y).t()))/m; // // Negate each element
    let dw = x.dot(&((&a - y).t()))/m; // // Negate each element

    // db = (1 / m) * np.sum(A - Y)
    // let db = (1.0 / m) * (&a - y).iter().sum::<f32>();
    let db: f32 = (&a - y).sum()/m;

    // (dw, db, cost)
    (dw, db, cost)
}

pub fn optimize(
    w: &Array2<f32>,
    b: f32,
    x: &Array2<f32>,
    y: &Array2<f32>,
    num_iterations: i32,
    learning_rate: f32,
    print_cost: bool,
) -> Result<(Array2<f32>, f32, Array2<f32>, f32, Vec<f32>), Box<dyn std::error::Error>> {
    // (Array2<f32>, f32, Array2<f32>, f32, Vec<f32>)
    /*
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    */

    info!("optimize starts");

    //w = copy.deepcopy(w)
    //b = copy.deepcopy(b)

    let mut w_owned = w.to_owned();
    let mut b_owned = b;

    // let dw = , db, cost)
    // costs = []
    let mut costs = Vec::new(); // Create an empty vector

    let mut dw: Array2<f32> = Array2::zeros((x.shape()[0], 1)); // (row (features), col (examples)) refers to (num_px * num_px * 3, number of examples)
    let mut db = 0.0;
    let mut cost = 0.0;

    for i in 0..num_iterations {
        //for i in range(num_iterations):
        // Cost and gradient calculation
        // grads, cost = ...
        // grads, cost = propagate(w, b, X, Y)
        // (dw, db, cost) = propagate(w, b, X, Y);
        //(dw, db, cost) = propagate(&w_owned, b_owned, x, y);
        (dw, db, cost) = propagate(&w_owned, b_owned, x, y);
        // # YOUR CODE ENDS HERE

        // info!("optimize debug message: db {:?}.", db);
        // info!("optimize debug message: cost {:?}.", cost);

        /*
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]


        // update rule (≈ 2 lines of code)
        w -= learning_rate * dw
        b -= learning_rate * db
        */
     

        w_owned = w_owned - learning_rate * &dw; // Dereference w_owned and apply element-wise multiplication
        b_owned = b_owned - learning_rate * db;
        //w = &w_owned;
        //b = b_owned;

        // Record the costs print interval
        let print_interval = 100;
        if i % print_interval == 0 {
            //if i % 100 == 0:
            // costs.append(cost)
            costs.push(cost);

            // Print the cost every 100 training iterations if request is True
            // print!("Cost after iteration %i: %f" % (i, cost))
            if print_cost {
                println!("Cost after iteration {:?}: {:?} {:?}", i, cost, b_owned);
            }
        }
    }
    /*
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    (params, grads, costs)
    */
    Ok((w_owned, b_owned, dw, db, costs))
}

pub fn predict(w: &Array2<f32>, b: f32, x: &Array2<f32>) -> Array2<f32> {
    /*
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    */

    // m = X.shape[1]
    let m = x.shape()[1];

    // Y_prediction = np.zeros((1, m))
    let mut y_prediction: Array2<f32> = Array2::zeros((1, m));

    // ensure column vector
    // w = w.reshape(X.shape[0], 1)
    assert_eq!(w.shape(), &[x.shape()[0], 1]);
    w.to_shape((x.shape()[0], 1)).unwrap();

    // # Compute vector "A" predicting the probabilities of a cat being present in the picture
    // A = sigmoid(np.dot(np.transpose(w), X) + b)
    let a = sigmoid((w.t()).dot(x) + b);

    // # Using loop
    /*
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    */
    //# Using no loop for better efficieny
    //# Y_prediction[A > 0.5] = 1
    // Iterate over the elements of 'a' and assign values to 'y_prediction'
    for ((i, j), value) in a.indexed_iter() {
        if *value > 0.5 {
            y_prediction[(i, j)] = 1.0;
        }
    }

    y_prediction
}

pub fn model(
    x_train: &Array2<f32>,
    y_train: &Array2<f32>,
    x_test: &Array2<f32>,
    y_test: &Array2<f32>,
    num_iterations: i32,
    learning_rate: f32,
    print_cost: bool,
) -> (
    Vec<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    f32,
    f32,
    i32,
) {
    /*
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    */

    /*
    # initialize parameters with zeros
    # w, b = ...

    # Gradient descent
    # params, grads, costs = ...

    # Retrieve parameters w and b from dictionary "params"
    # w = ...
    # b = ...

    # Predict test/train set examples
    # Y_prediction_test = ...
    # Y_prediction_train = ...
    */

    let (w, b) = initialize_with_zeros(x_train.shape()[0]);

    /*
    let result = initialize_with_zeros(x_train.shape()[0]);
    match result {
        Ok((w, b)) => (w, b),
        Err(error) => error!("Error initialize_with_zeros in model(): {:?}", error), // Keep the message for logging
    }
    */
    /*
    params, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost
    )
    */
    let Ok((w, b, _dw, _db, costs)) = optimize(
        &w,
        b,
        x_train,
        y_train,
        num_iterations,
        learning_rate,
        print_cost,
    ) else {
        todo!()
    };
    // println!("logging exception");

    // w = params["w"]
    // b = params["b"]

    // Y_prediction_test = predict(w, b, X_test)
    // Y_prediction_train = predict(w, b, X_train)

    let y_prediction_test = predict(&w, b, x_test);
    let y_prediction_train = predict(&w, b, x_train);

    // # Print train/test Errors
    /*
    if print_cost:
        print(
            "train accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
            )
        )
        print(
            "test accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
            )
        )
        */
    if print_cost {
        println!(
            "train accuracy: {:?}",
            100.0 - ((&y_prediction_train - y_train).abs()).mean().unwrap() * 100.0
        );
        println!(
            "test accuracy: {:?}",
            100.0 - ((&y_prediction_test - y_test).abs()).mean().unwrap() * 100.0
        );
        /*
        println!(
            " model: y_prediction_test.shape {:?}",
            y_prediction_test.shape()
        );
        info!(
            " model: y_prediction_test.shape {:?}",
            y_prediction_test.shape()
        );
        */
    }

    let target_value: f32 = 1.0;
    let first_row: Vec<f32> = y_prediction_test.row(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_w = find_indices_filter(&first_row, &target_value); // search Vector of  Vec<f32>

    info!(
        "Found given digit {:?} times out of total {:?} in _y_prediction_test",
        index3_w.len(),
        first_row.len()
    );

    /*
    info!(
        "Found given digit {:?} times out of total {:?} in _y_prediction_test ... {:?}",
        index3_w.len(),
        first_row.len(),
        index3_w
    );
    */
    let target_value: f32 = 1.0;
    let first_row: Vec<f32> = y_test.row(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_w = find_indices_filter(&first_row, &target_value); // search Vector of  Vec<f32>

    info!(
        "Found given digit {:?} times out of total {:?} in y_test ...",
        index3_w.len(),
        first_row.len()
    );


    let target_value: f32 = 1.0;
    let first_row: Vec<f32> = y_prediction_train.row(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_w = find_indices_filter(&first_row, &target_value); // search Vector of  Vec<f32>

    info!(
        "Found given digit {:?} times out of total {:?} in y_prediction_train ...",
        index3_w.len(),
        first_row.len()
    );

    let target_value: f32 = 1.0;
    let first_row: Vec<f32> = y_train.row(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_w = find_indices_filter(&first_row, &target_value); // search Vector of  Vec<f32>
    info!(
        "Found given digit  {:?} times out of total {:?} in y_train ...",
        index3_w.len(),
        first_row.len()
    );
    /*
    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    } */

    (
        costs,
        y_prediction_test,
        y_prediction_train,
        w,
        b,
        learning_rate,
        num_iterations,
    )
}
