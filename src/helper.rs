
use ndarray::{Array2};
use std::f32::consts::LOG10_E;
use std::collections::HashMap;
use std::f32::consts::E;

//pub fn sigmoid(z: f32) -> f32 {
pub fn sigmoid(z: Array2<f32>) -> Array2<f32>{
    /*
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z), a probability between 0 and 1
    */
    // 1 / (1 + np.exp(-z))
    // Apply the exponential function to each element
    // let exp_array = (-z).mapv(|x| E.powf(x));
    1.0 / (1.0 + (-z).mapv(|x| E.powf(x)))
}

pub fn initialize_with_zeros(dim: usize) -> (Array2<f32>, f32) {
    /*
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    */

    let w: Array2<f32> = Array2::zeros((dim, 1));
    let owned_w = w.to_owned();
    let b = 0.0;
    (owned_w, b)
}


pub fn propagate<K, V>(w: Array2<f32>, b: f32, X: Array2<f32>, Y: Array2<f32>) -> (HashMap<K, V>, f32){
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
    let m = X.shape()[1];

    /*
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    # compute cost by using np.dot to perform multiplication.
    # And don't use loops for the sum.
    */
    
    // A = sigmoid(np.dot(np.transpose(w), X) + b)
    let A = sigmoid(w.t().dot(&X) + b);

    // cost = -(1 / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    let cost = (-1 / m) * (Y * A.mapv(|x| x.log10()) + (1 - Y) * (1 - A).mapv(|x| x.log10())).iter().sum();

    //# BACKWARD PROPAGATION (TO FIND GRAD)
    
    // dw = (1 / m) * np.dot(X, np.transpose(A - Y))
    let dw = (1 / m) * X.dot(&(A - Y).t());

    // db = (1 / m) * np.sum(A - Y)
    let db = (1 / m) * (A - Y).iter().sum();

    // cost = np.squeeze(np.array(cost))
    let mut grads = HashMap::new();

    // grads = {"dw": dw, "db": db}
    // Insert key-value pairs
    grads.insert("dw", dw);
    grads.insert("db", db);

    (grads, cost)
}
