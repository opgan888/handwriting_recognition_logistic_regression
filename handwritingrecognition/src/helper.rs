use statrs::distribution::Exp;

pub fn sigmoid(z: f32) -> f32 {
    /*
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z), a probability between 0 and 1
    */
    // 1 / (1 + np.exp(-z))
    1.0 / (1.0 + f32::exp(-z))
}
