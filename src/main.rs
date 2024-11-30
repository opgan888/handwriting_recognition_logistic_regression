mod data;
use handwritingrecognition::data::injest;
use handwritingrecognition::helper::sigmoid;
use handwritingrecognition::helper::model;
use std::env;
use ndarray::{Array2};
use log::{debug, error, info, warn};
use fern::Dispatch;
use log::{LevelFilter};

fn main() -> Result<(), Box<dyn std::error::Error>>{
    // Initialize the logger
    fern::Dispatch::new()
    .format(|out, message, record| {
        out.finish(format_args!("[{}] [{}] {}", record.level(), record.target(), message))
    })
    .level(LevelFilter::Debug)    
    .chain(fern::log_file("debugging.log")?)
    .apply().unwrap();

    let args: Vec<String> = env::args().collect();
    println!("Logistic Regression classification of handwriting digits");

    // injest DIGIT, modeling DIGIT, predict_test_example INDEX, predict_unseen_example IMAGE_FILE
    let cmd: &str = &args[2]; // injest, modeling, predict_test_example, predict_unseen_example commands
    //let s_string: String = s.to_owned();
    let param: &str = &args[3]; // a string rep digit, index, file name
    match cmd {
        "injest" => Ok(injest_cmd(param)),
        "modeling" => Ok(model_cmd(param)),
        "predict_test_example" => Ok(predict_test_example_cmd(param)),
        "predict_unseen_example" => Ok(predict_unseen_example_cmd(param)),
        _ => Ok(println!(
            "Enter a command: injest, modeling, predict_test_example, predict_unseen_example"
        )),
    }
}

fn predict_test_example_cmd(string_number: &str) {
    println!(
        "Data preprocessed for recognising a digit of {}!",
        string_number
    );
    let digit: f32 = string_number.parse().unwrap();
    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
}

fn predict_unseen_example_cmd(string_number: &str) {
    println!(
        "Data preprocessed for recognising a digit of {}!",
        string_number
    );
    let digit: f32 = string_number.parse().unwrap();
    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
}

fn injest_cmd(string_number: &str) {
    println!(
        "Data preprocessed for recognising a digit of {}!",
        string_number
    );
    let digit: f32 = string_number.parse().unwrap();
    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
}

fn model_cmd(string_number: &str) {
    println!("Model trained to classify a digit of {}!", string_number);
    let digit: f32 = string_number.parse().unwrap();
    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
    let mut num_iterations = 101;
    let mut learning_rate = 0.005;
    let print_cost = true;
    let mut costs: Vec<f32> = Vec::new(); // Create an empty vector
    let mut Y_prediction_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let mut Y_prediction_train = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let mut w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let mut b = 0.0;
    let (
        costs,
        Y_prediction_test,
        Y_prediction_train,
        w,
        b,
        learning_rate,
        num_iterations,
    ) = model(
        &_train_x,
        &_train_y,
        &_test_x,
        &_test_y,
        num_iterations,
        learning_rate,
        print_cost,
    );
    // save model to a file
    //click.echo("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))
    println!("Cost is {:?}!",costs);
    //log("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))
}

/*

@cli.command()
@click.argument("digit", type=int)
def modeling(digit):
    """train NN model weights and bias to classify a given handwriting digit supplied in the argument"""

    # injest datasets
    train_set_x, train_set_y, test_set_x, test_set_y = myLib.data.injest(digit)

    # EDA

    # train model
    logistic_regression_model = myLib.helper.model(
        train_set_x,
        train_set_y,
        test_set_x,
        test_set_y,
        num_iterations=100,
        learning_rate=0.005,
        print_cost=True,
    )

    # save model to an NPY file
    np.save("model_weights.npy", logistic_regression_model["w"])
    np.save("model_bias.npy", np.array([logistic_regression_model["b"]]))

    # save datasets to an NPY file
    np.save("test_set_x.npy", test_set_x)
    np.save("test_set_y.npy", test_set_y)

    click.echo("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))
    log("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))


@cli.command()
@click.argument("example", type=int)
def predict_test(example):
    """predict a example of test dataset referenced by index supplied in the argument"""

    # injest test datasets from the NPY file
    test_set_x = np.load("test_set_x.npy")
    test_set_y = np.load("test_set_y.npy")

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar
    test_set_x_example = test_set_x[:, example].reshape(test_set_x[:, example].size, 1)
    a = "Actual = " + str(test_set_y[:, example])
    p = "Prediction = " + str(predict(w, b, test_set_x_example))
    click.echo(a)
    click.echo(p)
    log("Example " + str(example) + " :: " + a + " : " + p)


@cli.command()
@click.argument("file_name")
def predict_unseen(file_name):
    """Predict unseen example from an image file supplied by file name in the argument"""
    # injest test datasets from the NPY file
    # test_set_x = np.load("test_set_x.npy")
    num_px = 28  # test_set_x.shape[1]
    log("num_px " + str(num_px))

    # injest image file my_image.jpg
    # Preprocess the image to fit  the NN algorithm.
    fname = "assets/images/" + file_name
    image = np.array(Image.open(fname).resize((num_px, num_px)))

    imageori = np.array(Image.open(fname))
    log("imageori shape: " + str(imageori.shape))

    image = image[:, :, 0]
    # plt.imshow(image)
    log("image.shape " + str(image.shape))
    click.echo("image.shape " + str(image.shape))

    image = image / 255.0
    image = image.reshape((1, num_px * num_px)).T

    log("image.shape " + str(image.shape))
    click.echo("image.shape " + str(image.shape))

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar

    a = "Actual = " + str(file_name)
    p = "Prediction = " + str(predict(w, b, image))
    click.echo(a)
    click.echo(p)
    log("Example " + str(file_name) + " :: " + a + " : " + p)


*/
