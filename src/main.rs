mod data;
use fern::Dispatch;
use handwritingrecognition::data::find_indices_filter;
use handwritingrecognition::data::injest;
use handwritingrecognition::helper::model;
use handwritingrecognition::helper::sigmoid;
use log::LevelFilter;
use log::{debug, error, info};
use ndarray::{arr2, Array2};
use ndarray_npy::read_npy;
use ndarray_npy::write_npy;
use ndarray_npy::ReadNpyExt;
use ndarray_npy::WriteNpyExt;
use npy::NpyData;
use std::env;

use ndarray::ErrorKind;
use ndarray::ShapeError;
use std::error::Error;
use std::num::ParseFloatError;

use ndarray_npy::ReadNpyError;
use ndarray_npy::WriteNpyError;
use std::convert::From;
use std::fs::File;

#[derive(Debug)]
enum Errors {
    ShapeError(ShapeError),
    ParseFloatError(ParseFloatError),
    ReadNpyError(ReadNpyError),
    WriteNpyError(WriteNpyError),
}

impl From<ReadNpyError> for Errors {
    fn from(err: ReadNpyError) -> Self {
        Errors::ReadNpyError(err)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger
    let _ = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                chrono::Local::now().format("%d-%m-%Y %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(LevelFilter::Debug)
        .chain(fern::log_file("debugging.log")?)
        .apply();
    //.unwrap();

    let args: Vec<String> = env::args().collect();
    println!("Logistic Regression classification of handwriting digits");

    // injest DIGIT, modeling DIGIT, predict_test_example INDEX, predict_unseen_example IMAGE_FILE
    let cmd: &str = &args[2]; // injest, modeling, predict_test_example, predict_unseen_example commands
                              //let s_string: String = s.to_owned();
    let param: &str = &args[3]; // a string rep digit, index, file name
    match cmd {
        "injest" => {
            let result = injest_cmd(param);
            match result {
                Ok(()) => info!("injest_cmd Ok"),
                Err(error) => error!("Error processing in injest_cmd {:?} : {:?}", param, error), // Keep the message for logging
            }
            Ok(())
        }
        "modeling" => {
            let result = model_cmd(param);
            match result {
                Ok(()) => info!("model_cmd Ok"),
                Err(error) => error!("Error processing in model_cmd {:?} : {:?}", param, error), // Keep the message for logging
            }
            Ok(())
        }
        "predict_test_example" => {
            let result = predict_test_example_cmd(param);
            match result {
                Ok(()) => info!("predict_test_example_cmd Ok"),
                Err(error) => error!(
                    "Error processing in predict_test_example_cmd {:?} : {:?}",
                    param, error
                ), // Keep the message for logging
            }
            Ok(())
        }
        "predict_unseen_example" => {
            let _ = predict_unseen_example_cmd(param);
            Ok(())
        }
        _ => {
            println!(
                "Enter a correct command from these: injest, modeling, predict_test_example, predict_unseen_example"
            );
            Ok(())
        }
    }
}

fn predict_test_example_cmd(string_number: &str) -> Result<(), Errors> {
    /*
        Predict a test example from test dataset
    */
    println!(
        "Predict a digit from test dateset for given index of {}!",
        string_number
    );

    let digit: f32 = parse_digit(string_number)?;

    let w: Array2<f32> = read_npy("model_weights.npy")?;
    let b: Array2<f32> = read_npy("model_bias.npy")?;
    info!("predict_test_example_cmd: read_npy w {:?}", w.shape());
    info!("predict_test_example_cmd: read_npy b shape {:?}", b.shape());
    info!("predict_test_example_cmd: read_npy b {:?}", b);

    // let w: Array2<f32> = read_npy("model_weights.npy").map_err(|_| Errors::ReadNpyError)?;
    // let _ = read_npy("model_weights.npy").map_err(|_| Errors::ReadNpyError)?;

    // let _ = read_npy("model_weights.npy")?.map_err(|err| Errors::from(err))?;

    /*
    let file = File::open("model_weights.npy");
    let array: ArrayD<f32> = ndarray_npy::read(file);
    let array2d = array.into_shape((2, 2)).unwrap();
    */
    /*
    let b_array: Array2<f32> = read_npy("model_bias.npy").map_err(|_| Errors::ReadNpyError);
    let b = b_array[0];
    */

    /*
        # injest test datasets from the NPY file
        test_set_x = np.load("test_set_x.npy")
        test_set_y = np.load("test_set_y.npy")

        # Load trained model from the NPY file
        w = np.load("model_weights.npy")
        b = np.load("model_bias.npy")[
            0
        ]  # convert a Python array with a single element to a scalar
    */

    Ok(())

    //let (_train_x, _train_y, _test_x, _test_y) = injest(digit);

    // let npy_data: NpyData<T> = read_npy("model_bias.npy")?; //?
    // let array: Array2<f32> = npy_data.into_array2()?; //? removed

    /*
    let file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
    */
    // println!("Saved bias {}", npy_data);

    /*
    let _ = write_npy("model_weights.npy", &_w);
    let _ = write_npy("model_bias.npy", &b_array);
    let _ = write_npy("test_set_x.npy", &_test_x);
    let _ = write_npy("test_set_y.npy", &_y_prediction_test);
    */

    /*
    # injest test datasets from the NPY file
    test_set_x = np.load("test_set_x.npy")
    test_set_y = np.load("test_set_y.npy")

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar
    */
}

fn predict_unseen_example_cmd(string_number: &str) -> Result<(), Errors> {
    println!("Predict digit from image file {}!", string_number);

    Ok(())
}

fn injest_cmd(string_number: &str) -> Result<(), Errors> {
    println!(
        "Preprocess datasets for recognising a given digit of {}!",
        string_number
    );
    // let digit: f32 = string_number.parse().unwrap();  // Avoid using unwrap, poor error handling
    let digit: f32 = parse_digit(string_number)?;
    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
    Ok(())
}

fn model_cmd(string_number: &str) -> Result<(), Errors> {
    /*
        Train model and save to NYP files
    */

    println!("Train model to classify a given digit {}!", string_number);

    let digit: f32 = parse_digit(string_number)?;

    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
    let _num_iterations = 5;
    let _learning_rate = 0.005;
    let print_cost = true;
    let _costs: Vec<f32> = Vec::new(); // Create an empty vector
                                       // avoid unwrap for robust error handling
                                       //let _y_prediction_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
                                       //let _y_prediction_train = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
                                       //let _w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let _b = 0.0;

    let _y_prediction_test = create_array(_b)?;
    let _y_prediction_train = create_array(_b)?;
    let _w = create_array(_b)?;

    let (costs, _y_prediction_test, _y_prediction_train, _w, _b, _learning_rate, _num_iterations) =
        model(
            &_train_x,
            &_train_y,
            &_test_x,
            &_test_y,
            _num_iterations,
            _learning_rate,
            print_cost,
        );

    let b_array = create_array(_b)?;

    // overwrite the file if it exists

    // let _ = write_npy("model_weights.npy", &_w)?; .map_err(|_| MyError::ParseFloatError)
    let _ = write_npy("model_weights.npy", &_w).map_err(|_| Errors::WriteNpyError); // .map_err(Errors::ParseFloatError);
    let _ = write_npy("model_bias.npy", &b_array).map_err(|_| Errors::WriteNpyError);
    //let _ = write_npy("model_bias.npy", &b_array)?;
    //let _ = write_npy("test_set_x.npy", &_test_x)?;
    //let _ = write_npy("test_set_y.npy", &_y_prediction_test)?;
    let _ = write_npy("test_set_x.npy", &_test_x).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("test_set_y.npy", &_y_prediction_test).map_err(|_| Errors::WriteNpyError);

    info!("main model_cmd: b {:?}.", b_array);
    info!("main model_cmd: w {:?}.", _w);
    info!("main model_cmd: cost {:?}.", costs);

    // find the index of elements in w equals target_number_f
    let target_value: f32 = 0.0;
    let first_row: Vec<f32> = _w.column(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_w = find_indices_filter(&first_row, &target_value); // search Vector of  Vec<f32>
    println!(
        "Found {} of {:?} times out of total {:?} in w ... {:?}",
        target_value,
        index3_w.len(),
        first_row.len(),
        index3_w
    );
    info!(
        "Found {} of {:?} times out of total {:?} in w ... {:?}",
        target_value,
        index3_w.len(),
        first_row.len(),
        index3_w
    );

    Ok(())
}

fn create_array(b: f32) -> Result<Array2<f32>, Errors> {
    /* let result = Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError);
    result */
    Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError)
}

fn parse_digit(string_number: &str) -> Result<f32, Errors> {
    string_number.parse().map_err(Errors::ParseFloatError)
}

/*
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
