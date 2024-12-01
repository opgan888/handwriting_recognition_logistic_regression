mod data;
use fern::Dispatch;
use handwritingrecognition::data::injest;
use handwritingrecognition::helper::model;
use handwritingrecognition::helper::sigmoid;
use log::LevelFilter;
use log::{debug, info};
use ndarray::{arr2, Array2};
use std::env;
use npy::NpyData;
use ndarray_npy::read_npy;
use ndarray_npy::write_npy;
use ndarray_npy::WriteNpyExt;

use ndarray::ErrorKind;
use std::error::Error;
use std::num::ParseFloatError;
use ndarray::ShapeError;

enum Errors {
    ShapeError(ShapeError),
    ParseFloatError(ParseFloatError),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}] [{}] {}",
                record.level(),
                record.target(),
                message
            ))
        })
        .level(LevelFilter::Debug)
        .chain(fern::log_file("debugging.log")?)
        .apply()
        .unwrap();

    let args: Vec<String> = env::args().collect();
    println!("Logistic Regression classification of handwriting digits");

    // injest DIGIT, modeling DIGIT, predict_test_example INDEX, predict_unseen_example IMAGE_FILE
    let cmd: &str = &args[2]; // injest, modeling, predict_test_example, predict_unseen_example commands
                              //let s_string: String = s.to_owned();
    let param: &str = &args[3]; // a string rep digit, index, file name
    match cmd {
        "injest" => {
            injest_cmd(param);
            Ok(())
        }
        "modeling" => {
            let _ = model_cmd(param);
            Ok(())
        }
        "predict_test_example" => {
            predict_test_example_cmd(param);
            Ok(())
        }
        "predict_unseen_example" => {
            predict_unseen_example_cmd(param);
            Ok(())
        }
        _ => {
            println!(
                "Enter a command: injest, modeling, predict_test_example, predict_unseen_example"
            );
            Ok(())
        }
    }
}

fn predict_test_example_cmd(string_number: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Predict a digit from index of test dateset {}!",
        string_number
    );
    /*

            fn parse_and_process(string_number: &str) -> Result<(), ()> {
            let num: i32 = string_number.parse()?;
            // Process the parsed number
            println!("Parsed number: {}", num);
            Ok(())
        }

    */
    let digit: f32 = string_number.parse().unwrap();
    
    //let (_train_x, _train_y, _test_x, _test_y) = injest(digit);

    // let npy_data: NpyData<T> = read_npy("model_bias.npy")?; //?
    // let array: Array2<f32> = npy_data.into_array2()?; //? removed

    /*

                fn divide(x: i32, y: i32) -> Result<i32, String> {
                if y == 0 {
                    return Err("Division by zero".to_string());
                }
                Ok(x / y)
            }
                /// 
            let result = divide(10, 2);
            match result {
                Ok(value) => println!("Result: {}", value),
                Err(err)   
        => println!("Error: {}", err),
    }
    */
    /*
    let file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
    */
    // println!("Saved bias {}", npy_data);

    Ok(())

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

//fn model_cmd(string_number: &str) -> std::io::Result<()> {
// fn model_cmd(string_number: &str) -> Result<(), ()> { T, E
fn model_cmd(string_number: &str) -> Result<(), Errors> { 
    println!("Model trained to classify a digit of {}!", string_number);
    // let digit: f32 = string_number.parse().unwrap()?;
    // let digit: f32 = string_number.parse()?;
    
    let digit: f32 = parse_digit(string_number)?;

    /*
    let digit_result: f32 = match string_number.parse(){
        Ok(digit) => digit,
        Err(e) => return Err(e),
    };
    */

    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
    let _num_iterations = 5;
    let _learning_rate = 0.005;
    let print_cost = true;
    let _costs: Vec<f32> = Vec::new(); // Create an empty vector
    let _y_prediction_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let _y_prediction_train = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let _w = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let _b = 0.0;
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

    // println!("Costs are {:?}!", costs);

    // save models and test datasets to a file: model_bias.npy model_weights.npy test_set_x.npy test_set_y.npy
    // let b_array = Array2::from_shape_vec((1, 1), vec![_b]).unwrap();
    // let b_array = Array2::from_shape_vec((1, 1), vec![_b])?; 

    let b_array = create_array(_b)?;

    /*
    let b_array = match Array2::from_shape_vec((1, 1), vec![_b]) {
        Ok(array) => array,
        Err(err) => Err(err),
        /*
        Err(err) => {
            // Handle the error here, e.g., log an error message or return a default value
            panic!("Error creating array: {}", err);
        } */
    };
    */

    let _ = write_npy("model_weights.npy", &_w);
    let _ = write_npy("model_bias.npy", &b_array);
    let _ = write_npy("test_set_x.npy", &_test_x);
    let _ = write_npy("test_set_y.npy", &_y_prediction_test);

    info!("main model_cmd: b {:?}.", b_array);
    info!("main model_cmd: w {:?}.", _w);
    info!("main model_cmd: cost {:?}.", costs);

    Ok(())
}

fn create_array(b: f32) -> Result<Array2<f32>, Errors> {
    let result = Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError);
    result
}

fn parse_digit(string_number: &str) -> Result<f32, Errors> {
    let digit_result: Result<f32, ParseFloatError> = string_number.parse();
    match digit_result {
        Ok(digit) => Ok(digit),
        Err(e) => Err(Errors::ParseFloatError(e)),
    }
}
/*
let result = create_array(b);
match result {
    Ok(()) => println!("Array created successfully"),
    Err(err) => match err {
        MyError::ShapeError(e) => eprintln!("Shape error: {}", e),
        MyError::ParseFloatError(e) => eprintln!("Parse float error: {}", e),
    }
}
*/
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
