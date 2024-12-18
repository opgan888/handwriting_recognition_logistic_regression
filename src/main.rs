mod data;
use fern::Dispatch;
use handwritingrecognition::data::find_indices_filter;
use handwritingrecognition::data::injest;
use handwritingrecognition::helper::model;
use handwritingrecognition::helper::predict;
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

use image::imageops;
use image::imageops::FilterType;
use image::DynamicImage;
use image::GenericImageView;
use image::ImageError;
use image::ImageReader;
use image::Pixel;

#[derive(Debug)]
enum Errors {
    ShapeError(ShapeError),
    ParseFloatError(ParseFloatError),
    ReadNpyError(ReadNpyError),
    WriteNpyError(WriteNpyError),
    ImageError(ImageError),
    IoError(std::io::Error),
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
       Predict the indexed test image from test dateset if belongs to trained number
    */
    println!(
        "Predict the indexed  {} test image from test dateset if belongs to trained number!",
        string_number
    );

    let digit: f32 = parse_digit(string_number)?;

    let test_set_x: Array2<f32> = read_npy("test_set_x.npy")?;
    let test_set_y: Array2<f32> = read_npy("test_set_y.npy")?;

    let w: Array2<f32> = read_npy("model_weights.npy")?;
    let b: Array2<f32> = read_npy("model_bias.npy")?;
    let test_set_x_example: Array2<f32> = test_set_x
        .column(digit as usize)
        .to_owned()
        .into_shape((test_set_x.shape()[0], 1))
        .map_err(Errors::ShapeError)?;

    info!("predict_test_example_cmd: read_npy b {:?}", b[(0, 0)]);
    info!(
        "predict_test_example_cmd: read_npy test_set_x {:?}",
        test_set_x.shape()
    );
    info!(
        "predict_test_example_cmd: read_npy test_set_y shape {:?}",
        test_set_y.shape()
    );
    info!(
        "predict_test_example_cmd: test_set_x_example shape {:?}",
        test_set_x_example.shape()
    );
    let a = "Actual is ".to_string() + &test_set_y[(0, digit as usize)].to_string();
    let p = "Prediction is ".to_string() + &predict(&w, b[(0, 0)], &test_set_x_example).to_string();
    info!("predict_test_example_cmd  {:?}  {:?} ", a, p);

    Ok(()) // Operation successful. () represents an empty tuple and signify the absence of a meaningful value.
}

fn predict_unseen_example_cmd(file_name: &str) -> Result<(), Errors> {
    println!("Predict image {}!", file_name);

    /*
    Arguments:
    file_name -- image file

    Returns:
    prediction results in debugging.log file for classifying the given image as the predefined digit or not.
    */

    // Load the parameters of Linear Regression model from the NPY file
    let w: Array2<f32> = read_npy("model_weights.npy")?;
    let b: Array2<f32> = read_npy("model_bias.npy")?;

    // Open image file, decode image, resize image
    let fname = format!("assets/images/{}", file_name);
    let img = image::open(&fname).map_err(Errors::ImageError)?;

    let num_px = 28; // unseen image must conform to the model image requirements of (num_px, num_px)

    // check if image size meet model requirement
    let (width, height) = img.dimensions();
    let mut need_resize = true;
    if width == num_px && height == num_px {
        need_resize = false;
    }

    info!(
        "debugging predict_unseen_example_cmd orginal image {:?} {:?} {:?}",
        need_resize, height, width
    );

    // Resize image using efficient `imageops::resize`
    let mut resized_image: DynamicImage = Default::default();

    if need_resize {
        resized_image =  image::DynamicImage::ImageRgba8(imageops::resize(
            &img,
            num_px,
            num_px,
            FilterType::Lanczos3, // Adjust filter type if desired
        ));
    }

    let (width, height) = resized_image.dimensions();

    
    info!(
        "debugging predict_unseen_example_cmd resized image {:?} {:?}",
         height, width
    );


    // Convert RGB image to Grayscale: gray scale is more green than red and blue
    // gray_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    // divide by 255 scale between 0 and 1

    // Create a 2D array to store pixel values ... flattened 
    let mut pixels: Vec<Vec<f32>> = vec![vec![0.0; 1]; (width * height) as usize];

    // Iterate over pixels and populate the array
    let mut index = 0;
    for y in 0..height {
        for x in 0..width {
            let pixel = resized_image.get_pixel(x, y);
            let channels = pixel.channels();
            pixels[index][0] = ( 0.2989 * channels[0] as f32
                + 0.5870 * channels[1] as f32
                + 0.1140 * channels[2] as f32) / 255.;
            index += 1;
        }
    }

    let rows = pixels.len(); // rows = 3
    let cols = pixels[0].len(); // cols = 3 

    info!(
        "debugging predict_unseen_example_cmd image flatten matrix of {:?} {:?} {:?} {:?} ",
        index, rows, cols, pixels[0][0]
    );

    // Convert Vec<Vec<f32>> to Array2
    let array2: Array2<f32> = Array2::from_shape_vec((rows, cols), pixels.into_iter().flatten().collect()).map_err(Errors::ShapeError)?;

    // Call Predict function
    let a = "Actual is ".to_string() + file_name;
    let p = "Prediction is ".to_string() + &predict(&w, b[(0, 0)], &array2).to_string();
    info!("predict_test_example_cmd  {:?}  {:?} ", a, p);

        /*

    // resized image to test_set_x_example
    // image = image / 255.0
    // image = image.reshape((1, num_px * num_px)).T

    // Call Predict function
    let a = "Actual is ".to_string() + file_name;
    let p = "Prediction is ".to_string() + &predict(&w, b[(0, 0)], &image).to_string();
    info!("predict_test_example_cmd  {:?}  {:?} ", a, p);

    */

    Ok(()) // Operation successful
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
    info!("Train model to classify a given digit {:?}!", string_number);

    let digit: f32 = parse_digit(string_number)?;

    let (_train_x, _train_y, _test_x, _test_y) = injest(digit);
    let _num_iterations = 2000;
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
    let _ = write_npy("model_weights.npy", &_w).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("model_bias.npy", &b_array).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("test_set_x.npy", &_test_x).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("test_set_y.npy", &_y_prediction_test).map_err(|_| Errors::WriteNpyError);

    info!("main model_cmd: b {:?}.", b_array);
    // info!("main model_cmd: w {:?}.", _w);
    // info!("main model_cmd w shape {:?}", _w.shape());
    info!("main model_cmd: cost {:?}.", costs);

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
