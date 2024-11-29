mod data;
use handwritingrecognition::data::injest;
use handwritingrecognition::helper::sigmoid;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("Logistic Regression classification of handwriting digits");
    let string_number = &args[2];
    println!("Recognise  {}!", string_number);
    let digit: f32 = string_number.parse().unwrap();
    let (_train_x, _train_y, _test_x, _test_y) = injest(digit); //9.0);
}
