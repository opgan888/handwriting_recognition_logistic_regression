[![Clippy Rust Lint with Github Actions](https://github.com/sktan888/handwriting_recognition_logistic_regression/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/handwriting_recognition_logistic_regression/actions/workflows/main.yml)

# handwriting_recognition_logistic_regression
This is RUST implementation of logistic regression of handwriting recognition


## Set up working environment
* install RUST: ```curl https://sh.rustup.rs -sSf | sh```
* restart current shell  ``` . "$HOME/.cargo/env"  ``` in .bashrc file
* check version ``` rustc --version ```
* create Makefile for make utility : ``` touch Makefile ```
``` 
rust-version:
	rustc --version
format:
	cargo fmt --quiet
lint:
	cargo clippy --quiet
test:
	cargo test --quiet
run:
	cargo run
release:
	cargo build --release
all: format lint test run
```
* add new project ```Cargo add project```
* add dependencies in Cargo.toml 
```
[dependencies]
itertools = "0.13.0"
mnist = "0.6.0"
ndarray = "0.16.1"
statrs = "0.17.1"
```


## Neural Network in python
* Injest in data.py ``` pub fn injest(digit: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {} ```

    The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits. (https://yann.lecun.com/exdb/mnist/)
    - Download (https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
    - Consisting of 70,000 28x28 black-and-white images of handwritten digits
    - 10 classes 0 to 9: one class for each  digit
    - 60,000 images in the training dataset, 10,000 images in the validation dataset
    - 7,000 images (6,000 train images and 1,000 test images) for each digit/class
    - Non binary classification of MNIST for 0 to 9, would require 10 output neurons to classify all 10 digits
    - Binary classification NN simplies to telling if a handwriting is the trained digit

* EDA
    - Each image is represented as an array of pixels of shape=(28, 28, 1), dtype=uint8
    - Each pixel is an integer between 0 and 255 
    - Label of the image is the numerical digit
    - Visualization:
        - ![Handwriting](/assets/images/digitHW.png)

* Modelling
    - NN with input layer (as many nodes X features), a hidden layer (one node) and an output layer (one node for Binary output)
        - ![NN](/assets/images/nn.png)
    - Sigmoid
        - ![Sigmoid](/assets/images/sigmoid.png)
    - Logistic Regression
        - ![LogisticRegression](/assets/images/lr.webp) 
* Conclusion

* Saving trained NN parameters (i.e. w and b) in NPY Files
    - NPY files are a binary file format used to store NumPy arrays efficiently storing large arrays and loading back

* Command Line: ``` cargo run main 4 ``` 4 refers to the number for recognising 

* Test: ``` cargo test ```
