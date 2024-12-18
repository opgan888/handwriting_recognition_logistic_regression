[![Clippy Rust Lint with Github Actions](https://github.com/sktan888/handwriting_recognition_logistic_regression/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/handwriting_recognition_logistic_regression/actions/workflows/main.yml)

# handwriting_recognition_logistic_regression
* This is RUST implementation of logistic regression for handwritten single digit recognition
* i.e classify if a handwriting matches a given digit or not
* Recognition of 10 digits handwriting from 0 to 9 would require modeling of LR for each of 10 digits. 


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
* add new project ```Cargo add project``` ``` tree.```
``` 
.
├── Cargo.lock
├── Cargo.toml
├── Makefile
├── README.md
├── assets
│   └── images
│       ├── digitHW.png
│       ├── hw.png
│       ├── lr.webp
│       ├── nine.jpeg
│       ├── nn.png
│       └── sigmoid.png
├── data
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── src
│   ├── data.rs
│   ├── helper.rs
│   ├── lib.rs
│   └── main.rs
└── tests
    └── test_helper.rs
```
* add dependencies in Cargo.toml 
```
[dependencies]
itertools = "0.13.0"
mnist = "0.6.0"
ndarray = "0.16.1"
statrs = "0.17.1"
```
## Command Line Interface
- Saving trained model parameters (i.e. w and b) in files
- Run main injest command: ``` cargo run main injest 4 ``` 4 refers to the number for recognising 
- Run main modelingcommand: ``` cargo run main modeling 4 ``` 4 refers to the number for recognising 
- Run main predict test example command: ``` cargo run main predict_test_example 4 ``` 4 refers to the index of test example datasets for recognising 
- Run main command: ``` cargo run main predict_unseen_example image.jpeg ``` image.jpeg refers to the image file name for recognising ... completed
- Run test command: ``` cargo test ```
- Run release  ``` ./target/release/handwritingrecognition main modeling 9```

## Machine Learning
* Injest 
    The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits. (https://yann.lecun.com/exdb/mnist/)
    - Download (https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
    - Consisting of 70,000 28x28 black-and-white images of handwritten digits
    - 10 classes or labels from 0 to 9: one class/label for each  digit
    - 60,000 images in the training dataset, 10,000 images in the validation dataset
    - 7,000 images (6,000 train images and 1,000 test images) for each digit/class

* EDA
    - Each image is represented as an array of pixels of shape=(28, 28, 1), dtype=uint8
    - Each pixel is between 0 and 255 
    - ![Handwriting](/assets/images/digitHW.png)

* Logistic Regression
    - Classify into 2 classes
        - ![LogisticRegression](/assets/images/lr.webp)
    - Similar is same as one layer NN with one node 
        - ![NN](/assets/images/nn.png)
    - Sigmoid
        - ![Sigmoid](/assets/images/sigmoid.png)
) 

* Conclusion

