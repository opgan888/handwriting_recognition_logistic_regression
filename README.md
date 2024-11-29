[![Clippy Rust Lint with Github Actions](https://github.com/sktan888/handwriting_recognition_logistic_regression/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/handwriting_recognition_logistic_regression/actions/workflows/main.yml)

# handwriting_recognition_logistic_regression
This is RUST implementation of logistic regression of handwriting recognition


## Set up working environment
* create virtual environment: ```virtualenv ENV```
    - remove directory: ``` rm -r hENV```
* add ```source ENV/bin/activate``` in .bashrc file
    edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* create Makefile for make utility : ``` touch Makefile ```
    - rename a file: ```mv oldfilename newfilename```
* create Requirements.txt for package install : ``` touch Requirement.txt ```
    - find out package version ```pip freeze | less```
    - create library folder: ``` mkdir myLib ```



## Neural Network in python
* Injest in data.py ``` train_data, test_data = keras.datasets.mnist.load_data() ```

    The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits. (https://yann.lecun.com/exdb/mnist/)
    - Download (https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
    - Consisting of 70,000 28x28 black-and-white images of handwritten digits
    - 10 classes 0 to 9: one class for each  digit
    - 60,000 images in the training dataset, 10,000 images in the validation dataset
    - 7,000 images (6,000 train images and 1,000 test images) for each digit/class
    - Non binary classification of MNIST for 0 to 9, would require 10 output neurons to classify all 10 digits
    - Binary classification NN simplies to telling if a handwriting is the trained digit
    ``` 
    train_set_y_binary = np.zeros((1, train_set_y.size))
    classN = digit  # digit to classify
    index = np.where(
        train_set_y == classN
    )  # index of (elements in train_set_y equals classN)
    train_set_y_binary[0, index[1]] = 1
    ```
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
```
import numpy as np

# Save the array to an NPY file in modelling.py
# save model to an NPY file
np.save('model_weights.npy', logistic_regression_model["w"])
np.save('model_bias.npy', np.array([logistic_regression_model["b"]]) )

# save datasets to an NPY file
np.save('test_set_x.npy', test_set_x)
np.save('test_set_y.npy', test_set_y)

# Load the array from the NPY file in predicting.py
# injest test datasets from the NPY file
test_set_x = np.load('test_set_x.npy')
test_set_y = np.load('test_set_y.npy')

# Load trained model from the NPY file
w = np.load('model_weights.npy')
b = np.load('model_bias.npy')[0] # convert a Python array with a single element to a scalar
```

* Command Line Interface (CLI)
    - ``` python main.py predict-test 55 ``` 55 refers to the example index of test dataset
```
## main.py

#!/usr/bin/env python3
"""The env command allows users to display the current environment or run a specified command in a changed environment."""

import click

@click.group()
def cli():
    """run NN modelling and prediction"""


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

if __name__ == "__main__":
cli()
```

* Test
```
#test_hello.py
from hello import more_hello, more_goodbye, add
def test_more_hello():
    assert "hi" == more_hello()
```
## Neural Network in RUST
* Injest in data.rs ```  let (x_train, y_train, x_test, y_test) = mnist.load_mnist(None).expect("Error loading MNIST");```
* download MNIST datasets
* The MNIST data set is a collection of four gzip files and can be found [here] (http://yann.lecun.com/exdb/mnist/). 
* There is one file for each of the following: the training set images, the training set labels, the test set images, and the test set labels.
* The four files must be downloaded and extracted. By default, they will be looked for in a “data” directory at the top of level of your crate.

``` cargo run main 2 ```