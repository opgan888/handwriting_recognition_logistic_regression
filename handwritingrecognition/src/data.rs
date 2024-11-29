extern crate mnist;
use mnist::*;
use ndarray::{Array2, Array3}; //, s}; //, Zip};

// arr: &[T] represents a slice, which is a reference to a contiguous sequence of elements of type T (type T is a declaration of a type alias)
// target: &T is a function parameter declaration that signifies a reference to a value of type T
// usize is unsigned integer type
fn find_indices_filter<T: PartialEq>(arr: &[T], target: &T) -> Vec<usize> {
    arr.iter()
        .enumerate()
        .filter(|(_, x)| **x == *target)
        .map(|(i, _)| i)
        .collect()
}

// Loading data for handwriting pub fn injest(digit: i32) {

pub fn injest(digit: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    //pub fn injest(digit: f32) -> ([[f32; 60000]; 784], [[f32; 60000]; 1] , [[f32; 10000]; 784], [[f32; 10000]; 1] ){
    // The default path to the MNIST data files is /data of top layer crate

    let m_train = 60_000;
    let m_test = 10_000;
    let _num_px = 28;

    // Download and Load the MNIST dataset
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(m_train)
        .test_set_length(m_test)
        .finalize();

    let _image_num = 0;

    let train_data = Array3::from_shape_vec((60_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((60_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    // Print the shape of the training data
    println!("Shape of train_data: {:?}", train_data.shape());
    println!("Shape of test_data: {:?}", test_data.shape());

    // # train set y and test_set_y are originally row vector (m, 1) array. Reshape to column vector (1,m) array
    println!("Shape of train_labels: {:?}", train_labels.shape());
    println!("Shape of test_labels: {:?}", test_labels.shape());
    // println!("Element of train_labels: {:?}", train_labels[(0, 0)]);

    // In handwriting dataset, y is digits 0 to 9 and requires 10 output neurons to classify all 10 digits
    // Since this is single output NN, consider classifying one digit at one time for now
    // train_set_y zeros for non N and ones for N
    //    train_set_y_binary = np.zeros((1, train_set_y.size))
    //    test_set_y_binary = np.zeros((1, test_set_y.size))

    // let train_labels_binary: Array2<i32> = Array2::zeros((60000, 1));
    let mut train_labels_binary: Array2<i32> = Array2::zeros((1, 60000));
    //assert_eq!(train_set_y_binary.shape(), &[60000, 1]);
    assert_eq!(train_labels_binary.shape(), &[1, 60000]);

    // classifying one digit at one time
    let target_number_f = &digit; // &6_f32; // explicitly specifies the type of the number as a 32-bit floating-point number
    println!("Target number to be classified {}: ", target_number_f);

    // find the index of elements in train_set_y equals target_number_f
    let first_column: Vec<f32> = train_labels.column(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_train_labels = find_indices_filter(&first_column, target_number_f); // search Vector of  Vec<f32>
                                                                                   //println!("Found {} at index {:?} in first_column", target_number_f, index3_col.len());

    // Using iterators and map to modify elements
    // index3_col.iter().enumerate().for_each(|(i, x)| {
    // index3_col.iter().enumerate().for_each(|(_, x)| {
    let mut index: usize = 0;
    index3_train_labels.iter().for_each(|x| {
        //println!("index {:?} value {:?}", i, x);
        //let index: usize = {*x};
        index = *x;
        train_labels_binary[[0, index]] = 1
    });

    //Searching for a Value in a 2D Array
    let target_value: i32 = 1;
    let mut found = train_labels_binary.iter().any(|&x| x == target_value);
    if found {
        //println!("Found the target value: {}", target_value);
    } else {
        //println!("Target value not found.");
    }
    // find the index of elements in train_set_y equals target_number_f
    let first_row: Vec<i32> = train_labels_binary.row(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_train_labels_binary = find_indices_filter(&first_row, &target_value); // search Vector of  Vec<f32>
    println!(
        "Found {} of {:?} times in train_labels",
        target_number_f,
        index3_train_labels_binary.len()
    );
    assert_eq!(index3_train_labels_binary.len(), index3_train_labels.len());

    //
    // let test_labels_binary: Array2<i32> = Array2::zeros((10000, 1));
    let mut test_labels_binary: Array2<i32> = Array2::zeros((1, 10000));
    //assert_eq!(train_set_y_binary.shape(), &[60000, 1]);
    assert_eq!(test_labels_binary.shape(), &[1, 10000]);

    // classifying one digit at one time
    // let target_number_f = &3_f32; // explicitly specifies the type of the number as a 32-bit floating-point number
    // find the index of elements in train_set_y equals target_number_f
    let first_column_test: Vec<f32> = test_labels.column(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_test_labels = find_indices_filter(&first_column_test, target_number_f); // search Vector of  Vec<f32>
                                                                                       //println!("Found {} at index {:?} in first_column", target_number_f, index3_col_test.len());

    // Using iterators and map to modify elements
    // index3_col.iter().enumerate().for_each(|(i, x)| {
    // index3_col.iter().enumerate().for_each(|(_, x)| {
    // let mut index: usize = 0;
    index3_test_labels.iter().for_each(|x| {
        //println!("index {:?} value {:?}", i, x);
        //let index: usize = {*x};
        index = *x;
        test_labels_binary[[0, index]] = 1
    });

    //Searching for a Value in a 2D Array
    //target_value = 1;
    found = test_labels_binary.iter().any(|&x| x == target_value);
    if found {
        //println!("Found the target value: {}", target_value);
    } else {
        //println!("Target value not found.");
    }
    // find the index of elements in train_set_y equals target_number_f
    let first_row_test: Vec<i32> = test_labels_binary.row(0).iter().cloned().collect(); // Extract the first column of 2D Array
    let index3_test_labels_binary = find_indices_filter(&first_row_test, &target_value); // search Vector of  Vec<f32>
    println!(
        "Found {} of {:?} times in test_labels",
        target_number_f,
        index3_test_labels_binary.len()
    );
    assert_eq!(index3_test_labels_binary.len(), index3_test_labels.len());

    /*
    # Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    */

    let m_train = train_data.shape()[0]; // [60000, 28, 28]
    let m_test = test_data.shape()[0]; // [10_000, 28, 28]
    let num_px2 = test_data.shape()[1]; // assume image is square

    println!("Number of training examples: m_train = {:?}", m_train);
    println!("Number of testing examples: m_test = {:?} ", m_test);
    println!("Height/Width of each image: num_px = {:?}", num_px2);
    println!("Each image is of size: {:?} {:?}", num_px2, num_px2);

    /*
    println!("Shape of train_data: {:?}", train_data.shape());
    println!("Shape of test_data: {:?}", test_data.shape());
    println!("Shape of train_labels: {:?}", train_labels.shape());
    println!("Shape of test_labels: {:?}", test_labels.shape());
    */

    // Reshape the ArrayView1 into a 2D array
    let _shape = (1, m_train); // Reshape into a (1,m) matrix
    let train_labels_colvector = train_labels.into_shape_with_order((1, 60_000)).unwrap();
    println!(
        "Flattened train_labels shape: {:?}",
        train_labels_colvector.shape()
    );

    let test_labels_colvector = test_labels.into_shape_with_order((1, 10_000)).unwrap();
    println!(
        "Flattened test_labels shape: {:?}",
        test_labels_colvector.shape()
    );

    // Flatten train dataset from array(60000, 28, 28) into (784,60000) and test dataset from (10000,28,28) to (748, 10000)

    assert_eq!(train_data.shape(), &[60000, 28, 28]);
    // Reshape the array into a 2D array with shape (60000, 784)
    let reshaped_data = train_data.to_shape((60000, 784)).unwrap();
    // Transpose the reshaped array to get the desired shape (784, 60000)
    let flattened_train_data = reshaped_data.t();
    println!("Flattened train shape: {:?}", flattened_train_data.shape());

    let reshaped_data = test_data.to_shape((10000, 784)).unwrap();
    // Transpose the reshaped array to get the desired shape (784, 10000)
    let flattened_test_data = reshaped_data.t();
    println!("Flattened test shape: {:?}", flattened_test_data.shape());

    // standardize dataset above
    //println!("Flattened standardized test: {:?}", flattened_test_data);
    found = flattened_test_data.iter().any(|&x| x > 0.5);
    if found {
        //println!("Found the target value > {}", 0.0);
    } else {
        //println!("Target value not found.");
    }

    //return train_set_x, train_set_y, test_set_x, test_set_y
    let owned_flattened_train_data = flattened_train_data.to_owned(); // OwnedRepr: Represents an array that owns its data. You can modify the elements of this array directly.
    let owned_flattened_test_data = flattened_test_data.to_owned();
    (
        owned_flattened_train_data,
        train_labels_colvector,
        owned_flattened_test_data,
        test_labels_colvector,
    )

    /*
    let dims = flattened_train_data.dim(); // This is of type `ndarray::Dim<[usize; 3]>`
    //let (_row, _col) = dims.into(); // Extract dimensions as tuple
    println!("Flattened train dim: {:?}", dims);

    // Create a 2D array
    let array: Array2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("Shape of  2D array: {:?}", array.shape());

    let (flattened_vec, _) = array.into_raw_vec_and_offset();
    println!("Shape of flattened_vec: {:?}", flattened_vec.len());

    let _nested_array: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];

    let vec3d: Vec<Vec<Vec<i32>>> = vec![
        vec![vec![1, 2, 3], vec![4, 5, 6]],
        vec![vec![7, 8, 9], vec![10, 11, 12]],
    ];
    let element = vec3d[1][0][2]; // Accesses the element at (1, 0, 2)

    println!("element: {:?}", element);


    let a: Array3<i32> = Array3::zeros((8, 5, 2));

    println!("Shape of array3d: {:?}", a.shape());

    let _array_dd: Array3<i32> = Array3::zeros((2, 3, 4)); // Create a 2x3x4 array filled with zeros

    let a = arr3(&[
        [
        [ 1,  2,  3],     // -- 2 rows  \_
        [ 4,  5,  6]],    // --         /
        [
        [ 7,  8,  9],     //            \_ 2 submatrices
        [10, 11, 12]
        ]
        ]);  //            /
        //  3 columns ..../.../.../


    let b = arr3(&[
        [
        [ 1,  2,  3],     // -- 2 rows  \_
        [ 4,  5,  6]
        ]
        ]);  //            /
        //  3 columns ..../.../.../

    let c = arr2(&[
        [ 1,  2,  3],
        [ 4,  5,  6]
        ]);


    assert_eq!(a.shape(), &[2, 2, 3]);
    assert_eq!(b.shape(), &[1, 2, 3]);
    assert_eq!(c.shape(), &[2, 3]);
    println!("Shape of array3d: {:?}", a.shape());
    println!("Shape of array2d: {:?}", b.shape());
    println!("Shape of array2d: {:?}", c.shape());

        // Access the dimension information
        /*
        let dim = arr.dim(); // This is of type `ndarray::Dim<[usize; 3]>`
        let (x, y, z) = dim.into_tuple(); // Extract dimensions as tuple
        */
    */

    /*
    //let _array = train_set_y_binary;
    let target_number = &3; // a reference to the integer, efficient memory management and preventing unnecessary copying
    let target_number_f = &3_f32; // explicitly specifies the type of the number as a 32-bit floating-point number

    let numbers = [1, 2, 3, 4, 5];  // Fixed Size: Once created, a list's size cannot be changed. allocated on the stack
    let numbers_v = vec![1, 2, 3, 4, 5]; // vector: size cannot be changed. grow and shrink dynamically. allocated on the heap
    let numbers_f: Vec<f32> = numbers_v.iter().map(|&x| x as f32).collect(); // convert a Vec<i32> to Vec<f32> using the iter() method, the map() method and the collect() method.
    println!(" Vec<f32> {:?}", numbers_f);

    let index2 = find_indices_filter(&numbers, target_number); // search List
    println!("Found {} at index {:?} in List", target_number, index2); //  :? is formatting specifier used with the println! macro
    let index2 = find_indices_filter(&numbers_v, target_number); // search Vector of Vec<i32>
    println!("Found {} at index {:?} in Vector<i32>", target_number, index2);
    let index2 = find_indices_filter(&numbers_f, target_number_f); // search Vector of  Vec<f32>
    println!("Found {} at index {:?} in Vector<f32>", target_number, index2);

    */

    // train_labels
    //let submatrix2 = train_labels.slice(s![0..10, 0..1]);
    //println!("Submatrix 2: {:?}", submatrix2);

    /*
    let arr: [i32; 10000] = [0; 10000]; // Create an array of 10000 zeros

    // Slice the first 10 elements
    let first_10 = &arr[0..10];
    println!("size of slice: {:?}", first_10.len());

    let arr = Array2::from_shape_fn((3, 3), |(i, j)| {
        (i * 3 + j + 1) as i32
    });
    // Slice the first row
    let first_row = &arr.row(0);

    // Slice the second column
    let second_column: Vec<i32> = arr.column(1).iter().cloned().collect();

    // Slice a submatrix
    let submatrix = arr.slice(s![1..3, 1..3]);

    println!("First row: {:?}", first_row);
    println!("Second column: {:?}", second_column);
    println!("Submatrix: {:?}", submatrix.shape());
    */
}
