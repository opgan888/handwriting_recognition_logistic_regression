use handwritingrecognition::helper::sigmoid;

#[test]
fn test_sigmoid() {
    assert_eq!(sigmoid(0.0), 0.5);
}

/*
#[cfg(test)]


mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
    }
}
*/
