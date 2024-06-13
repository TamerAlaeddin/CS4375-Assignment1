# CS4375 Assignment 1

## Description

This repository contains the implementation for CS4375 Assignment 1. The assignment consists of two parts: 
1. Linear Regression using Gradient Descent.
2. Linear Regression using a Machine Learning Library (Scikit-learn).

The dataset used is the "Auto MPG" dataset from the UCI ML Repository, hosted on GitHub.

## Part 1: Linear Regression using Gradient Descent

### How to Run

1. Ensure you have Python 3 installed.
2. Clone the repository:
    ```bash
    git clone https://github.com/TamerAlaeddin/CS4375-Assignment1.git
    cd CS4375-Assignment1
    ```
3. Create a virtual environment and activate it:
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```
4. Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
5. Run the script:
    ```bash
    python part1.py
    ```

### Output

The script will print the Mean Squared Error (MSE) for the test dataset and plot the cost function during gradient descent. The cost history for each iteration will be logged in `log.txt`.

## Part 2: Linear Regression using ML Library

### How to Run

1. Ensure you have Python 3 installed.
2. Clone the repository:
    ```bash
    git clone https://github.com/TamerAlaeddin/CS4375-Assignment1.git
    cd CS4375-Assignment1
    ```
3. Create a virtual environment and activate it:
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```
4. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn
    ```
5. Run the script:
    ```bash
    python part2.py
    ```

### Output

The script will print the Mean Squared Error (MSE) for both the training and test datasets.

## Dataset

The dataset used in this assignment is the "Auto MPG" dataset from the UCI ML Repository. It is hosted on GitHub and can be accessed at the following URL:
[Auto MPG dataset](https://raw.githubusercontent.com/TamerAlaeddin/CS4375-Assignment1/main/auto-mpg.data)

## Additional Information

### Pre-processing Steps

1. Removed null or NA values.
2. Converted categorical variables to numerical variables.
3. Dropped the `car name` column as it is not suitable for regression.
4. Converted all columns to numeric types.

### Training and Test Split

The dataset is split into training and test parts with an 80/20 ratio.

### Parameters for Gradient Descent (Part 1)

- Learning Rate: 0.001
- Iterations: 1000

### Logging

For Part 1, a log file (`log.txt`) is generated that records the cost for each iteration during gradient descent.

### Evaluation Metrics

- Mean Squared Error (MSE) for both training and test datasets.
- The script also plots the cost function during gradient descent for Part 1.

### Hosting Data on GitHub

The dataset is hosted on GitHub to ensure that the scripts can fetch the data directly without needing a local copy. This makes the scripts more portable and easier to run on any machine.

## References

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Free File Hosting on GitHub](https://www.labnol.org/internet/free-file-hosting-github/29092/)

## Contributors

- Tamer Alaeddin
