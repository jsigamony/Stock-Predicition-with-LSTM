# Stock Predicition on Amazon stocks data with Long-Short Term Memory (LSTM) model (PyTorch Code)
This project implements a Long Short-Term Memory (LSTM) neural network to predict Amazon (AMZN) stock prices. The model is trained on historical stock data and can be used to forecast future stock prices.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Structure](#code-structure)
6. [Model Architecture](#model-architecture)
7. [Results](#results)


## Overview

This project uses PyTorch to implement an LSTM model for time series forecasting. It includes data preprocessing, model training, and visualization of results. The main steps in the process are:

1. Loading and preprocessing the AMZN stock data
2. Splitting the data into training and testing sets
3. Scaling the data
4. Creating sequences for the LSTM model
5. Defining and training the LSTM model
6. Making predictions and visualizing results

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/amazon-stock-prediction.git
cd amazon-stock-prediction

2. Install the required packages:
pip install -r requirements.txt

## Usage

1. Ensure you have the AMZN.csv file in the same directory as the script.

2. Run the main script:
python stock_prediction.py

3. The script will train the model and display various plots showing the actual vs predicted stock prices.

## Code Structure

- `stock_prediction.py`: The main script containing all the code for data preprocessing, model definition, training, and evaluation.
- `AMZN.csv`: The dataset containing historical Amazon stock prices (not included in the repository, you need to download this separately).

## Model Architecture

The LSTM model used in this project has the following architecture:
- Input dimension: 1 (closing price)
- Hidden dimension: 8
- Number of LSTM layers: 1
- Output dimension: 1 (predicted closing price)

The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer.

## Results

The script generates several plots:
1. Closing values of the stock over time
2. Untrained model predictions vs actual prices
3. Trained model predictions vs actual prices

The final plot shows the comparison between the actual closing prices and the model's predictions, allowing you to visually assess the model's performance.
