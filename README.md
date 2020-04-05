

**Summary**: As an extension of Buchner, Kaserer and Wagner’s paper on stochastic modeling of private equity, this research aims to build a functioning LSTM model to predict distribution and contribution cash flows of three types of Private Equity funds, specifically Buyout Funds (all, large, mid, and small-sized), Venture Capital Funds, and Real Estate Funds. First, quarterly historical benchmark data from 2014 to 2019 are obtained from Preqin and interpolated using stochastic interpolation. Next, we built an LSTM model using Pytorch and tuned hyperparameters for each fund type and size, experimenting with different architectures and hyperparameters. 

The program that has been written for this project consists of three files; 'PE_Data.py' for reading in, interpolating and formatting the data, 'PE_LSTM_V5.py' for the purpose of training and testing the LSTM models on said data, and 'PE_MRSR.py' for fitting and simulating diffusion processes on the same data.

1. **PE_Data.py** consists of three functions: get_dat, interpolate_gaussian. and prep_dat. 
- The get_dat function takes historical cash flow data from Preqin in a specific format, with of a field for the cash flow type (Called Up or Distributed) and the number of time steps between quarters to be used in the interpolation of the data.  This program reads in the cashflow for funds within a given category of all different sizes data into pandas data frames and interpolating each data set using the interpolate_gaussian function, using different parameters for the noise term of the training and test data sets. The training and test set are the first 70 and the last 30 percent of the time series provided. The function  returns data frames of time series for each fund size, with one column corresponding to each step in the series. 
- The second function, interpolate_gaussian, accepts a time series of data, the number of time steps between given points to interpolate to, and a factor by which to scale the variance of the noise term in the interpolation. The function works by interpolating the data linearly, but adding a zero mean gaussian noise term at every time step, to simulate a random walk with drift between given observations. The function returns a pandas data frame of the interpolated data. 
- Finally, the prep_dat function takes a time series of data, the number of historical observations to use as inputs and the number of time steps ahead that will be considered the response variable as arguments. The function reshapes the data into a pandas data frame with rows that have observed data for the given historical observation sequence length, as well as the response variable. Each row is shifted one time step ahead from the previous row. The function returns said data frame.

2. **PE_LSTM_V5.py** begins by setting hyper parameters for the LSTM models such as the learning rate, input sequence length and number of epochs and calling the functions in the PE_Data.py file to format the desired data. It then consists of a class for the LSTM model, with constructor taking arguments for the input dimension (1 for our purposes since time series observations are fed in one at a time), sequence length, batch size output dimension (1 since we are only predicting one observation at a time) and number of layers. The class also has functions for the initialization of the hidden layer and for forward passes of data through the model. The file also contains a function to train the model, trainModel, which takes an instance of the aforementioned class, the training input dataset, the training response dataset and the epoch number. It then trains an LSTM model for the given data using batch optimization (with normalization) and an optimization routine that is declared outside of the function. Next, the file has the testModel function, which takes a trained instance of the LSTM class, the input test data and the response test data. It passes the test data through the LSTM model by batches, records the test loss and plots the predicted data versus the actual data for the test set. Finally, this file initiates an instance of the LSTM class, declares the loss function (mean square error by default) and optimization routine and calls the functions for training and testing the model.

3. **PE_MRSR.py** reads in and prepares the desired cash flow data using the functions from the 'PE_Data.py' file. It then contains code to fit parameters based on the training set of data (again first 70 percent of time series), according to least squares estimation, for Ornstein-Uhlenbeck and mean reverting square root diffusion processes. The file then contains a loop to simulate and store future values of the cash flows, attempting to predict the values in the test set, according to the aforementioned diffusion processes, using an Euler-Maruyama discretization, with the number of simulation iterations specified as a parameter. Taking the average predicted values over all of the iterations in typical Monte Carlo simulation fashion, the program terminates by plotting the predicted values and calculating the MSE of the predicted values in comparison to the actual values of the test dataset.
