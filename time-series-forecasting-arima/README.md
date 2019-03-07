Pre-requisites: Econometric Toolbox needs to be installed (You can also use online version of MATLAB for preinstalled setup, go to https://matlab.mathworks.com/)

Local Run:
==========
1. Open code.m MATLAB script
2. Go to Editor tab in taskbar
3. Click Run

Cloud Run: (https://matlab.mathworks.com/)
==========
1. Copy code.m MATLAB script to a new script
2. Save it
3. Go to Editor tab in taskbar
4. Click Run

Expected Results:
=================
The command line shows:
1. a. the value of h (result of ADF test, 1 indicates stationary data, 0 indicates non-stationary data)
2. b. results of estimation of ARIMA model
3. c. forecasted values (20 values)

Following graphs are generated:
1. Fig 1. Plot of given data, its moving average and moving standard deviation
2. Fig 2. Plot of differentiated data, its moving average and moving standard deviation
3. Fig 3. ACF plot
4. Fig 4. PACF plot
5. Fig 5. Predicted values along with given data
6. (Fig 6. Forecasts for different p, d and q values UNCOMMENT THE CODE)

To generate the given vs estimated data, econometric modeler is used.
