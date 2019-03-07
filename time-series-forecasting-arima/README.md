Name: Govardhan, Aditya Nagesh
ASU ID: 1215374199

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
a. the value of h (result of ADF test, 1 indicates stationary data, 0 indicates non-stationary data)
b. results of estimation of ARIMA model
c. forecasted values (20 values)

Following graphs are generated:
Fig 1. Plot of given data, its moving average and moving standard deviation
Fig 2. Plot of differentiated data, its moving average and moving standard deviation
Fig 3. ACF plot
Fig 4. PACF plot
Fig 5. Predicted values along with given data
(Fig 6. Forecasts for different p, d and q values UNCOMMENT THE CODE)

To generate the given vs estimated data, econometric modeler is used.