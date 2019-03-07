%% LOAD DATA
data = xlsread("data.xlsx");
format shortG

%% CHECK STATIONARITY
mov_avg = movmean(data, [9 0]);
mov_dev = movstd(data, [9 0]);

figure(1)
plot(1:124, data, 1:124, mov_avg, 1:124, mov_dev)
legend('provided data', 'moving average', 'moving S.D.')
legend('Location','northwest')

%% TRANSFORM DATA, DETERMINE d VALUE
ddata = diff(data);

mov_avg = movmean(ddata, [9 0]);
mov_dev = movstd(ddata, [9 0]);

figure(2)
plot(1:123, ddata, 1:123, mov_avg, 1:123, mov_dev)
legend('first order difference data', 'moving average', 'moving S.D.')
legend('Location','northwest')

h = adftest(ddata)

%% DETERMINATION OF p AND q VALUES
figure(3)
autocorr(ddata, 50)

figure(4)
parcorr(ddata, 50)

%% MODEL CREATION AND FORECASTING
p = 4; % since pacf falls below confidence levels at 4
d = 1; % since first order of difference makes the data stationary
q = 2; % since acf falls below confidence levels at 2

Mdl = arima(p, d, q);
EstMdl = estimate(Mdl,data);

Y = forecast(EstMdl, 20, 'Y0', data)

figure(5)
plot(1:124, data, 'k', 125:144, Y, 'r')
legend('given', 'predicted')
legend('Location','northwest')

%{
%% FORECASTS FOR DIFFERENT P,D AND Q VALUES
Mdl1 = arima(4,0,0);
EstMdl1 = estimate(Mdl1, data);
Y1 = forecast(EstMdl1, 20, 'Y0', data)

Mdl2 = arima(4,0,2);
EstMdl2 = estimate(Mdl2, data);
Y2 = forecast(EstMdl2, 20, 'Y0', data)

Mdl3 = arima(10,1,5);
EstMdl3 = estimate(Mdl3, data);
Y3 = forecast(EstMdl3, 20, 'Y0', data)

figure(6)
plot(1:124, data, 'k', 125:144, Y, 'r', 125:144, Y1, 125:144, Y2, 125:144, Y3)
legend('given data', 'p = 4, d = 1, q = 2', 'p = 4, d = 0, q = 0', 'p = 4, d = 0, q = 2', 'p = 10, d = 1, q = 5')
legend('Location','northwest')
%}