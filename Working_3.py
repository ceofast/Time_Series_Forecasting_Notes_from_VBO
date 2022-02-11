###########################
# Time Series Forecasting #
###########################

# Introduction to Time Series and Basic Concepts

# Smoothing Methods
#   Single Exponential Smoothing (a)
#   Double Exponential Smoothing (a, b)
#   Triple Exponential Smoothing a.k.a. Holt-Winters (a, b, g)

# Statistical Methods
#   AR (p), MA (q), ARMA (p, q)
#   ARIMA (p, d, q)
#   SARIMA (p, d, q)(P, D, Q)m

# Machine Learning for Time Series Forecasting

###################################################################################################################

# What is Time Series?

# A time series is data that consists of observation values over time.

# 1. Stationary
# 2. Trend
# 3. Seasonality 
# 4. Cycle 

#################
# 1. Stationary #
#################

# Stationarity is the statistical properties of the series that do not change over time.
# A time series is considered stationary if its mean, variance, and covariance remain constant over time. 
# There is an assumption that if the statistical properties of the series will not differ significantly in the future, 
# then it is easier to predict.
# If stationarity is not ensured, there is an assumption that the prediction success will decrease and the confidence in the model will decrease.
##############################
# 2. Trend                   #
##############################

# The structure in which a time series increases or decreases in the long run is called a trend.
# If there is a trend, we can think that there is no stationarity in this case.

#################################
# 3. Seasonality                #
#################################

# Seasonality is when a time series repeats a certain behavior at certain intervals.

##############################
# 4. Cycle                   #
##############################

# Circularity contains repetitive patterns. While seasonality develops in a way that overlaps with more obvious short-term times such as day, week, year, 
# season, cyclicity occurs in a longer-term indefinite structure and does not overlap with structures such as day, week, year, season. 
# It mostly occurs with structural, causal cyclical changes. For example, changes in business, politics, etc.


##################################################
# Understanding the Nature of Time Series Models #
##################################################

##############################################
# Moving Average                             #
##############################################

# The future value of a time series is the average of its k previous values. You can best predict today with yesterday.

##############################################
# Weighted Average                           #
##############################################

# The weighted average is similar to the moving average. It carries the idea of giving more weight to later observations.

##############################################
# Smoothing Yöntemleri                       #
##############################################

# Single Exponential Smoothing (SES) (Only in stationary series. No trend and seasonality)
#   Makes predictions by exponentially correcting.
#   The effects of the past are weighted assuming that the future is more related to the recent past.
#   Estimates are made by exponentially weighting the past actual values and past estimated values.
#   It is suitable for univariate time series without trend and seasonality.

# Double Exponenetial Smoothing (Level + Trend. There should be no seasonality.)

# Triple Exponential Smoothing a.k.a. Holt-Winters (Level + Trend + Seasonality)

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

# This dataset was collected from 1958 to 2001 in Hawaii, USA.
# It is a data set about CO2 in the atmosphere.
# It is a time series data set by years. Our aim for the coming years
# Estimate what the amount of CO2 might be.

data = sm.datasets.co2.load_pandas()
y = data.data
y.index.sort_values()

# There are 2284 observation units in this data set.
# This data set is weekly and records are kept on Saturdays.
# We will convert the period of this dataset to monthly period.

y = y['co2'].resample('MS').mean()
y

y.isnull().sum()
# When we look at the data set, there are 5 missing values.
# It doesn't make much sense to fill in missing values using the mean in time series data.
# In general, it makes more sense if the missing values are filled with the value before or after them.
# Because it may contain seasonal and trend components.

y = y.fillna(y.bfill())
y.head()

# We visualize the series.
y.plot(figsize=(15, 6))
plt.show()

# There appears to be an increase from 1959 to 2001. There is an increasing trend and seasonality in this series.

###################
# Holdout Method #
###################

# Here we will apply holdout method with train test.
# Let's create a train set from 1958 to the end of 1997.

train = y[:'1997-12-01']
len(train)

# When we look at the observation units, since we have converted the weekly period observation units to the monthly period
# There are 478 observation units, that is, 478 moons.

# Let's create a test set from the first month of 1998 to the end of 2001.

test = y['1998-01-01':]
len(test)

# There are 48 observation units, ie 48 months, in the test set.

# We will build a model over 478 months and test this model within 48 months.

###################################
# Time Series Structural Analysis #
###################################

# Stationarity Test (Dickey-Fuller Test)

def is_stationary(y):
    # "H0: Non-stationary"
    # "H1: Stationary"
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

        
# Time Series Components and Stationarity Test
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

# Here our model can be additive or multiplicative. While the residuals of the additive model are independent of each other, 
# the residuals of the multiplicative model are dependent on each other.

for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, stationary=False)

# The decompose operation separates the time series into its components. One of these components is the level component, that is, the average.
# Single exponential smoothing was modeling the level component at the top of the graph. Part 2 is the trend part, part 3 is the seasonality part.
# The 4th part is the residuals of the model. In part 4, the residuals of the model are gathered around 0.
# In the other graph, it gathers around model 1. Thus, we can state an opinion that the model cannot be multiplicative.

for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, stationary=True)
# Result: Non-stationary (H0: non-stationary, p-value: 0.999)
# Result: Non-stationary (H0: non-stationary, p-value: 0.999)

# We cannot reject the H0 hypothesis. Our series is not stationary. Because the H0 hypothesis is not less than 0.05.

################################
# Single Exponential Smoothing #
################################

# SES Level is used in models and stationary series, but when we examine this data set, it does not seem stationary.
# Not used if there are trends and seasonality. # It can be used but does not give the desired result.
# If the seasonality and residual components are independent of trend, the series is additive.
# If seasonality and residual components are shaped according to trend, that is, if it is dependent, this series is a multiplicative series.

# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

# "smoothing_level" is alpha editing parameter.
# The smoothing_level parameter of the SES method normally works to find the most appropriate value with the maximum likelihood method.

y_pred = ses_model.forecast(48)

# We made predictions using the model. The number 48 indicates the number of months in the test set.

mean_absolute_error(test, y_pred)

# We visually look at the predictive success of the model.

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()

# If we zoom the image;

train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()

# We use the following function to show visual and numerical success together. This function is a special function for this problem.

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae, 2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params
# {'smoothing_level': 0.5,
#  'smoothing_trend': nan,
#  'smoothing_seasonal': nan,
#  'damping_trend': nan,
#  'initial_level': 316.4419295466226,
#  'initial_trend': nan,
#  'initial_seasons': array([], dtype=float64),
#  'use_boxcox': False,
#  'lamda': None,
#  'remove_bias': False}

###############################
# Hyperparameter Optimization #
###############################

def ses_optimizer(train, alphas, step=48):
    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)
# We say let our alpha value go from 0.8 to 1, increasing by 0.01.

ses_optimizer(train, alphas)
# alpha: 0.8 mae: 4.953
# alpha: 0.81 mae: 4.9282
# alpha: 0.82 mae: 4.9035
# alpha: 0.83 mae: 4.8792
# alpha: 0.84 mae: 4.8551
# alpha: 0.85 mae: 4.8316
# alpha: 0.86 mae: 4.8091
# alpha: 0.87 mae: 4.7869
# alpha: 0.88 mae: 4.765
# alpha: 0.89 mae: 4.7434
# alpha: 0.9 mae: 4.7221
# alpha: 0.91 mae: 4.7012
# alpha: 0.92 mae: 4.6805
# alpha: 0.93 mae: 4.6602
# alpha: 0.94 mae: 4.6402
# alpha: 0.95 mae: 4.6205
# alpha: 0.96 mae: 4.6012
# alpha: 0.97 mae: 4.5822
# alpha: 0.98 mae: 4.5634
# alpha: 0.99 mae: 4.5451
# best_alpha: 0.99 best_mae: 4.5451

# The "sound_optimizer" function put the alpha values we gave one by one into the SES formula and brought the outputs to us.

# Previously best_mae dropped from 5.71 to 4.54.

best_alpha, best_mae = ses_optimizer(train, alphas)

###################
# Final SES Model #
###################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)

plot_co2(train, test, y_pred, "Single Exponential Smoothing")
mean_absolute_error(test, y_pred)

######################################
# Double Exponential Smoothing (DES) #
######################################

# Double Exponential Smoothing makes exponential smoothing by considering the trend effect.
# We get the past true value, the past level information plus the past trend information to get the average of the level, ie time t.
# DES = Level (SES) + Trend
# The basic approach is the same. In addition to Single Exponential Smoothing, the trend is also taken into account.
# It is suitable for univariate time series with and without seasonality.

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5, smoothing_trend=0.5)

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

des_model.params
# 'smoothing_level': 0.5,
#  'smoothing_trend': 0.5,
#  'smoothing_seasonal': nan,
#  'damping_trend': nan,
#  'initial_level': 317.5299849618756,
#  'initial_trend': -0.3037546623833925,
#  'initial_seasons': array([], dtype=float64),
#  'use_boxcox': False,
#  'lamda': None,
#  'remove_bias': False}

###############################
# Hyperparameter Optimization #
###############################

def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha,
                                                                     smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)

# While the success of the model before optimizing was 5.75, the success of the model after optimizing decreased to 1.74.

###################
# Final DES Model #
###################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha, smoothing_slope= best_beta)
y_pred = final_des_model.forecast(48)


plot_co2(train, test, y_pred, "Double Exponential Smoothing")

##############################################################
# * Triple Exponential Smoothing a.k.a. Holt-Winters (TES) * #
##############################################################

# Triple exponential smoothing a.k.a. Holt-Winters method models LEVEL(SES), Trend and Seasonality.
# Triple exponential smoothing is the most advanced smoothing method.
# This method makes predictions by evaluating the effects of level, trend and seasonality dynamically.
# It can be used in univariate series with trend and/or seasonality.

# is the smoothing factor of the alpha level component.
# is the smoothing factor of the beta trend component.
# gamma is the smoothing factor of the seasonality component.

tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.5, smoothing_slope=0.5, smoothing_seasonal=0.5)

# We set the seasonality period to be every 12 months.

y_pred = tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

###############################
# Hyperparameter Optimization #
###############################

alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")

    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)

# Our previous error was 4.66, now it's down to 0.53.


###################
# Final TES Model #
###################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

###########################################################################################################

#######################
# Statistical Methods #
#######################

################
# AR, MA, ARMA #
################

# * AR(p): Autoregression * #

# p here denotes the degree of the pattern. is an argument about whether to put a delay in the model or not. It refers to past real values.
# Estimation is done with a linear combination of observations from previous time steps.
# It is suitable for univariate time series without trend and seasonality.
# p: is the time delay number. If p = 1, it means that the model was established with the previous time step.



# * MA(q): Moving Average * #

# The q value contains past errors. We use it to determine the number of periods of error to consider.
# Prediction is made with a linear combination of errors obtained in previous time steps.
# It is suitable for univariate time series without trend and seasonality.
# q: is the time delay number.



# * ARMA(p, q) = AR(p) + MA(q) * #

# AutoRegressive Moving Average combines AR and MA methods.
# Prediction is made with a linear combination of past values and past errors.
# It is suitable for univariate time series without trend and seasonality.
# p and q are time delay numbers. p is used for the AR model and q is used for the MA model.


# * ARIMA(p, d, q) * #
# (Autoregressive Integrated Moving Average)

# Estimate is made by a linear combination of differenced observations and errors from previous time steps.
# Suitable for univariate data with a trend but no seasonality.
# p: actual value delay number (autoregressive degree) If p = 2, yt-1 and yt-2 are in the model.
# d: number of difference operations (difference degree, I)
# q: is the error delay number. (moving average rating)



# * SARIMA(p, d, q)(P, D, Q)m * #
# (Seasonal Autoregressive Integrated Moving-Average)

# SARIMA = ARIMA + seasonality.
# It can be used in univariate series with trend and/or seasonality.
# p, d, q Parameters from ARIMA are Trend elements. ARIMA can model trend.
# p: actual value delay number (autoregressive degree)
# If p = 2, yt-1 and yt-2 are in the model.
# d: number of difference operations (degree of difference).
# q: number of error delays. (moving average rating)
# If q=2, et-1 and et-2 are in the model.
# P, D, Q are seasonal lag numbers. Season elements.
# m is the number of time steps for a single seasonality period. Expresses the structure of seeing seasonality.

# If a series is Stationary; We use SES, AR, MA, ARMA models.
# If a series is Trend; We use DES, ARIMA, SARIMA models.
# If there is Trend and Seasonality in a series, we use TES, SARIMA models.


##############################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average) #
##############################################################

arima_model = sm.tsa.arima.ARIMA(train, order=(1,1,1))

# We don't know the model grades. We enter the number of past and future values as 1, take the difference as 1, and error residuals as 1 in the model.

arima_model = arima_model.fit()
print(arima_model.summary())
y_pred = arima_model.forecast(48)[0]

# Since we have a 48 unit guess, we enter the argument 48. Since the value we want is in 0. index, we get 0.

y_pred = pd.Series(y_pred, index=test.index)
plot_co2(train, test, y_pred, "ARIMA")

# Our model caught the trend and level, but not seasonality.


##############################################################
# Hyperparameter Optimization (Determining Model Grades) #
##############################################################

#1. Determining Model Grade Based on AIC Statistics
#2. Determining Model Grade Based on ACF & PACF Charts

##############################################################
# Determining Model Grade Based on AIC and BIC Statistics #
##############################################################

# Generating p and q combinations

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params


best_params_aic = arima_optimizer_aic(train, pdq)

arima_model = sm.tsa.arima.ARIMA(train, best_params_aic)
arima_model = arima_model.fit()
y_pred = arima_model.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "ARIMA")

########################################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving Average) #
########################################################################

model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))
sarima_model = model.fit(disp=0)

# The "order" argument is the trend components from ARIMA.
# The "seasonal_order" argument is the model degrees related to the seasonality component. Our seasonality period is 12.

y_pred_test = sarima_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean

y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")

##############################################################
# Hyperparameter Optimization (Model Derecelerini Belirleme) #
##############################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)


###############
# Final Model #
###############

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

# MAE
y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")

##################################################
# SARIMA Optimization by MAE                     #
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=48)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)

                # mae = fit_model_sarima(train, val, param, param_seasonal)

                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)


##################################################
# Examining the Statistical Outputs of the Model #
##################################################
sarima_final_model.plot_diagnostics(figsize=(15, 12))
plt.show()

###############
# Final Model #
###############

model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=1)
y_pred = y_pred_test.predicted_mean

##################################################
# Examining the Statistical Outputs of the Model #
##################################################

# In traditional methods, there are necessary assumptions for establishing a time series model.
# These;
# There should be no autocorrelation between the residuals.
# The mean of the residuals must be zero.
# The variance of the residuals must be constant.
# Residues should be normally distributed.

# Among these assumptions, the most important one is whether the model residuals, namely errors, are normally distributed with a zero mean.

sarima_final_model.plot_diagnostics(figsize=(15, 12))
plt.show()

# The graph in the upper right corner shows whether our prediction residuals and the standard normal distribution are compatible.
# Our residuals overlapped with the theoretical standard normal distribution.

# Our graph in the lower left corner shows whether our distribution is compatible with the theoretical normal standard distribution.

# When we look at the graph in the upper left corner, it shows that there is no trend and no obvious seasonality in the residuals. 
# It is randomly distributed around 0.

# Indicates the autocorrelation in delays in the lower right graph. There is no correlation in delays either.

####################################
# * DEMAND FORECASTING CHALLENGE * #
####################################

# 3-month item-level sales forecast for different store,
# There are 10 different stores and 50 different items in a 5-year dataset.
# Accordingly, we need to give forecasts for 3 months after the store-item breakdown.

import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

#############################
# Exploration Data Analysis #
#############################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

df = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_9/datasets/airline-passengers.csv", index_col='month', parse_dates=True)

df.shape

# There are 144 observation units in the data set.

df.head()

check_df(df)

######################
# Data Visualization #
######################

df[['total_passengers']].plot(title='Passengers Data')
plt.show()

df.index.freq = "MS"

# This data set is organized by month. We need to indicate that this data set is in a monthly period.

train = df[:120]

# We separate 120 observations as a train set.

test = df[120:]

# We separate the 24 observations as the test set.

################################
# Single Exponential Smoothing #
################################

def ses_optimizer(train, alphas, step=48):
    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.01, 1, 0.10)

best_alpha, best_mae = ses_optimizer(train, alphas, step=24)
# best_alpha: 0.11 best_mae: 82.528

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)

y_pred = ses_model.forecast(24)

def plot_prediction(y_pred, label):
    train["total_passengers"].plot(legend=True, label="TRAIN")
    test["total_passengers"].plot(legend=True, label="TEST")
    y_pred.plot(legend=True, label="PREDICTION")
    plt.title("Train, Test and Predicted Test Using"+label)
    plt.show()

plot_prediction(y_pred, "Single Exponential Smoothing")

# There is trend and seasonality in this series. The SES model did not capture this dataset well.

################################
# Double Exponential Smoothing #
################################

def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha,
                                                                     smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas, step=24)
# best_alpha: 0.01 best_beta: 0.11 best_mae: 54.1036

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha, smoothing_slope=best_beta)

y_pred = des_model.forecast(24)

plot_prediction(y_pred, "Double Exponential Smoothing")

# In this series, it caught the trend, but not the seasonality.

################################
# Triple Exponential Smoothing #
################################

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")

    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg, step=24)

# best_alpha: 0.3 best_beta: 0.3 best_gamma: 0.5 best_mae: 11.9947
# Error of additive       : 11.99
# Error of multiplicative : 15.12

tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

y_pred = tes_model.forecast(24)

plot_prediction(y_pred, "Triple Exponential Smoothing")

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params

best_params_aic = arima_optimizer_aic(train, pdq)

# Tuned Model

arima_model = sm.tsa.arima.ARIMA(train, best_params_aic)
arima_model = arima_model.fit()
y_pred = arima_model.forecast(24)[0]
mean_absolute_error(test, y_pred)
#206.341

plot_prediction(pd.Series(y_pred, index=test.index), "ARIMA")

##############
# SARIMA AIC #
##############

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

# Tuned Model

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# Out[62]: 12341.123118443598

plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")

# It looks very bad. We made this optimization according to the AIC statistic.

##############
# SARIMA MAE #
##############

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=24)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)

                # mae = fit_model_sarima(train, val, param, param_seasonal)

                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order, best_mae

best_order, best_seasonal_order, best_mae = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 30.6627

plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")

# The Holt-Winters method turned out better.
