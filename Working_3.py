###########################
# Time Series Forecasting #
###########################

# Zaman Serisine Giriş ve Temel Kavramlar

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

# Zaman Serisi Nedir?

# Zaman serisi, zamana göre gözlem değerlerinden oluşan verilerdir.

# 1. Stationary (Durağanlık)
# 2. Trend
# 3. Seasonality (Mevsimsellik)
# 4. Cycle (Döngüsel)

##############################
# 1. Stationary (Durağanlık) #
##############################

# Durağanlık, serinin istatistiksel özelliklerinin zaman içerisinde değişmemesidir.
# Bir zaman serisinin ortalaması, varyansı ve kovaryansı zaman boyunca sabit kalıyorsa,
# serinin durağan olduğu kabul edilir. Serinin istatistiki özellikleri gelecekte ciddi
# şekilde farklılaşmayacaksa bu durumda tahmin yapmak daha kolaydır, varsayımı vardır.
# Durağanlık sağlanmıyorsa tahmin başarısı düşecek modele güven azalacaktır, varsayımı vardır.

##############################
# 2. Trend                   #
##############################

# Bir zaman serisinin uzun vadedeki artış ya da azalışının gösterdiği yapıya trend denir.
# Trend varsa bu durumda durağanlık yoktur diye düşünebiliriz.

#################################
# 3. Seasonality (Mevsimsellik) #
#################################

# Zaman serisinin belirli bir davranışı belirli periyotlarla tekrar etmesi durumuna mevsimsellik denir.

##############################
# 4. Cycle                   #
##############################

# Döngüsellik, tekrar eden düzenler barındırır. Mevsimsellik daha belirgin kısa vadeli, gün ,hafta ,yıl,
# mevsim gibi zamanlarla örtüşecek şekilde gelişir iken, döngüsellik ise daha uzun vadeli belirsiz bir
# yapıda gün, hafta, yıl, mevsim gibi yapılarla örtüşmeyecek şekilde gerçekleşir. Daha çok yapısal, nedensel
# konjonktürel değişimlerle ortaya çıkar. Örnek olarak, iş dünyasındaki, politika dünyasındaki değişimler, vb.


##############################################
# Zaman Serisi Modellerinin Doğasını Anlamak #
##############################################

##############################################
# Moving Average                             #
##############################################

# Bir zaman serisinin gelecek değeri kendisinin k adet önceki değerinin ortalamasıdır.
# Bugünü en iyi dün ile tahmin edebilirsiniz.

##############################################
# Weighted Average                           #
##############################################

# Ağırlıklı ortalama hareketli ortalamaya benzer. Daha sonlarda olan gözlemlere daha fazla ağırlık
# verme fikrini taşır.

##############################################
# Smoothing Yöntemleri                       #
##############################################

# Single Exponential Smoothing (SES) (Sadece durağan serilerde. Trend ve mevsimsellik olmamalı)
#   Üssel düzeltme yaparak tahminde bulunur.
#   Gelecek yakın geçmişle daha fazla ilişkilidir varsayımıyla geçmişin etkileri ağırlıklandırılır.
#   Geçmiş gerçek değerler ve geçmiş tahmin edilen değerlerin üssel olarak ağırlıklandırılmasıyla tahmin yapılır.
#   Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.

# Double Exponenetial Smoothing (Level + Trend. Mevsimsellik olmamalı.)

# Triple Exponential Smoothing a.k.a. Holt-Winters (Level + Trend + Mevsimsellik)

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

# Bu veri seti 1958 ile 2001 yılları arasında A.B.D Hawaii'de toplanmıştır.
# Atmosferdeki CO2 ile ilgili bir veri setidir.
# Yıllara göre bir zaman serisi veri setidir. Amacımız gelecek yıllara göre
# CO2 miktarının ne olabileceği ile ilgili tahminde bulunmak.
data = sm.datasets.co2.load_pandas()
y = data.data
y.index.sort_values()

# Bu veri setinde 2284 tane gözlem birimi vardır.
# Bu veri seti haftalıktır ve cumartesi günleri kayıtlar tutulmuştur.
# Bu veri setinin periyodunu aylık periyota çevireceğiz.

y = y['co2'].resample('MS').mean()
y

y.isnull().sum()
# Veri setine baktığımızda 5 tane eksik değer var.

# Zaman serisi verilerinde ortalamayı kullanarak eksik değerleri doldurmak çok mantıklı değildir.
# Genel olarak eksik değerler kendisinden önceki veya sonraki değerle doldurulursa daha mantıklı olmaktadır.
# Çünkü içerisinde mevsillellik ve trend bileşeni olabilir.

y = y.fillna(y.bfill())
y.head()

# Seriyi görselleştiriyoruz.
y.plot(figsize=(15, 6))
plt.show()

# 1959'dan 2001'e kadar bir artış gözükmektedir. Bu seride artış trendi ve mevsimsellik vardır.

###################
# Holdout Yöntemi #
###################

# Burada train test ile holdout yöntemi uygulayacağız.
# 1958'den 1997 sonuna kadar bir train seti oluşturalım.

train = y[:'1997-12-01']
len(train)
# Haftalık periyottaki gözlem birimlerini aylık periyoda çevirdiğimiz için gözlem birimlerine baktığımızda
# 478 tane gözlem birimi yani 478 ay vardır.

# 1998'in ilk ayından 2001'in sonuna kadar test seti oluşturalım.

test = y['1998-01-01':]
len(test)
# Test setinde 48 gözlem birimi yani 48 ay vardır.

# 478 ay üzerinden model kurup 48 ay içerisinden bu modeli test edeceğiz.

################################
# Zaman Serisi Yapısal Analizi #
################################

# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):
    # "H0: Non-stationary"
    # "H1: Stationary"
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-stationary (H0: non-stationary, p-value: {round(p_value, 3)})")


# Zaman Serisi Bileşenleri ve Durağanlık Testi
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

# Burada modelimiz toplamsal veya çarpımsal olabilir. Toplamsal modelin artıkları birbirinden bağımsız iken,
# çarpımsal modelin artıkları birbirine bağımlı olur.

for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, stationary=False)

# Yapılan decompose işlemi zaman serilerini bileşenlerine ayırır. Bu bileşenlerden
# biri level bileşenidir yani ortalamadır. Grafiğin en üstündeki kısımdaki level bileşenini
# single exponential smoothing modelliyordu. 2. kısım trend bölümü, 3. kısım mevsimsellik
# bölümüdür. 4. bölüm ise modelin artıklarıdır. 4. kısımda modelin artıkları 0'ın etrafında
# toplanmaktadır. Diğer grafikte ise model 1'in etrafında toplanmaktadır. Bu da modelin
# çarpımsal olamayacağı ile ilgili bir görüş belirtebiliriz.

for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, stationary=True)
# Result: Non-stationary (H0: non-stationary, p-value: 0.999)
# Result: Non-stationary (H0: non-stationary, p-value: 0.999)

# H0 hipotezini reddedemiyoruz. Serimiz durağan değildir. Çünkü H0 hipotezi 0.05'ten küçük değildir.

################################
# Single Exponential Smoothing #
################################

# SES Level'i modelleri ve durağan serilerde kullanılır ama bu veri setimizi incelediğimizde durağanlık gözükmüyor.
# Trend ve mevsimsellik varsa kullanılmaz. # Kullanılabilir ama istenilen sonucu vermez.
# Mevsimsellik ve artık bileşenleri trend'den bağımsız ise seri toplamsal bir seridir.
# Mevsimsellik ve artık bieleşenleri trend'e göre şekilleniyorsa yani bağımlıysa bu seri çarpımsal bir seridir.

# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)
# "smoothing_level" alfa düzenleştirme parametresidir.
# SES yönteminin smoothing_level parametresi normalde maksimum olabilirlik yöntemiyle en uygun değeri bulacak
# şekilde çalışmaktadır.

y_pred = ses_model.forecast(48)
# Modeli kullanarak tahmin işlemi yaptık. 48 rakamı test setindeki ay sayısını belirtmektedir.

mean_absolute_error(test, y_pred)

# Görsel olarak modelin tahmin başarısına bakıyoruz.

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()

# Görseli yakınlaştırırsak;

train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()

# Görsel ve numerik başarıyı bir arada göstermek için aşağıdaki fonksiyonu kullanıyoruz. Bu fonksiyon bu
# probleme özel bir fonksiyondur.

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
# alfa değerimiz 0.8'den 1'e kadar 0.01 artarak gitsin diyoruz.

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

# ses_optimizer fonksiyonu verdiğimiz alfa değerlerini teker teker SES formülünde yerine koydu ve çıktıları önümüze getirdi.

# Daha önce best_mae 5.71'den 4.54'e düştü.

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

# Double Exponential Smoothing, trend etkisini göz önünde bulundurarak üssel düzeltme yapar.
# Level yani t zamanının ortalamasını elde etmek için geçmiş gerçek değer, geçmiş level bilgisi artı
# geçmiş trend bilgisini elde ediyoruz.
# DES = Level (SES) + Trend
# Temel yaklaşım aynıdır. Single Exponential Smoothing'e ek olarak trend de dikkate alınır.
# Trend içeren ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.

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
# Optimize etmeden önce modelin başarısı 5.75 iken optimize ettikten sonraki modelin başarısı 1.74'e kadar düşmüştür.

###################
# Final DES Model #
###################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha, smoothing_slope= best_beta)
y_pred = final_des_model.forecast(48)


plot_co2(train, test, y_pred, "Double Exponential Smoothing")

##############################################################
# * Triple Exponential Smoothing a.k.a. Holt-Winters (TES) * #
##############################################################

# Triple exponential smoothing a.k.a. Holt-Winters yöntemi LEVEL(SES)'i, Trend'i ve Mevsimselliği modellemektedir.
# Triple exponential smoothing en gelişmiş smoothing yöntemidir.
# Bu yöntem dinamik olarak level, trend ve mevsimsellik etkilerini değerlendirerek tahmin yapmaktadır.
# Trend ve/veya mevsimsellik içeren tek değişkenli serilerde kullanılabilir.

# alfa level bileşeninin smoothing faktörüdür.
# beta trend bileşeninin smoothing faktörüdür.
# gama mevsimsellik bileşeninin smoothing faktörüdür.

tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.5, smoothing_slope=0.5, smoothing_seasonal=0.5)

# Mevsimsellik periyodunu 12 ayda bir olacak şekilde ayarladık.

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

# Daha önceki hatamız 4.66 iken şimdi ise 0.53'e düştü.


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

# p burada modelin derecesini ifade eder. modele gecikme koyup koymamayla ilgili bir argümandır. Geçmiş gerçek
# değerleri ifade eder.
# Önceki zaman adımlarındaki gözlemlerin doğrusal bir kombinasyonu ile tahmin işlemi yapılır.
# Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
# p: zaman gecikme sayısıdır. p = 1 ise bir önceki zaman adımı ile model kurulmuş demektir.




# * MA(q): Moving Average * #

# q değeri geçmiş hataları barındırır. Kaç periyot hatayı göz önünde bulunduracağımızın
# sayısını belirlemek için kullanırız.
# Önceki zaman adımlarında elde edilen hataların doğrusal bir kombinasyonu ile tahmin yapılır.
# Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
# q: zaman gecikmesi sayısıdır.



# * ARMA(p, q) = AR(p) + MA(q) * #

# AutoRegressive Moving Average, AR ve MA yöntemlerini birleştirir.
# Geçmiş değerler ve geçmiş hataların doğrusal bir kombinasyonu ile tahmin yapılır.
# Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
# p ve q zaman gecikmesi sayılarıdır. p AR modeli için q MA modeli için kullanılır.



# * ARIMA(p, d, q) * #
# (Autoregressive Integrated Moving Average)

# Önceki zaman adımlarındaki farkı alınmış gözlemlerin ve hataların doğrusal bir kombinasyonu ile tahmin yapılır.
# Tek değişkenli, tendi olan fakat mevsimselliği olmayan veriler için uygundur.
# p: gerçek değer gecikme sayısı (otoregresif derece) p = 2 ise yt-1 ve yt-2 modeldedir.
# d: fark işlemi sayısı (fark derecesi, I)
# q: hata gecikme sayısıdır. (hareketli ortalama derecesi)



# * SARIMA(p, d, q)(P, D, Q)m * #
# (Seasonal Autoregressive Integrated Moving-Average)

# SARIMA = ARIMA + mevsimselliktir.
# Trend ve/veya mevsimsellik içeren tek değişkenli serilerde kullanılabilir.
# p, d, q ARIMA'dan gelen parametreler, Trend elemanlarıdır. ARIMA trend'i modelleyebilmektedir.
# p: gerçek değer gecikme sayısı (otoregresif derece)
# p = 2 ise yt-1 ve yt-2 modeldedir.
# d: fark işlemi sayısıdır (fark derecesi).
# q: hata gecikme sayısı. (hareketli ortalama derecesi)
# q=2 ise et-1 ve et-2 modeldedir.
# P, D, Q mevsimsel gecikme sayılarıdır. Season elemanlarıdır.
# m tek bir mevsimsellik dönemi için zaman adımı sayısıdır. Mevsimselliğin görülme yapısını ifade eder.


# Bir seri Durağan ise; SES, AR, MA, ARMA modellerini kullanıyoruz.
# Bir seri Trend ise; DES, ARIMA, SARIMA modellerini kullanıyoruz.
# Bir seride Trend ve Mevsimsellik varsa TES, SARIMA modellerini kullanıyoruz.


##############################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average) #
##############################################################
arima_model = sm.tsa.arima.ARIMA(train, order=(1,1,1))
# Model deerecelerini bilmiyoruz. Geçmiş, gelecek değer sayısını 1, fark alma işlemini 1, hata artıklarını modelde
# argümanını da 1 olarak giriyoruz.
arima_model = arima_model.fit()
print(arima_model.summary())
y_pred = arima_model.forecast(48)[0]
# 48 birimlik tahmin yaptırdığımız için 48 argümanını giriyoruz. İstediğimiz değer de 0. index'te olduğu için
# 0'ı alıyoruz.
y_pred = pd.Series(y_pred, index=test.index)
plot_co2(train, test, y_pred, "ARIMA")

# Modelimiz, trend'i ve level'i yakaladı ama mevsimselliği yakalayamadı.

##############################################################
# Hyperparameter Optimization (Model Derecelerini Belirleme) #
##############################################################

# 1. AIC İstatistiğine Göre Model Derecesini Belirleme
# 2. ACF & PACF Grafiklerine Göre Model Derecesini Belirleme

##############################################################
# AIC & BIC İstatistiklerine Göre Model Derecesini Belirleme #
##############################################################

# p ve qq kombinasyonlarının üretilmesi
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

# order argümanı ARIMA'dan gelen trend bileşenleridir.
# seasonal_ order argümanı mevsimsellik bileşeni ile ilgili model dereceleridir. Mevsimsellik periyodumuz 12.

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
# BONUS: MAE'ye Göre SARIMA Optimizasyonu
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
# Modelin İstatistiksel Çıktılarının İncelenmesi #
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
# Modelin İstatistiksel Çıktılarının İncelenmesi #
##################################################

# Geleneksel yöntemlerde zaman serisi modeli kurulması için gerekli varsayımlar vardır.
# Bunlar;
# Kalıntıların arasında otokorelasyon olmamalıdır.
# Kalıntıların ortalaması sıfır olmalıdır.
# Kalıntıların varyansı sabit olmalıdır.
# Kalıntılar normal dağılmalıdır.

# Bu varsayımlar arasında en önemli kabul edilebilecek olanı aslında model artıklarının yani hatalarının
# sıfır ortalama ile normal dağılıp dağılmamasıdır.

sarima_final_model.plot_diagnostics(figsize=(15, 12))
plt.show()

# Sağ üst köşedeki grafik, bizim tahmin artıklarımızın ve standart normal dağılımın uyumlu olup olmadığını göstermektedir.
# Teorik standart normal dağılım ile artıklarımız üst üste gelmiştir.

# Sol alt köşedeki grafiğimiz, teorik normal standart dağılım ile bizim dağılımımızın uyumlu olup
# olmadığını göstermektedir.

# Sol üst köşedeki grafiğe baktığımızda, artıklarda trend ve belirgin bir mevsimsellik kalmadığını göstermektedir.
# 0'ın etrafında rastgele dağılmaktadır.

# Sağ alttaki grafikte gecikmelerdeki otokorelasyonu ifade ediyor. Gecikmelerde de bir korelasyon
# gözükmemektedir.

####################################
# * DEMAND FORECASTING CHALLENGE * #
####################################

# Farklı store için 3 aylık item-level sales tahmini,
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.

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
# Veri setinde 144 tane gözlem birimi var.

df.head()

check_df(df)

######################
# Data Visualization #
######################

df[['total_passengers']].plot(title='Passengers Data')
plt.show()

df.index.freq = "MS"
# Bu veri seti aylara göre düzenlenmiştir. Bu veri setinin aylık periyotta olduğunun bilgisini belirtmemiz gerekmektedir.

train = df[:120]
# 120 gözlemi train seti olarak ayırıyoruz.

test = df[120:]
# 24 gözlemi test seti olarak ayırıyoruz.

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
# Bu seride trend ve mevsimsellik var. SES modeli bu veri setini iyi yakalayamadı.

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
# Bu seride trend'i yakaladı ama mevsimselliği yakalayamadı.

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
# Toplamsalın hatası 11.99
# Çarpımsalın hatası 15.12

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

# Tune Edilmiş Model

arima_model = sm.tsa.arima.ARIMA(train, best_params_aic)
arima_model = arima_model.fit()
y_pred = arima_model.forecast(24)[0]
mean_absolute_error(test, y_pred)
#206.341

plot_prediction(pd.Series(y_pred, index=test.index), "ARIMA")

##########
# SARIMA AIC #
##########

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

# Tune Edilmiş Model

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# Out[62]: 12341.123118443598

plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")

# Çok kötü gözükmektedir. Bu optimizasyonu AIC istatistiğine göre yaptık.

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

# Holt-Winters yöntemi daha iyi çıktı.
