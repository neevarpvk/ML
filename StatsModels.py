
## T-Tests, ANOVA, Skewness, Kurtosis

import warnings
warnings.simplefilter('ignore')

import statsmodels.api as sm
import matplotlib.pyplot as plt
import  pandas as pd

from statsmodels.stats.weightstats import ttest_ind

df = pd.read_csv('datasets/jobtest.csv')

df.boxplot(column=['JPERF'], by = ['MINORITY'])

df_minority_0 = df[df['MINORITY'] == 0]
df_minority_1 = df[df['MINORITY'] == 1]

df_h= ttest_ind(df_minority_1['JPERF'], df_minority_0['JPERF'], alternative='two-sided', value=0)
df_h

######################################################################

df_sp500 = pd.read_csv('datasets/sp500_1987.csv', sep=",")

df_sp500 = df_sp500.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

df_sp500.rename(columns={'Adj Close': 'Close'}, inplace=True)

df_sp500.count()

df_sp500.plot(figsize=(13,6))


df_sp500.dtypes

df_sp500['Date'] = pd.to_datetime(df_sp500['Date'])

df_sp500.dtypes

df_sp500['Returns'] = df_sp500['Close'].pct_change()

df_sp500 = df_sp500.dropna()

df_sp500.count()

sm.stats.stattools.robust_skewness(df_sp500['Returns'])

sm.stats.stattools.robust_kurtosis(df_sp500['Returns'], excess=True)

df_sp500_without_oct19 = df_sp500[df_sp500['Date'] != '1987-10-19']

df_sp500_without_oct19.count()

sm.stats.stattools.robust_skewness(df_sp500_without_oct19['Returns'])

sm.stats.stattools.robust_kurtosis(df_sp500_without_oct19['Returns'], excess=True)
