
## T-Tests

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

ttest_ind(df_minority_1['JPERF'], df_minority_0['JPERF'], alternative='two-sided', value=0)
