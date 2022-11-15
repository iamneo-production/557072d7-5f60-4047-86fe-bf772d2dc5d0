import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# Statistical Methods ❄️")
st.sidebar.markdown("# Statistical Methods ❄️")

@st.cache
def load_data():
  data_1 = pd.read_excel("NASS.xlsx", index_col='DATE', parse_dates=True)
  data_1["spread"] = data_1["DGS30"] - data_1["DGS1"]   
  data_1 = data_1.drop(['DGS30', 'DGS1'], axis=1)
  return data_1

data_1 = load_data()
tx = data_1.copy()

st.subheader("Statistical Graphs")

st.title("Histogram")
import seaborn as sns
fig = plt.figure(figsize=(10, 4))
sns.histplot(tx)
st.pyplot(fig)





st.title("Boxplot")
rad = st.sidebar.radio("Choose Whisker parameter for boxplot", options=[1, 1.5, 2, 2.5])
st.write("The width of the box (Q1 to Q3) is called interquartile range (IQR) calculated as the difference between the 75th and 25th percentiles (Q3 – Q1). The lower fence is calculated as Q1 - (1.5 x IQR), and the upper fence as Q3 + (1.5 x IQR). Any observation less than the lower boundary or greater than the upper boundary is considered a potential outlier")
fig4 = plt.figure(figsize=(10, 4))
sns.boxplot(tx['spread'], whis=rad)
st.pyplot(fig4)
st.write("A box plot provides more information than a histogram and can be a better choice for spotting outliers. In a box plot, observations that are outside the whiskers or boundaries are considered outliers. The whiskers represent the visual boundary for the upper and lower fences")

st.title("Boxenplot")
st.write("boxen (letter-value) plots are better suited when working with larger datasets (higher number of observations for displaying data distribution and more suitable for differentiating outlier points for larger datasets).")
fig5 = plt.figure(figsize=(10, 4))


k = st.sidebar.radio("Choose k_depth parameter for Boxenplot", options=["tukey", "proportion", "trustworthy", "full"])
sns.boxenplot(tx['spread'], k_depth=k)
plt.title(k)
st.pyplot(fig5)

st.title("Violin Plot")
st.write("hybrid between a box plot and a kernel density estimation (KDE). A kernel is a function that estimates the probability density function, the larger peaks (wider area), for example, show where the majority of the points are concentrated. This means that there is a higher probability that a data point will be in that region as opposed to the much thinner regions showing much lower probability.")
fig6 = plt.figure(figsize=(10, 4))
sns.violinplot(tx['spread'])
st.pyplot(fig6)
st.write("There is more than one peak, we call it a multimodal distribution")

st.title("Lag Plot")
st.write("we plot the same variable against its lagged version. This means, it is a scatter plot using the same variable, but the y axis represents passenger count at the current time (t) and the x axis will show passenger count at a prior period (t-1), which we call lag. The lag parameter determines how many periods to go back;")
from pandas.plotting import lag_plot
fig7 = plt.figure(figsize=(10, 4))
lag_plot(tx)
st.pyplot(fig7)

st.title("Detecting outliers using the Tukey method")

def iqr_outliers(data, p):
    q1, q3 = np.percentile(data, [25, 75])
    IQR = q3 - q1
    lower_fence = q1 - (p * IQR)
    upper_fence = q3 + (p * IQR)
    return data[(data.spread > upper_fence) | (data.spread < lower_fence)]

p = st.sidebar.radio("Choose parameter for Turkey Method", options=[1,1.3, 1.5,1.7,1.9, 2.0,2.3, 2.5])
st.write(p)
st.write(iqr_outliers(tx, p))


st.title("Detecting outliers using a z-score")
def zscore(df, degree=3):
  data = df.copy()
  data['zscore'] = (data - data.mean())/data.std()
  outliers = data[(data['zscore'] <= -degree) | (data['zscore'] >= degree)]
  return outliers['spread'], data

threshold = 2.5

outliers, transformed = zscore(tx, threshold)

fig8 = plt.figure(figsize=(10, 4))
transformed.hist()
st.write(outliers)


import matplotlib.pyplot as plt

def plot_zscore(data, d=3):
  n = len(data)
  fig = plt.figure(figsize=(8,8))
  plt.plot(data,'k^')
  plt.plot([0,n],[d,d],'r--')
  plt.plot([0,n],[-d,-d],'r--')
  st.pyplot(fig)

data = transformed['zscore'].values
d = st.sidebar.radio("Choose parameter for Zscore", options=[1,1.5,2.0,2.5, 3.0])
fig9 = plt.figure(figsize=(10, 4))
plot_zscore(data, d=2.5)
st.write("The data points which are above and below the horizontal lines represent the outliers that were returned by the zscore function. Run the function using different threshold values to gain a deeper understanding of this simple technique.")


st.title("Kolmogorov-Smirnov Test")
st.write(". The null hypothesis is that the data comes from a normal distribution. The test returns the test statistics and a p-value; if the p-value is less than 0.05, you can reject the null hypothesis (data is not normally distributed). Otherwise, you would fail to reject the null hypothesis (data is normally distributed).")

from statsmodels.stats.diagnostic import kstest_normal

def test_normal(df):
    t_test, p_value = kstest_normal(df)
    if p_value < 0.05:
        st.write("Reject null hypothesis. Data is not normal")
    else:
       st.subheader("Result")
       st.write("-> Fail to reject null hypothesis. Data is normal")

test_normal(tx)       


st.title("Detecting outliers using a modified z-score")

import scipy.stats as stats

def modified_zscore(df, degree=3):
  data = df.copy()
  s = stats.norm.ppf(0.75)
  numerator = s*(data - data.median())
  MAD = np.abs(data - data.median()).median()
  data['m_zscore'] = numerator/MAD
  outliers = data[(data['m_zscore'] > degree) | (data['m_zscore'] < -degree)]
  return outliers['spread'], data

threshold = 3
outliers, transformed = modified_zscore (tx, threshold)

fig9 = plt.figure(figsize=(10, 4))
transformed.hist()
st.write(outliers)

def plot_m_zscore(data, d=3):
  n = len(data)
  fig = plt.figure(figsize=(8,8))
  plt.plot(data,'k^')
  plt.plot([0,n],[d,d],'r--')
  plt.plot([0,n],[-d,-d],'r--')
  st.pyplot(fig)

data = transformed['m_zscore'].values
plot_m_zscore(data, d=2.5)

st.title("Quantile-Quantile plot (QQ-plot)")
import scipy
import matplotlib.pyplot as plt

fig10 = plt.figure(figsize=(8,8))
res = scipy.stats.probplot(tx.values.reshape(-1), plot=plt)
st.write(fig10)



st.write("The solid line represents a reference line for what normally distributed data would look like. If the data you are comparing is normally distributed, all the points will lie on that straight line. In Figure, we can see that the distribution is almost normal (not perfect), and we see issues toward the distribution's tails. Showing the majority of the outliers are at the bottom tail end (less than -2 standard deviation).")