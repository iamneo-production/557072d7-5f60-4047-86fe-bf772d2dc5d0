from lppls import lppls, data_loader
import numpy as np
import pandas as pd
from datetime import datetime as dt
import streamlit as st

st.subheader("Log Periodic Power Law Singularity (LPPLS) Model")

st.write("The LPPLS model provides a flexible framework to detect bubbles and predict regime changes of a financial asset. A bubble is defined as a faster-than-exponential increase in asset price, that reflects positive feedback loop of higher return anticipations competing with negative feedback spirals of crash expectations. It models a bubble price as a power law with a finite-time singularity decorated by oscillations with a frequency increasing with time.")
# read example dataset into df 
data = data_loader.nasdaq_dotcom()

# convert time to ordinal
time = [pd.Timestamp.toordinal(dt.strptime(t1, '%Y-%m-%d')) for t1 in data['Date']]

# create list of observation data
price = np.log(data['Adj Close'].values)

# create observations array (expected format for LPPLS observations)
observations = np.array([time, price])

# set the max number for searches to perform before giving-up
# the literature suggests 25
MAX_SEARCHES = 25

# instantiate a new LPPLS model with the Nasdaq Dot-com bubble dataset
lppls_model = lppls.LPPLS(observations=observations)

# fit the model to the data and get back the params
tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)


# visualize the fit
fig = lppls_model.plot_fit()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)


res = lppls_model.mp_compute_nested_fits(
    workers=8,
    window_size=120, 
    smallest_window_size=30, 
    outer_increment=1, 
    inner_increment=5, 
    max_searches=25,
    # filter_conditions_config={} # not implemented in 0.6.x
)

fig2 = lppls_model.plot_confidence_indicators(res)
st.pyplot(fig2)


#Reference

st.write("Research paper : https://www.researchgate.net/publication/312549574_LPPLS_Bubble_Indicators_over_Two_Centuries_of_the_SP_500_Index")
st.write("LPPLS Library : https://github.com/Boulder-Investment-Technologies/lppls")