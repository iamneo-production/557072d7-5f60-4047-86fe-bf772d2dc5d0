# 557072d7-5f60-4047-86fe-bf772d2dc5d0
https://sonarcloud.io/summary/overall?id=examly-test_557072d7-5f60-4047-86fe-bf772d2dc5d0


# Stock-Market-Prediction-Web-App-using-Machine-Learning
*Stock Market Prediction* Web App based on *Machine Learning* and *Sentiment Analysis* of Tweets  * and Financial Models. <a href="https://gaurav7888-nass-app-h64hf0.streamlit.app/">Finally  the Web App is deployed  on **Streamlit.</a>Finally The dataset has beeen taken from  **NASDAQ* or *NSE* .  Predictions are made using three algorithms: *ARIMA, LSTM, etc. 

# Screenshots
<img src="">
<img src="">

Find more screenshots in the <b>screenshots folder</b> Or <a href="">click here</a>


# Problem Statement
Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. 
We have developed a system which is essentially a web App and it uses machine learning based models coupled with Financial Framework or knowledge. 
The app forecasts prediction of stock market using algorithms such as Correlation method,Heatmap, KNN Outlier Detection,LOF outlier, Cluster based local outlier factor,isolation forest, One Class Support Vector Machine, COPOD,Mean Absolute Deviation, Log Periodic Power Law Singulairty Model(LPPLS). Deep Learning models based on RNN,LSTM,BI-directional LSTM and Arima. Statistical Methods such as Boxplot,Boxenplot,Violin Plot,Lag Plot,Turkey Method,Z-Score method, Kolmogorov-Smirnov Test,modified Z-Score,Quantile-Quantile Plot(QQ-plot).
We are considering twitter sentiment with other stock market features to get better accuracy. 
Quant models and Financial Frameworks are also used for observing the pattern in stock market, using data for framing a framework which would predict the stock market crash
Financial data such as GDP,CPI,Labour Market Data, ISM, Yield Curve, NFIB Small Business Optimism Index etc. 
Housing Market Data, P/E ratio, D/E ratio,EPS.





# File and Directory Structure
<pre>

screenshots - Screenshots of Web App
pages - consists of different routes/webpages
    -static: static files such as images 
    -data: dataset containing stock market plus Twitter data
    -Finance Model.py: Financial data and other quant aggregator models 
    -Forecasting with Twitter Sentiment.py : Bi-directional LSTM model
    -Log Periodic Power Law Singulairty(LPPLS) Model.py: flexible framework to detect economic bubble and predict regime changes of a financial asset
    -Statistical Model.py: Statistical models to detect crash events 
    -we have saved the model parameter as pickle file
app.py : contains our machine learning model 
requirement.txt: contains all the requirements and libraries which are to be installed
packages.txt:contains all the packages 
Datsets related to Data Yield curve 
</pre>

# Technologies Used
<ul>

<a href="https://streamlit.io/"><li>Streamlit</a></li>
<a href="https://www.tensorflow.org/"><li>Tensorflow</a></li>
<a href="https://keras.io/"><li>Keras</a></li>
<a href="https://www.bloomberg.com/asia/"><li>Bloomberg Finance</a></li>
<a href="https://scikit-learn.org/"><li>Scikit-Learn</a></li>
<a href="https://www.python.org/"><li>Python</a></li>
<a href="https://www.nltk.org/"><li>NLTK</a></li>

</ul>

# How to Install and Use
<b>Python 3.8.7 is required for the python packages to install correctly</b><br>
<ol>

<li>Directly run the Streamlit</li>
<li>Clone the repo. Download and install <b>Streamlit server</b> </li>
<li>Check the requirement.txt file and install all the dependencies using pip install -r requirement.txt</li>
<li>Run the command streamlit run app.py --server.port 8080 </li>
<li>Open http://localhost:8080 and experience the web interface </li>

</ol>
<br>
<b> Video of setup and demo is available <a href="https://youtu.be/slQ4YsFy28s">here</a></b>
<br>


</ul>
