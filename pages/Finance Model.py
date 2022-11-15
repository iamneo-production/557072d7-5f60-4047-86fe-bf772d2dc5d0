from pyrsistent import v
import streamlit as st
import pandas as pd
import numpy as np


st.header("Financial and Quant Factors")

st.write("Bubble, in an economic context, generally refers to a situation where the price for something—an individual stock, a financial asset, or even an entire sector, market, or asset class—exceeds its fundamental value by a large margin. Financial bubbles, aka asset bubbles or economic bubbles, fit into four basic categories: stock market bubbles, market bubbles, credit bubbles, and commodity bubbles.")
st.header("TYPES OF BUBBLES")
st.subheader("Stock market bubbles ")
st.subheader("Asset Market bubbles")
st.subheader("Credit Bubbles ")
st.subheader("Commodity bubbles")
st.image("/home/gaurav/Documents/nass/pages/static/stage.png")

st.image("/home/gaurav/Documents/nass/pages/static/arc.png")

st.header("1. Gross Domestic Product(GDP)")
st.write("It’s never good when the economy contracts, much less two quarters in a row. While the revised second quarter number was an improvement over the initial reading, that will do little to reassure U.S. markets and households about the economy’s health")
st.write(" Less growth in economy essentially indicates bad GDP and this is one of the indicator of the recession,which in turn forms one of the strong indicator of the stock market crash, although not necessarily.")
st.image("/home/gaurav/Documents/nass/pages/static/gdp.png")
st.image("/home/gaurav/Documents/nass/pages/static/covid.png")
st.image("/home/gaurav/Documents/nass/pages/static/covid_report.png")


st.header("2. Consumer Price Index (CPI)")
st.write("Inflation may have moderated somewhat from June and July, but it’s still sky-high and wrecking the purchasing power of everyday Americans. Price growth is so hot that the Fed is willing to increase unemployment and slow the economy even further to get inflation under control.")
st.image("/home/gaurav/Documents/nass/pages/static/cpi.png")
st.image("/home/gaurav/Documents/nass/pages/static/cpi2.png")

st.write("Observation: Once the consumer price index gets low, it's a strong indication that the stock market bubble will be formed.")


st.header("3. ISM Manufacturing Index")
st.write("This survey of corporate executives in industrial companies has been positive for a long time and remained so in August. According to this index, sentiment among executives has been positive—aka a reading above 50—for 27 consecutive months")
st.write(" Observation: Inflation affects the stock market crash or the overall stocks as consumer spending drops. Value stocks might do well because of their prices haven’t kept with their peers. However growth stocks tend to be shunned by the investors. ")
st.write("Industrial production has been hit due to the coronavirus pandemic since March 2020, when it had contracted 18.7 per cent. It shrank 57.3 per cent in April 2020 due to a decline in economic activities in the wake of the lockdown imposed to curb the spread of coronavirus infections. TAGS: Index of industrial production.")

st.header("4. Industrial Production")
st.write("Industrial production declined in August, as businesses struggle with a difficult environment.")
st.write("“Industrial production lost momentum in August falling 0.2%,” per a Wells Fargo Securities report. “While a 2.3% drop in utilities and a flat reading for mining offered no help, manufacturing did eke out a scant 0.1% gain, but regional Fed surveys portend trouble on the horizon for the factory sector.”")
st.image("/home/gaurav/Documents/nass/pages/static/industri_prod.png")
st.image("/home/gaurav/Documents/nass/pages/static/russia.png")
st.write("Whenever any bubble gets formed, or any unexpected situation aggravates the situation of bubble, as depicted in the above picture, industrial production gets affected badly")


st.header("5. Retail Sales")
st.write("The good news: Retail sales gained in August. The bad? The July numbers were revised downward.“July retail sales was significantly revised lower so taken together with the August report, consumer demand for goods is clearly slowing,” said Jeffrey Roach, Chief Economist for LPL Financial.")
st.write("Observation: 1) Less industrial production indicates that the economy is not in good condition. For instance - 2)This essentially forms a cycle of de-growth in the economy and this leads to the stock market crash")



st.header("Recession Tracker: Markets Data")
st.header("1. The Stock Market (S&P 500)")
st.write("Observation: ")
st.image("/home/gaurav/Documents/nass/pages/static/2002.png")
st.image("/home/gaurav/Documents/nass/pages/static/2008.png")
st.write("In the above picture,Whenever there is exponential and continuous rise in the stock market over a longer span of time, it's a strong indicator of economic bubble")
st.write("GDP growth or slowdown is a strong barometer of the economy of the market. The above two graphs indicates that the GDP growth egts slowdown , this leads to the crash of the stock market. ")






st.header("2. Treasury Yield Curve")
st.image("/home/gaurav/Documents/nass/pages/static/yieldcurve.png")
st.write("When short-term interest rates yield more than longer-term rates, it’s called an inverted yield curve. This is typically a tell-tale sign of an impending recession, as the market believes economic growth will be weak. The yield curve has been inverted since early July.")

st.header("Stock Market Crash : Jobs Data Unemployment Rate")
st.image("/home/gaurav/Documents/nass/pages/static/unemp.png")

st.write("Despite wobbliness throughout the economy and concerns about a further slowdown in the coming months, the U.S. labor market remains robust. The unemployment rate has recovered to its pre-pandemic level, and is down 2 percentage points from the same time last year. Meanwhile, employers added 315,000 jobs in August after increasing payroll by 526,000 in July. Essentially, anyone who wants a job can find one. The Fed has two mandates: Maximize employment and keep prices stable. A strong labor situation has allowed the Fed to focus on bringing down inflation.")
st.header("Job Openings and Labor Turnover Survey (JOLTS)")
st.write("Even as the unemployment rate remains low, the total number of available jobs is near recent highs. There were roughly 7 million job openings in July 2019, compared to more than 11.2 million now. For every two job openings, there’s about one person available to work, Roach says.")


st.header("Recession Tracker: Economic Confidence Data University of Michigan Consumer Confidence Survey")
st.write("According to the University of Michigan Survey of Consumers, consumer sentiment ticked up in August, rising by 13% compared to the month before. Meanwhile, consumer expectations increased even more dramatically, largely thanks to moderating energy prices. This is a positive development for an index that has consistently shown that Americans were morose about their financial position. “The relative relief felt by consumers reflected in their inflation expectations,” said Joanne Hsu, the director of the Surveys of Consumers. “The median expected year-ahead inflation rate was 4.8%, down from 5.2% last month and its lowest reading in [eight] months.” Still, consumer sentiment is down 17% from this point last year, showing that there’s still much room to recover.")
st.header("NFIB Small Business Optimism Index")
st.write("The National Federation of Independent Business (NFIB) Small Business Optimism Index rose in August by 1.9 points to 91.8. However, the index has remained below its 48-year average for eight consecutive months. “It’s a complicated environment,” said Jason Greenberg, chief economist at Homebase. Businesses, per Greenberg, are struggling with high costs and finding employees, but feeling good about the future. More than 64% of businesses in Homebase’s Owner Pulse Surveys believe they’ll be better off in 12 months, compared to 57% of organizations in July.")
st.header("Stock Market Crash : Housing Market Data")
st.write("Home building rebounded in August, rising by a seasonally adjusted annual rate of 12.9% in August, after dropping nearly 11% in July. Rising rent helped push demand for construction of multi-family housing.The overall housing situation, though, is down thanks to increased borrowing costs and high prices.")
st.header("NAHB Home Builders Index")
st.write("U.S. home builders are not optimistic. The Home Builders Index fell another three points in September to 46, indicating that most builders view the housing market as poor. This was the ninth consecutive decline, which dovetails with higher mortgage rates and fewer single-homes being constructed.The South is the only region in the U.S. above the breakeven mark of 50.")

st.header("P/E Ratio")
st.image("/home/gaurav/Documents/nass/pages/static/PE.png")
st.write("Higher P/E ratio, either compared to competitor's share or industry standards showcases that a market is overvalued and it's an indication of sideways market soon. However, having higer P/E could also be a sign of growth of a particular company but run-up is rare.")
st.write(" Lower P/E ratio is also a sign of worry because ..")
st.header("DEBT-TO-EQUITY RATIO")
st.image("/home/gaurav/Documents/nass/pages/static/DET.png")
st.write("If D/E>1, it's one of the strong indicator to show that overall health of market is bad and it's one of the indicator of crash")
st.write(" Having low D/E ratio is good as it helps the companies to survive better during tough times")
st.write("Sector specific D/E ratio significance")

st.header("EPS")
st.write("Earnings per share is one of the most important metrics employed when determining a firm's profitability on an absolute basis. It is also a major component of calculating the price-to-earnings (P/E) ratio, where the E in P/E refers to EPS. By dividing a company's share price by its earnings per share, an investor can see the value of a stock in terms of how much the market is willing to pay for each dollar of earnings.")
st.write(")EPS is one of the many indicators you could use to pick stocks. If you have an interest in stock trading or investing, your next step is to choose a broker that works for your investment style. Low EPS indicates that the economy is not in the right state and it’s an indicator of the stock market crash. ")
st.write(" Comparing EPS in absolute terms may not have much meaning to investors because ordinary shareholders do not have direct access to the earnings. Instead, investors will compare EPS with the share price of the stock to determine the value of earnings and how investors feel about future growth.")

st.write("Earnings per share (EPS) is a company's net profit divided by the number of common shares it has outstanding. EPS indicates how much money a company makes for each share of its stock and is a widely used metric for estimating corporate value. A higher EPS indicates greater value because investors will pay more for a company's shares if they think the company has higher profits relative to its share price.EPS can be arrived at in several forms, such as excluding extraordinary items or discontinued operations, or on a diluted basis. Like other financial metrics, earnings per share is most valuable when compared against competitor metrics, companies of the same industry, or across a period of time.")





st.header("We live in world following Globalisation. Crude Oil prices went up throughout the world due to Russia-Ukraine war . Inflation increased throughout the world due to the same. In order to control the inflation, Government took measures to control it by increasing the interest rates. All these indicates bad state of economy")
st.header("Other Factors")
st.write("Expert Opine: Ray Dalio and Michael Burry ,social media such as Twitter and Reddit, famous new channels such as CNBC_TV, Geopolitics of Specific Countries ,just like China Evergrande Group,Unexpected Scenarios such as Covid, Russia Ukraine war, FED Interest Rates- Don’t bet against the FED, Repo rates in India, Artificial Growth in economy")
st.image("/home/gaurav/Documents/nass/pages/static/gg.png")
st.header("Finance Modelling Approaches")
st.header("Probabilistic")
st.write("One of the main assumptions about the financial markets, at least as far as quantitative finance goes, is that asset prices are random. We tend to think of describing financial variables as following some ran- dom path, with parameters describing the growth of the asset and its degree of randomness. We effectively model the asset path via a specified rate of growth,on average, and its deviation from that average. This approach to modelling has had the greatest impact over the last 30 years, leading to the explosive growth of the derivatives markets")
st.write("Pattern of Probability: Generally speaking, crashes usually occur under the following conditions: a prolonged period of rising stock prices (a bull market) and excessive economic optimism, a market where price–earnings ratios exceed long-term averages, and extensive use of margin debt and leverage by market participants.")
st.header("Deterministic")
st.write("The idea behind this approach is that our model will tell us everything about the future. Given enough data, and a big enough brain, we can write down some equations or an algorithm for predicting the future. Interestingly, the subjects of dynamical systems and chaos fall into this category. And, as you know, chaotic systems show such sensitivity to initial condi- tions that predictability is in practice impossible. This is the ‘butterfly effect,’ that a butterfly flapping its wings in Brazil will ‘cause’ rainfall over Manchester. (And what doesn’t!) A topic popular in the early 1990s, this has not lived up to its promises in the financial world.")
st.header("Discrete: difference equations")
st.write("Whether probabilistic or determinis- tic the eventual model you write down can be discrete or continuous. Discrete means that asset prices and/or time can only be incremented in finite chunks, whether a dollar or a cent, a year or a day. Continuous means that no such lower increment exists. The mathemat- ics of continuous processes is often easier than that of discrete ones. But then when it comes to number crunching you have to anyway turn a continuous model into a discrete one.")
st.header("Continuous: differential equations")
st.write("")
st.header("Simulations")
st.write("f the financial world is random then we can experiment with the future by running simulations. For example, an asset price may be represented by its average growth and its risk, so let’s simulate what could happen in the future to this random asset. If we were to take such an approach we would want to run many, many simulations. There’d be little point in running just the one, we’d like to see a range of possible future scenarios.")
st.header("Approximations")
st.write("The complement to simulation methods, there are many types of these. The best known of these are the finite-difference methods which are discretizations of continuous models such as Black– Scholes. Depending on the problem you are solving, and unless it’s very simple, you will probably go down the sim- ulation or finite-difference routes for your number crunching.")
st.header("Asymptotic analysis")
st.write("This is an incredibly useful technique, used in most branches of applicable mathematics, but until recently almost unknown in finance. The idea is simple, find approximate solutions to a complicated problem by exploiting parameters or variables that are either large or small, or special in some way. For example, there are simple approximations for vanilla option values close to expiry.")

st.header("Green’s functions")
st.write("This is a very special technique that only works in certain situations. The idea is that solu- tions to some difficult problems can be built up from solutions to special solutions of a similar problem.")




st.header("Some strong indicators")
st.header("Prolonged Dovish Monetary Policy")
st.image("/home/gaurav/Documents/nass/pages/static/fed.png")
st.header("An Extended Bull Market")
st.header("A High Cyclically Adjusted Price-to-Earnings (CAPE) Ratio")
st.write("Some use the CAPE ratio to determine whether the markets as a whole are undervalued, in which case they should go higher, or over-valued, which suggests they're headed for a crash. The CAPE ratio for the S&P 500 Index at the end of January 2021 stood at 33.74.")
st.header("Rising Inflation")
st.image("/home/gaurav/Documents/nass/pages/static/inflation_stock.png")

st.header("The Buffett Indicator")
st.image("/home/gaurav/Documents/nass/pages/static/buffet_indic.png")
st.header("An Inverted Yield Curve")

st.header("Final Framework")
st.image("/home/gaurav/Documents/nass/pages/static/allbub.png")
st.header("2002 DotCom Bubble")
st.subheader("Factors")
st.subheader("1. Speculation of Crash")
st.subheader("2. Investor Hype & Overevaluation  ")
st.subheader("3. Expert's Opinion: Alan Greenspan’s Speech ")
st.subheader("4. Taxpayer Relief Act of 1997")
st.write("Our Economic Framework: We have essentially created a framework consisting of check lists of economic parameters. Given that we consider the above-mentioned parameters, probability of most of the economic bubbles could be predicted")
st.image("/home/gaurav/Documents/nass/pages/static/final1.png")

st.image("/home/gaurav/Documents/nass/pages/static/tb2.png")
st.write("This picture shows that cycle of economy consists of Recessions, Expansions and Long term Debt cycles")

st.header("Changing World Order: A different financial narrative on the tech bubble ")
st.image("/home/gaurav/Documents/nass/pages/static/tb1.png")
st.write("The above picture depicts that the US devt held by public. It shows everytime a new high comes in the stock markrt, Debt cycle starts ")



st.image("/home/gaurav/Documents/nass/pages/static/tb4.png")
st.image("/home/gaurav/Documents/nass/pages/static/tb5.png")
st.write("The above picture depicts the debt cycle bubble ")


st.header("2008 Global Financial Crisis ")
st.subheader("Factors")
st.subheader("1. Excessive risk-taking in a favourable macroeconomic environment ")
st.subheader("2. Increased borrowing by banks and investors")
st.subheader("3.  Regulation and policy errors")
st.write("Our Economic Framework: We have essentially created a framework consisting of check lists of economic parameters. Given that we consider the above-mentioned parameters, probability of most of the economic bubbles could be predicted ")
st.image("/home/gaurav/Documents/nass/pages/static/final2.png")
st.image("/home/gaurav/Documents/nass/pages/static/final3.png")

st.header("CONCLUSION")
st.write("The world of stock market is extremely subjective. No one can time or predict the market but with the aid of Technology and other financial data points or models, probability of prediction of stock market could be done. In this, different data points have been used and using this framework of considering the above-mentioned data points ,one can check the existence bubble in stock market")