# Using Machine Learning to Predict Stock Prices
### Description
This project utilizes Logistric Regression, Linear Regression, KNearest Neighbors, Decision Tree, MLP, and SVC Machine Learning models to predict the stock prices of stocks. I was able to not only predict with the models, but also use pandas to sort the data and matplotlib to later plot the graphs of the model performances when predicting the prices.

**Libraries Used**: SKLearn (RandomForestClassifier, train_test_split, LogisticRegression, LinearRegression, KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier, SVC), Pandas, Matplotlib, and Numpy.

The data used in the program was found on Kaggle, which supplied historical stock and ETF data dating all the way up to 2017. I mention in the conclusion in the jupyter notebook that more data can be uploaded, formatted, and used to make the datasets bigger and more recent. 

### Findings (Found in Jupyter Notebook)
Iterating through the data, we find that every time we make a range for the next close using the data from the previous day, we get it right. For instance, the closes that the iterations are predicting turned out to be, in order (**FOR MSFT STOCK**):
* 11/02/2017 closed at: 83.18 - Between 82.96 and 84.24 - Correct
* 11/03/2017 closed at: 83.18 - Between 82.71 and 84.00 - Correct
* 11/04/2017 closed at: 84.05 - Between 82.43 and 83.71 - Wrong
* 11/05/2017 closed at: 84.14 - Between 83.28 and 84.56 - Correct
* 11/06/2017 closed at: 84.47 - Between 83.20 and 84.48 - Correct
* 11/07/2017 closed at: 84.26 - Between 83.77 and 85.06 - Correct
* 11/08/2017 closed at: 84.56 - Between 83.53 and 84.81 - Correct
* 11/09/2017 closed at: 84.09 - Between 83.55 and 84.83 - Correct
* 11/10/2017 closed at: 83.87 - Between 82.61 and 83.89 - Correct
* (Found Online) 11/11/2017 closed at: 83.93 - Between 82.89 and 84.17 - Correct

Our findings here show us our model seems to be right a lot of the time. However, we do see that we found one date which was wrong, showing that the model isn't always 100% correct, but rather, 0.999% correct.


### Visualization of Model Performances
Below, I have attatched the graphs that the program produces to show how the model performances stack up against each other:

#### AAPL Stock Model Performances
![aapl-bar-graph](https://github.com/lcswnn/ML-Stock-Prediction-Models/assets/118494460/e0c9f0e0-dcef-4761-8e8a-afb739228dc5)

#### AMZN Stock Model Performances
![amzn-bar-graph](https://github.com/lcswnn/ML-Stock-Prediction-Models/assets/118494460/df3408b2-220a-411d-818d-afc382bdf94c)

#### FB/META Stock Model Performances
![fb-bar-graph](https://github.com/lcswnn/ML-Stock-Prediction-Models/assets/118494460/c55642e8-07cb-4121-baf7-5111b2be022e)

#### MSFT Stock Model Performances
![msft-bar-graph](https://github.com/lcswnn/ML-Stock-Prediction-Models/assets/118494460/adea67dc-8608-4ed5-9f9a-557c5e96e677)

#### NVDA Stock Model Performances
![nvda-bar-graph](https://github.com/lcswnn/ML-Stock-Prediction-Models/assets/118494460/73dcc2fc-5151-46f0-bbb4-24f8bd35274e)

#### TSLA Stock Model Performances
![tsla-bar-graph](https://github.com/lcswnn/ML-Stock-Prediction-Models/assets/118494460/ead567f1-6ab3-4990-8fa3-b406c7be2067)

### Conclusions (Also found in Jupyter Notebook)
1. There are different results for different stocks. When it comes to AAPL and MSFT, we find that the models do better. This could be do to less volatility in their prices listed in the data sets. When we choose a stock like AMZN for instance, we find that the models don't fare as well, again, possibly due to volatility and unpredictiveness in the stock price.

2. Linear Regression trumps all other models in this use case. We find that no matter what stock you pass in into the program, it is always predicting with 0.98 accuracy or better. It was consistent in all cases I have tested with it. This would be due to the linear-like path stocks tend to take as time goes on. You can look at all stock charts and see this visually. The consistency of the stock prices helps with this, as we find that it is a gradual climb or fall for many stocks.

3. The worst performing model would be logistic regression. Even though it is strong in other use cases, here, it is not. Again, looking at stock charts, we find that they take more of a linear path, not a logistic one. In the future when more data is available and if the market seems to plateau, showing a logistic path, then we may find this one working in the future, but not at this time.

4. KNeighbors, Decision Trees, Neural Networks, and SVC models all seem to perform based on what stock is being passed into the program. If it is a stock that is not a volatile, or doesn't change in price drastically, then we find that the models tend to perform decently, sitting in the 0.88-0.61 range. However, when we find a stock that would be considered to be more volatile or more drastic in its price changes, we find that the models don't perform as well, somestimes even sitting below 0.10. AAPL bar graphs and MSFT bar graphs show the models performing well, and the FB (now META) and AMZN charts show the model performing poorly.

5. If given more data, especially with the recent boom in AI, I think we would find the models performing differently. This historical data is updated only to 2017, so more recent data and recent trends would make the models produce different results. For instance, one company who has felt the effects of the AI boom for the better would be NVDA. Here, the highest their price had got to was 213.08. Now, the stock is over 1,000. The models currently do not take in news and other data of that kind, meaning they wouldn't know an AI boom has happened. This would affect the accuracy of the models and their predictions would be off, making their scores lower.

