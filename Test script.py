import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Read mega store data
data = pd.read_csv('test_data.csv')
data2 = pd.read_csv('test_data_2.csv')

#If any nulls found replace with the mean value
meanDict =  {'Row ID': 2, 'Postal Code': 10035, 'Sales': 233.3821511413383, 'Quantity': 3.7642276422764227, 'Discount': 0.1557285803627267}

if data.isnull().values.any():
  for cols in meanDict:
     data[cols] = data[cols].replace(np.nan,meanDict[cols])
'''with open('postal code mod value.pkl', 'rb') as f:
    mod1 = pickle.load(f)
for i in range(len(data)):
    if pd.isnull(data.loc[i, 'Postal Code']):
        data.loc[i, 'Postal Code'] = mod1[i]'''




if data2.isnull().values.any():
  for cols in meanDict:
     data2[cols] = data2[cols].replace(np.nan,meanDict[cols])
'''with open('row id mod value.pkl', 'rb') as f:
    mod2 = pickle.load(f)
for i in range(len(data2)):
    if pd.isnull(data2.loc[i, 'Postal Code']):
        data2.loc[i, 'Postal Code'] = mod2[i]'''

# preprocesssing1

# unwanted features
'''data.drop('Customer Name', axis=1, inplace=True)
data.drop('Country', axis=1, inplace=True)'''

# encoding categorical data
# handle category tree --> split it into two new columns [main category - sub category]

data[['MainCategory', 'SubCategory']] = data['CategoryTree'].apply(lambda x: pd.Series(eval(x)))
data.drop('CategoryTree', axis=1, inplace=True)

# split order & ship date

data[['order mounth', 'order day', 'order year']] = data['Order Date'].str.split('/', expand=True)
data[['ship mounth', 'ship day', 'ship year']] = data['Ship Date'].str.split('/', expand=True)

# Extract a new feature from date columns --> TimeDuration = ShipDate - OrderDate

data["Order Date"] = pd.to_datetime(data["Order Date"])
data["Ship Date"] = pd.to_datetime(data["Ship Date"])
data["Time Duration"] = (data["Ship Date"] - data["Order Date"]).dt.days.astype(int)

data["order mounth"] = pd.to_datetime(data["Order Date"])
data["order day"] = pd.to_datetime(data["Order Date"])
data["order year"] = pd.to_datetime(data["Order Date"])

data["ship mounth"] = pd.to_datetime(data["Ship Date"])
data["ship day"] = pd.to_datetime(data["Ship Date"])
data["ship year"] = pd.to_datetime(data["Ship Date"])

data.drop('Order Date', axis=1, inplace=True)
data.drop('Ship Date', axis=1, inplace=True)




# preprocesssing2

# unwanted features
'''data2.drop('Customer Name', axis=1, inplace=True)
data2.drop('Country', axis=1, inplace=True)'''

# encoding categorical data
# handle category tree --> split it into two new columns [main category - sub category]

data2[['MainCategory', 'SubCategory']] = data2['CategoryTree'].apply(lambda x: pd.Series(eval(x)))
data2.drop('CategoryTree', axis=1, inplace=True)

# split order & ship date

data2[['order mounth', 'order day', 'order year']] = data2['Order Date'].str.split('/', expand=True)
data2[['ship mounth', 'ship day', 'ship year']] = data2['Ship Date'].str.split('/', expand=True)

# Extract a new feature from date columns --> TimeDuration = ShipDate - OrderDate

data2["Order Date"] = pd.to_datetime(data2["Order Date"])
data2["Ship Date"] = pd.to_datetime(data2["Ship Date"])
data2["Time Duration"] = (data2["Ship Date"] - data2["Order Date"]).dt.days.astype(int)

data2["order mounth"] = pd.to_datetime(data2["Order Date"])
data2["order day"] = pd.to_datetime(data2["Order Date"])
data2["order year"] = pd.to_datetime(data2["Order Date"])

data2["ship mounth"] = pd.to_datetime(data2["Ship Date"])
data2["ship day"] = pd.to_datetime(data2["Ship Date"])
data2["ship year"] = pd.to_datetime(data2["Ship Date"])

data2.drop('Order Date', axis=1, inplace=True)
data2.drop('Ship Date', axis=1, inplace=True)



with open('features for encoding.pkl', 'rb') as f:
    cols = pickle.load(f)

with open('features for encoding regression.pkl', 'rb') as f:
    cols2 = pickle.load(f)    

#columns = X[list(cols)]
#Data encoding 
with open('Feature Encoding.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('Feature Encoding for reression.pkl', 'rb') as f:
    encoder2 = pickle.load(f)

# Encode new data
for c in cols:
    # Find new categories
    new_categories = set(data[c].unique()) - set(encoder.classes_)

    # Encode new categories as a new label
    if new_categories:
        encoder.classes_ = np.append(encoder.classes_, list(new_categories))
        encoder.transform(list(new_categories))

    # Transform data using encoder
    data[c] = encoder.transform(list(data[c].values))



# Encode new data2
for c in cols2:
    # Find new categories
    new_categories = set(data2[c].unique()) - set(encoder2.classes_)

    # Encode new categories as a new label
    if new_categories:
        encoder2.classes_ = np.append(encoder2.classes_, list(new_categories))
        encoder2.transform(list(new_categories))

    # Transform data using encoder
    data2[c] = encoder2.transform(list(data2[c].values))

Y = data['ReturnCategory'] 
Y2 = data2['Profit']  
X = data.drop('ReturnCategory', axis=1)
X2 = data2.drop('Profit', axis=1) 


#Data scaling
with open('scaler.pkl', 'rb') as f:
    scaling = pickle.load(f)

'''X[['Order ID']] = scaling.transform(X[['Order ID']])
X[['Product ID']] = scaling.transform(X[['Product ID']])
X[['Product Name']] = scaling.transform(X[['Product Name']])'''
X[['Sales']] = scaling.transform(X[['Sales']])



#selected features
with open('selected features.pkl', 'rb') as f:
    selected = pickle.load(f)
X = X[selected.columns]




#selected features
with open('selected features for regression.pkl', 'rb') as f:
    selected2 = pickle.load(f)
X2 = X2[selected2.columns]



print('================================================================================')
print('=                         Classification Test Script                           =')
print('================================================================================')
#=================models=================#
with open('decision tree.pkl', 'rb') as f:
    dt = pickle.load(f) 
y_pred = dt.predict(X)
accuracy = dt.score(X,Y)
print('Accuracy Of Decesion tree model:',accuracy)  





with open('RandomForest.pkl', 'rb') as f:
    rf = pickle.load(f)
y_pred = rf.predict(X)
accuracy = rf.score(X,Y)
print('Accuracy Of Random Forest model:',accuracy)





with open('Naive Bayes.pkl', 'rb') as f:
    nb = pickle.load(f)
y_pred = nb.predict(X)
accuracy = nb.score(X,Y)
print('Accuracy Of Naive Bayes model:',accuracy)




with open('SVM.pkl', 'rb') as f:
    svm = pickle.load(f)
y_pred = svm.predict(X)
accuracy = svm.score(X,Y)
print('Accuracy Of SVM model:',accuracy)


print('================================================================================')
print('=                           Regression Test Script                             =')
print('================================================================================')
#=================models=================#
with open('linear_regression_model.sav', 'rb') as f:
    model = pickle.load(f)
linear_pred = model.predict(X2)
mse = mean_squared_error(Y2, linear_pred)
r_score = r2_score(Y2, linear_pred)
print("Mean Squared Error linear regression:", mse)
print("R-squared score:", r_score) 
print('====================================================')   
#========================================#
with open('poly_model.sav', 'rb') as f:
    poly_model = pickle.load(f)
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X2)
poly_pred = poly_model.predict(X_poly)
mse = mean_squared_error(Y2, poly_pred)
r_score = r2_score(Y2, poly_pred)
print("Mean Squared Error polynomial regression:", mse)
print("R-squared score:", r_score)
print('====================================================')
#========================================#
with open('lasso_model.sav', 'rb') as f:
    lasso_model = pickle.load(f)
lasso_pred = lasso_model.predict(X2)
mse = mean_squared_error(Y2, lasso_pred)
r_score = r2_score(Y2, lasso_pred)
print("Mean Squared Error lasso regression:", mse)
print("R-squared score:", r_score)
print('====================================================')
#========================================#
with open('en_model.sav', 'rb') as f:
    en_model = pickle.load(f)
en_pred = en_model.predict(X2)
mse = mean_squared_error(Y2, en_pred)
r_score = r2_score(Y2, en_pred)
print("Mean Squared Error Elastic Net regression:", mse)
print("R-squared score:", r_score)
print('====================================================')
#========================================#   