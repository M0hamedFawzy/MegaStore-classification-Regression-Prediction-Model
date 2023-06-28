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



#======================================================================#

#Feature Encoding Function
def Feature_Encoding(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

#======================================================================#
# Read mega store data 
data = pd.read_csv('megastore-regression-dataset.csv')

#Apply train test split 

X = data.drop('Profit', axis=1)
Y = data['Profit']

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data.to_csv('test_data_2.csv', index=False)






mod1 = train_data['Postal Code'].mod(10)
mod2 = train_data['Row ID'].mod(10)
with open('postal code mod value.pkl', 'wb') as f:
    pickle.dump(mod1, f)
with open('row id mod value.pkl', 'wb') as f:
    pickle.dump(mod2, f)    




# Get the mode value for 'Postal Code' column
postal_code_mode = train_data['Postal Code'].mode().values[0]
print("Mode value for 'Postal Code' column:", postal_code_mode)

# Get the mode value for 'Row ID' column
row_id_mode = train_data['Row ID'].mode().values[0]
print("Mode value for 'Row ID' column:", row_id_mode)    
#======================================================================#

#preprocesssing 

train_data.dropna(how='any',inplace=True,axis=0)
train_data.drop_duplicates(subset=None, keep='first', inplace=False)

#handle category tree --> split it into two new columns [main category - sub category]

train_data[['MainCategory', 'SubCategory']] = train_data['CategoryTree'].apply(lambda x: pd.Series(eval(x)))

#split order & ship date 

train_data[[' order mounth',' order day','order year']]= train_data['Order Date'].str.split('/',expand = True)
train_data[[' ship mounth',' ship day','ship year']]= train_data['Ship Date'].str.split('/',expand = True)

#Extract a new feature from date columns --> TimeDuration = ShipDate - OrderDate

train_data["Order Date"] = pd.to_datetime(train_data["Order Date"])
train_data["Ship Date"] = pd.to_datetime(train_data["Ship Date"])
train_data["Time Duration"] = (train_data["Ship Date"] - train_data["Order Date"]).dt.days.astype(int)


#drop CategoryTree & Order Date & Ship Date

train_data.drop('CategoryTree', axis=1, inplace=True)
train_data.drop('Order Date', axis=1, inplace=True)
train_data.drop('Ship Date', axis=1, inplace=True)
train_data.drop('Country', axis=1, inplace=True)

# Encoding 

cols=('State','City','MainCategory','SubCategory','Region','Ship Mode','Segment','Order ID','Customer ID','Customer Name','Product ID','Product Name')
with open('features for encoding regression.pkl', 'wb') as f:
    pickle.dump(cols, f)
#train_data=Feature_Encoding(train_data,cols)
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train_data[c].values))
    train_data[c] = lbl.transform(list(train_data[c].values))

with open('Feature Encoding for reression.pkl', 'wb') as f:
    pickle.dump(lbl, f)
#Drop The redundant columns / unValid Data

X_train_dis=train_data[['State','City','MainCategory','SubCategory','Region','Ship Mode','Segment','Order ID','Customer ID','Customer Name','Product ID','Product Name','Postal Code']]

X_train_con = train_data[['Sales','Quantity','Discount','Time Duration']]

# Calculate Kendall's correlation coefficient for Categorical Data

#corr_train, p_value_train = kendalltau(X_train_dis[['State','City']],Y_train)
corr, pval = kendalltau(X_train_dis['State'], train_data['Profit'])
print(f"Kendall correlation state: {corr:.3f}")
corr, pval = kendalltau(X_train_dis['City'], train_data['Profit'])
print(f"Kendall correlation city : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['MainCategory'], train_data['Profit'])
print(f"Kendall correlation MainCategory : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['SubCategory'], train_data['Profit'])
print(f"Kendall correlation SubCategory : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Region'], train_data['Profit'])
print(f"Kendall correlation Region : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Ship Mode'], train_data['Profit'])
print(f"Kendall correlation Ship Mode : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Segment'], train_data['Profit'])
print(f"Kendall correlation Segment : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Order ID'], train_data['Profit'])
print(f"Kendall correlation Order ID : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Customer ID'], train_data['Profit'])
print(f"Kendall correlation Customer ID : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Customer Name'], train_data['Profit'])
print(f"Kendall correlation Customer Name : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Product ID'], train_data['Profit'])
print(f"Kendall correlation Product ID : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Product Name'], train_data['Profit'])
print(f"Kendall correlation Product Name : {corr:.3f}")
corr, pval = kendalltau(X_train_dis['Postal Code'], train_data['Profit'])
print(f"Kendall correlation Postal Code : {corr:.3f}")
# state , maincat,product id ,region 


# Calculate the Spearman's correlation between two columns --> Numirical Data
corr, _ = spearmanr(X_train_con['Sales'], train_data['Profit'])
print("Spearman's correlation coefficient:", corr)
corr, _ = spearmanr(X_train_con['Quantity'], train_data['Profit'])
print("Spearman's correlation coefficient:", corr)
corr, _ = spearmanr(X_train_con['Discount'], train_data['Profit'])
print("Spearman's correlation coefficient:", corr)
corr, _ = spearmanr(X_train_con['Time Duration'], train_data['Profit'])
print("Spearman's correlation coefficient:", corr)

#=====================================================================================#

#Handeling the outliers
cols = ['Sales', 'Quantity', 'Discount']
 
# Loop through each column
for col in cols:
    # Calculate the IQR
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
 
    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
 
    # Remove outliers from the column
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
 
#Update the columns      
train_data['Sales'] = pd.DataFrame(train_data['Sales'])
train_data['Quantity'] = pd.DataFrame(train_data['Quantity'])
train_data['Discount'] = pd.DataFrame(train_data['Discount'])
#Search for OUtlires
columns_of_interest = ['Sales', 'Quantity', 'Discount','MainCategory', 'Region', 'State' ]
z_scores = (train_data[columns_of_interest] - train_data[columns_of_interest].mean()) / train_data[columns_of_interest].std()
threshold = 3
outliers = train_data[z_scores > threshold]
print(outliers.count())

#sales ,quantity ,discount 
#total =train_data[['State','MainCategory','Product ID','Region','Sales','Quantity','Discount']]
total =train_data[['Product ID','Sales','Quantity','Customer Name']]
with open('selected features for regression.pkl', 'wb') as f:
    pickle.dump(total, f)
#=======================================================================================#

# split the training data into training and validation sets (75/25 split)
X_train_v, X_val, y_train, y_val = train_test_split(total, train_data['Profit'], test_size=0.25, random_state=42)


# Create a linear regression model
model = linear_model.LinearRegression()
scores = cross_val_score(model, X_train_v, y_train, scoring='neg_mean_squared_error', cv=5)
model_1_score = abs(scores.mean())
# Train the model using the training data
model.fit(X_train_v, y_train)
print("linear model cross validation score is "+ str(model_1_score))


 # Make predictions on the testing data
linear_pred = model.predict(X_val)

# Calculate the mean squared error (MSE) as a measure of the model's performance
mse = mean_squared_error( y_val, linear_pred)

# Print the MSE
print("Mean Squared Error linear regression :", mse)
r_score = r2_score(y_val, linear_pred)

# Print the R score
print("R score:", r_score)

# save the model to a file
filename = 'linear_regression_model.sav'
pickle.dump(model, open(filename, 'wb'))
#======================================================================================#

# transforms the existing features to higher degree features.
poly_features = PolynomialFeatures(degree=5)
X_train_poly = poly_features.fit_transform(X_train_v)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
scores = cross_val_score(poly_model, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)
model_2_score = abs(scores.mean())
poly_model.fit(X_train_poly, y_train)
print("polynomial model cross validation score is "+ str(model_2_score))
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_val))

# predicting on test data-set
poly_pred = poly_model.predict(poly_features.fit_transform(X_val))


print('Mean Square Error polynomial regression : ', metrics.mean_squared_error(y_val, poly_pred))
r_score = r2_score(y_val, poly_pred)

# Print the R score
print("R score:", r_score)
# save the model to a file
filename = 'poly_model.sav'
pickle.dump(poly_model, open(filename, 'wb'))
#=====================================================================================================#

# Initialize the Lasso model with desired hyperparameters
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha value as per your requirement
scores = cross_val_score(lasso_model, X_train_v, y_train, scoring='neg_mean_squared_error', cv=5)
model_3_score = abs(scores.mean())
# Train the Lasso model
lasso_model.fit(X_train_v, y_train)
print(" lasso model cross validation score is "+ str(model_3_score))


# Make predictions on the test dataset
lasoo_pred = lasso_model.predict(X_val)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_val, lasoo_pred)

print("Mean Squared Error lasoo :", mse)
r_score = r2_score(y_val, lasoo_pred)

# Print the R score
print("R score:", r_score)
# save the model to a file
filename = 'lasso_model.sav'
pickle.dump(lasso_model, open(filename, 'wb'))
#=====================================================================================================#
# Create the Elastic Net Regression model
en = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
scores = cross_val_score(en, X_train_v, y_train, scoring='neg_mean_squared_error', cv=5)
model_4_score = abs(scores.mean())
# Train the Lasso model
# Train the model on the training data
en.fit(X_train_v, y_train)
print(" lasso model cross validation score is "+ str(model_3_score))


# Make predictions on the testing data
y_pred_ = en.predict(X_val)

# Evaluate the model using mean squared error on the testing set
mse_test = mean_squared_error(y_val, y_pred_)
print("Mean squared error of Elastic:", mse_test)
r_score = r2_score(y_val, y_pred_)

# Print the R score
print("R score:", r_score)
# save the model to a file
filename = 'en_model.sav'
pickle.dump(en, open(filename, 'wb'))


#============================================================================================================================================#

#Test Data 

#preprocesssing 

test_data.dropna(how='any',inplace=True,axis=0)
test_data.drop_duplicates(subset=None, keep='first', inplace=False)

#handle category tree --> split it into two new columns [main category - sub category]

test_data[['MainCategory', 'SubCategory']] = test_data['CategoryTree'].apply(lambda x: pd.Series(eval(x)))

#split order & ship date 

test_data[[' order mounth',' order day','order year']]= test_data['Order Date'].str.split('/',expand = True)
test_data[[' ship mounth',' ship day','ship year']]= test_data['Ship Date'].str.split('/',expand = True)

#Extract a new feature from date columns --> TimeDuration = ShipDate - OrderDate

test_data["Order Date"] = pd.to_datetime(test_data["Order Date"])
test_data["Ship Date"] = pd.to_datetime(test_data["Ship Date"])
test_data["Time Duration"] = (test_data["Ship Date"] - test_data["Order Date"]).dt.days.astype(int)


#drop CategoryTree & Order Date & Ship Date

test_data.drop('CategoryTree', axis=1, inplace=True)
test_data.drop('Order Date', axis=1, inplace=True)
test_data.drop('Ship Date', axis=1, inplace=True)
test_data.drop('Country', axis=1, inplace=True)
# Encoding 

cols=('State','City','MainCategory','SubCategory','Region','Ship Mode','Segment','Order ID','Customer ID','Customer Name','Product ID','Product Name')
test_data =Feature_Encoding(test_data,cols)

#Drop The redundant columns / unValid Data




test_data_dis=test_data[['State','City','MainCategory','SubCategory','Region','Ship Mode','Segment','Order ID','Customer ID','Customer Name','Product ID','Product Name','Postal Code']]

test_data_con = test_data[['Sales','Quantity','Discount','Time Duration']]

# Calculate Kendall's correlation coefficient for Categorical Data

#corr_train, p_value_train = kendalltau(X_train_dis[['State','City']],Y_train)
corr, pval = kendalltau(test_data_dis['State'], test_data['Profit'])
print(f"Kendall correlation state: {corr:.3f}")
corr, pval = kendalltau(test_data_dis['City'], test_data['Profit'])
print(f"Kendall correlation city : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['MainCategory'], test_data['Profit'])
print(f"Kendall correlation MainCategory : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['SubCategory'], test_data['Profit'])
print(f"Kendall correlation SubCategory : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Region'], test_data['Profit'])
print(f"Kendall correlation Region : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Ship Mode'], test_data['Profit'])
print(f"Kendall correlation Ship Mode : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Segment'], test_data['Profit'])
print(f"Kendall correlation Segment : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Order ID'], test_data['Profit'])
print(f"Kendall correlation Order ID : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Customer ID'], test_data['Profit'])
print(f"Kendall correlation Customer ID : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Product ID'], test_data['Profit'])
print(f"Kendall correlation Product ID : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Product Name'], test_data['Profit'])
print(f"Kendall correlation Product Name : {corr:.3f}")
corr, pval = kendalltau(test_data_dis['Postal Code'], test_data['Profit'])
print(f"Kendall correlation Postal Code : {corr:.3f}")
# state , maincat,product id ,region 


# Calculate the Spearman's correlation between two columns --> Numirical Data
corr, _ = spearmanr(test_data_con['Sales'], test_data['Profit'])
print("Spearman's correlation coefficient for sales:", corr)
corr, _ = spearmanr(test_data_con['Quantity'], test_data['Profit'])
print("Spearman's correlation coefficient for quantity:", corr)
corr, _ = spearmanr(test_data_con['Discount'], test_data['Profit'])
print("Spearman's correlation coefficient dicount:", corr)
corr, _ = spearmanr(test_data_con['Time Duration'], test_data['Profit'])
print("Spearman's correlation coefficient time duration:", corr)

#=====================================================================================#

#Handeling the outliers
cols = ['Sales', 'Quantity', 'Discount']
 
# Loop through each column
for col in cols:
    # Calculate the IQR
    Q1 = test_data[col].quantile(0.25)
    Q3 = test_data[col].quantile(0.75)
    IQR = Q3 - Q1
 
    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
 
    # Remove outliers from the column
    test_data = test_data[(test_data[col] >= lower_bound) & (test_data[col] <= upper_bound)]
 
#Update the columns      
test_data['Sales'] = pd.DataFrame(test_data['Sales'])
test_data['Quantity'] = pd.DataFrame(test_data['Quantity'])
test_data['Discount'] = pd.DataFrame(test_data['Discount'])
#Search for OUtlires
columns_of_interest = ['Sales', 'Quantity', 'Discount','MainCategory', 'Region', 'State' ]
z_scores = (test_data[columns_of_interest] - test_data[columns_of_interest].mean()) / test_data[columns_of_interest].std()
threshold = 3
outliers = test_data[z_scores > threshold]
#print(outliers.count())

#sales ,quantity ,discount 
total =test_data[['Product ID','Sales','Quantity','Customer Name']]
#=======================================================================================#

# split the training data into training and validation sets (75/25 split)
x= test_data[['Product ID','Sales','Quantity','Customer Name']]
y=test_data["Profit"]

#=================================== linear regression =================================#

 # Make predictions on the testing data
linear_test_pred = model.predict(x)

# Calculate the mean squared error (MSE) as a measure of the model's performance
mse = mean_squared_error(y, linear_test_pred)

# Print the MSE
print("Mean Squared Error linear regression :", mse)
linear_score= r2_score(y, linear_test_pred)
print("R2 score of linear regression:" ,linear_score)

with open('linear.pkl', 'wb') as f:
    pickle.dump(linear_test_pred, f)

#================================ polynomial regression ================================#

# transforms the existing features to higher degree features.

poly_pred = poly_model.predict(poly_features.fit_transform(x))
# predicting on training data-set




print('Mean Square Error polynomial regression : ', metrics.mean_squared_error(y, poly_pred))
polynomial_score= r2_score(y, poly_pred)
print("R2 score of polynomial regression:" ,polynomial_score)

with open('poly.pkl', 'wb') as f:
    pickle.dump(poly_pred, f)
#======================================== lasso regression =============================#

# Initialize the Lasso model with desired hyperparameters
 # You can adjust the alpha value as per your requirement

# Make predictions on the test dataset
lasoo_pred = lasso_model.predict(x)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, lasoo_pred)

print("Mean Squared Error lasoo :", mse)
lasso_score= r2_score(y, lasoo_pred)
print("R2 score of lasso regression:" ,lasso_score)

with open('lasso.pkl', 'wb') as f:
    pickle.dump(lasoo_pred, f)
#========================================= elastic net regression ========================#
# Create the Elastic Net Regression model



# Make predictions on the testing data
y_pred_test = en.predict(x)

# Evaluate the model using mean squared error on the testing set
mse_test = mean_squared_error(y, y_pred_test)
print("Mean squared error of Elastic:", mse_test)

elastic_score= r2_score(y, y_pred_test)
print("R2 score of elastic regression:" ,elastic_score)

with open('elastic.pkl', 'wb') as f:
    pickle.dump(y_pred_test, f)