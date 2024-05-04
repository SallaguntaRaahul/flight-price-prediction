

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df= pd.read_excel("Data_Train.xlsx")

df.head()

df.info()

df.shape

"""Data Cleaning"""

#checking for null values
df.isnull()

#Step:1 Finding null values and removing them
df.isnull().sum()

#droping null values
df=df.dropna()

#we can see that the null values are dropped.
df.isnull().sum()

#Step:2
#Checking for Duplicates
df.duplicated().sum()

#Removing Duplicates
df=df.drop_duplicates()

df.duplicated().sum()

#step:3
#outliers
sns.distplot(df['Price'])

#to see the outliers clearly
sns.boxplot(df['Price'])

#Inter Quartile Range
q1=df['Price'].quantile(0.25)
q3=df['Price'].quantile(0.75)
iqr=q3-q1

q1, q3, iqr

upper_limit=q3+(1.5*iqr)
lower_limit=q1-(1.5*iqr)
upper_limit,lower_limit

df.loc[(df['Price']>upper_limit) | (df['Price']<lower_limit) ]

df=df.loc[(df['Price']<upper_limit) & (df['Price']>lower_limit)]

df.shape

#After removing Outliers
sns.boxplot(df['Price'])

#Step 4
#Dropping un-wanted columns
df['Route'].value_counts()

df['Additional_Info'].value_counts()

#route can be used to find the total stops but as we have the total stops as seperate column we can actually drop route
#dropping Additional_info because for most of the flights there is no info i.e more than 80%.

df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

df.head()

#step:5
#changing column name Dep_time to Departure_time for understanding what it is
# It is step for feature engineering
df = df.rename(columns={'Dep_Time': 'Departure_Time'})

df.head()

df.info()

#Step:6 Converting Stops from categorical Data to corresponding values i.e Non-stop to 0, 1 stop to 1 , 2 stops to 2 , 3 stops to 3, and 4 stops to 4
df1=df.copy()
df1['Total_Stops'].value_counts()

df1.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

df1.head()

df.head()

#Step:7
#Converting data types of , Departure_Time, Arrival_Time
df1["Date_of_Journey"] = pd.to_datetime(df1.Date_of_Journey)
df1["Departure_Time"] = pd.to_datetime(df1.Departure_Time)
df1["Arrival_Time"] = pd.to_datetime(df1.Arrival_Time)

df1.info()

df1.head()

#Step:8
#To standardize the column in the datset we are now going
#to seperate the hours and minutes in departure and times times
#and write them into new columns named as departure_hours,departure_minutes,
#arrival_hour,and arrival_minutes
#And we are also divide the date, month and year of journet seperately
#Extracting features from journet date, Arrival time, Destination time(Feature Engineering).
df1['Depature_Hours'] = df1['Departure_Time'].dt.hour
df1['Depature_Minutes'] = df1['Departure_Time'].dt.minute
df1['Arrival_Hours'] = df1['Arrival_Time'].dt.hour
df1['Arrival_Minutes'] = df1['Arrival_Time'].dt.minute
df1['Journey_Day']=df1['Date_of_Journey'].dt.day
df1['Journey_Month']=df1['Date_of_Journey'].dt.month

df1.info()

df1.head()

#Now we can drop Date_of_Journey, Departure_Time,Arrival_Time

df1=df1.drop('Date_of_Journey',axis=1)
df1=df1.drop('Departure_Time',axis=1)
df1=df1.drop('Arrival_Time',axis=1)

df1.head()

#Step:9
#Coverting duration into minutes for consistency
import re
def convert_to_minutes(duration):
    total_minutes = 0
    # Extract hours and minutes using regular expressions
    hours= re.search(r'(\d+)h', duration)
    minutes= re.search(r'(\d+)m', duration)

    # Add hours and minutes to total minutes
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))

    return total_minutes
    print(total_minutes)


# Apply the function to the 'Duration' column
df1['Duration_Minutes'] = df1['Duration'].apply(convert_to_minutes)

df1=df1.drop('Duration',axis=1)

df1.head()

#After Converting Duration into minutes
df1.head()

#Step10
#Standardizing Case
df1['Airline'] = df1['Airline'].str.upper()
df1['Source'] = df1['Source'].str.upper()
df1['Destination'] = df1['Destination'].str.upper()

df1.head()

#Step11:
#Converting the categorical Data into labels using labelEncoder
df1['Airline'].value_counts()

df1['Source'].value_counts()

df1['Destination'].value_counts()

columns_to_encode = ['Airline', 'Source', 'Destination']

df1 = pd.get_dummies(df1, columns=columns_to_encode)

df1.head()

df1.head()

"""Exploratory Data Analysis"""

#Step:1
#Exploring the correlations between the features
#We can how features are correlated to each other and how it effects the price.
corr_matrix = df1.corr()
plt.figure(figsize=(26, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix between the features')
plt.show()

#Step:2
#representing number of Airlines in bar graph
plt.figure(figsize=(24,10))
sns.countplot(x="Airline",data=df)

#step:3
#Percentage distribution of source and destination
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['Source'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution of Source')

# Create a pie chart for Category2
plt.subplot(1, 2, 2)
df['Destination'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution of Destination')

plt.tight_layout()
plt.show()

#Step:4
#Exploring relation between two numerical variables
plt.scatter(df1['Duration_Minutes'],df1['Price'],color='blue')
plt.xlabel('Duration')
plt.ylabel('Price')
plt.show()

#Step:5
#visualize how the price is distributed
plt.figure(figsize=(10, 6))
plt.hist(df1['Price'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#Step:6
#comparing the prices across the Source and destination
plt.figure(figsize=(12, 6))
sns.pointplot(x='Source', y='Price', hue='Destination', data=df, palette='viridis')
plt.title('Price Comparison across Source and Destination')
plt.xlabel('Source')
plt.ylabel('Price')
plt.show()

#Step:7
#Line Graph between the price and airline
plt.figure(figsize=(12, 6))
sns.pointplot(x='Airline', y='Price', data=df, color='blue', markers='1', linestyles='-')
plt.title('Price Comparison across Airline')
plt.xlabel('Airline')
plt.ylabel('Price')
plt.show()

#Step:8
#Analyzing probability density function for a numerical variable i.e price using kde plots
sns.kdeplot(df['Price'],fill=True,color='red')

#Step:9
#Understanding the cumulative distribution function for price
sns.ecdfplot(df['Price'],stat='proportion',color='skyblue')

#Step:10
#visualizing top 3 airlines using bar chart
top=df['Airline'].value_counts().nlargest(3)
top.plot(kind='bar',color='green')

#Step:11
#Distribution comparision between price and source
sns.boxenplot(x='Source',y='Price',data=df)

"""DIC PROJECT PHASE-2"""

df1.shape

df1.columns

X= df1.loc[:, ['Total_Stops',  'Depature_Minutes',
       'Arrival_Hours', 'Arrival_Minutes', 'Journey_Day', 'Journey_Month',
       'Duration_Minutes', 'Airline_AIR ASIA', 'Airline_AIR INDIA',
       'Airline_GOAIR', 'Airline_INDIGO', 'Airline_JET AIRWAYS',
       'Airline_MULTIPLE CARRIERS',
       'Airline_MULTIPLE CARRIERS PREMIUM ECONOMY', 'Airline_SPICEJET',
       'Airline_TRUJET', 'Airline_VISTARA', 'Airline_VISTARA PREMIUM ECONOMY',
       'Source_BANGLORE', 'Source_CHENNAI', 'Source_DELHI', 'Source_KOLKATA',
       'Source_MUMBAI', 'Destination_BANGLORE', 'Destination_COCHIN',
       'Destination_DELHI', 'Destination_HYDERABAD', 'Destination_KOLKATA',
       'Destination_NEW DELHI']]

Y = df1.iloc[:, 1]



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred= linear.predict(X_test)

#Linear Regression
from sklearn import metrics
print("Linear Regression")
print('MAE:', metrics.mean_absolute_error(y_test, linear_pred))
print('MSE:', metrics.mean_squared_error(y_test, linear_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linear_pred)))
print("R Square:",metrics.r2_score(y_test, linear_pred))

plt.title("Linear Regression")
sns.distplot(y_test-linear_pred)
plt.show()

#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
KNN = KNeighborsRegressor()
KNN.fit(X_train, y_train)
KNN_pred= KNN.predict(X_test)

print("KNN Regressor")
print('MAE:', metrics.mean_absolute_error(y_test, KNN_pred))
print('MSE:', metrics.mean_squared_error(y_test, KNN_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, KNN_pred)))
print("R Square:",metrics.r2_score(y_test, KNN_pred))

plt.title("KNeighborsRegressor")
sns.distplot(y_test-KNN_pred)
plt.show()

#SVM
from sklearn.svm import SVR
SVM = SVR(kernel='linear')
SVM.fit(X_train, y_train)
SVM_pred= SVM.predict(X_test)

print("SVR")
print('MAE:', metrics.mean_absolute_error(y_test, SVM_pred))
print('MSE:', metrics.mean_squared_error(y_test, SVM_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, SVM_pred)))
print("R Square:",metrics.r2_score(y_test, SVM_pred))

plt.title("SVR")
sns.distplot(y_test-SVM_pred)
plt.show()

#Random Forest
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train, y_train)
RF_pred= RF.predict(X_test)

print("Random Forest")
print('MAE:', metrics.mean_absolute_error(y_test, RF_pred))
print('MSE:', metrics.mean_squared_error(y_test, RF_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, RF_pred)))
print("R Square:",metrics.r2_score(y_test, RF_pred))

plt.title("RandomForestRegressor")
sns.distplot(y_test-RF_pred)
plt.show()

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
DT= DecisionTreeRegressor()
DT.fit(X_train, y_train)
DT_pred= DT.predict(X_test)

print("Decision Tree")
print('MAE:', metrics.mean_absolute_error(y_test, DT_pred))
print('MSE:', metrics.mean_squared_error(y_test, DT_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, DT_pred)))
print("R Square:",metrics.r2_score(y_test, DT_pred))

plt.title("DecisionTreeRegressor")
sns.distplot(y_test-DT_pred)
plt.show()

#XGBOOST
import xgboost as xgb
XG = xgb.XGBRegressor()
XG.fit(X_train, y_train)
XG_pred= XG.predict(X_test)

print("XG BOOST")
print('MAE:', metrics.mean_absolute_error(y_test, XG_pred))
print('MSE:', metrics.mean_squared_error(y_test, XG_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, XG_pred)))
print("R Square:",metrics.r2_score(y_test, XG_pred))

plt.title("XGBoostRegressor")
sns.distplot(y_test-XG_pred)
plt.show()

#GradientBoost
from sklearn.ensemble import GradientBoostingRegressor
GB= GradientBoostingRegressor()
GB.fit(X_train, y_train)
GB_pred= GB.predict(X_test)

print("Gradient BOOST")
print('MAE:', metrics.mean_absolute_error(y_test, GB_pred))
print('MSE:', metrics.mean_squared_error(y_test, GB_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, GB_pred)))
print("R Square:",metrics.r2_score(y_test, GB_pred))

plt.title("GradientBoostRegressor")
sns.distplot(y_test-GB_pred)
plt.show()

#AdaBoost
from sklearn.ensemble import AdaBoostRegressor
adab = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
adab.fit(X_train, y_train)
ada_pred = adab.predict(X_test)
print("ADA BOOST")
print('MAE:', metrics.mean_absolute_error(y_test, ada_pred))
print('MSE:', metrics.mean_squared_error(y_test, ada_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ada_pred)))
print("R Square:",metrics.r2_score(y_test, ada_pred))

plt.title("AdaBoostRegressor")
sns.distplot(y_test-ada_pred)
plt.show()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 2, figsize=(15, 15))
fig.suptitle('Model Predictions vs. Actual Values')
def plot_model(ax, title, y_test, y_pred):
    ax.scatter(y_test, y_pred, color='blue', alpha=0.3, label='Predicted')
    ax.plot(y_test, y_test, color='red', linewidth=2, label='Actual')
    ax.set_title(title)
    ax.set_xlabel('Actual Flight Price')
    ax.set_ylabel('Predicted Flight Price')
    ax.legend()
# Graphical presentation each models prediction
plot_model(axes[0, 0], 'Linear Regression', y_test, linear_pred)
plot_model(axes[0, 1], 'KNN', y_test, KNN_pred)
plot_model(axes[1, 0], 'SVR', y_test, SVM_pred)
plot_model(axes[1, 1], 'Random Forest Regression', y_test, RF_pred)
plot_model(axes[2, 0], 'Decision Tree Regression', y_test, DT_pred)
plot_model(axes[2, 1], 'XG Boost', y_test, XG_pred)
plot_model(axes[3, 0], 'Gradient Boost', y_test, GB_pred)
plot_model(axes[3, 1], 'ADA Boost', y_test, ada_pred)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#creating pkl file
pickle.dump(XG, open("model.pkl", "wb"))