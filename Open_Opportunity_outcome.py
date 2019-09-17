import pandas as pd
from sklearn.metrics import accuracy_score

#read Data
data = pd.read_csv(r"Data/Opportunity.csv" , encoding="ISO-8859-1")
original_data = data.copy()
data.head()

#lets Explore the data
columns = data.columns
desc = data.describe()

#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/ data.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#here we can see that lot of columns are duplicate
#and lot of column are empty
#so we will remove them

#drop duplicates
#columns
data = data.T.drop_duplicates().T
#row
data = data.drop_duplicates(keep=False) 
#drop empty columns
data = data.dropna(how = 'all' , axis= 1)

#drop column with same value for each row
nunique = data.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
data = data.drop(cols_to_drop, axis = 1)

desc = data.describe()

#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/ data.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


################################################################################################
#drop column if missing data is more then 50%
colWithMissingData = missing_data.loc[missing_data['Percent'] >= 50]
data = data.drop(colWithMissingData.index, axis = 1)

#dropping all the ID columns, bcoz its seems it may not be much useful here
idColumns = [col for col in data.columns if 'id' in col.lower()]
data = data.drop(idColumns, axis = 1)

#check how many date columns are there
#we can find the duration between this dates
#and use the turn around time in prediction
#currently dont know the significance so dropping them
dateColumns = [col for col in data.columns if 'date' in col.lower()]
#turnAroundTime = data['CloseDate'] - data['CreatedDate']
data = data.drop(dateColumns, axis = 1)


#fiscal column is further divided into fiscal quarter and fiscal year
#so we will drop that too
data = data.drop('Fiscal', axis = 1)

#fill missing values
total = data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/ data.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

#"Spigit_Description__c" contain desciption + number, so dropping it
data = data.drop('Spigit_Description__c', axis = 1)
data = data.drop('NextStep', axis = 1)
data = data.drop('Spigit_Product_Type__c', axis = 1)
data = data.drop('Spigit_Rep_Assesment__c', axis = 1)
data = data.drop('Spigit_Opportunity_Number__c', axis = 1)


#filling the missing values
data['Spigit_Channel_Type__c'] = data['Spigit_Channel_Type__c'].fillna(data['Spigit_Channel_Type__c'].mode()[0])
data['Spigit_Community_Type__c'] = data['Spigit_Community_Type__c'].fillna(data['Spigit_Community_Type__c'].mode()[0])
data['Spigit_Stage__c'] = data['Spigit_Stage__c'].fillna(data['Spigit_Stage__c'].mode()[0])
data['Roll_Up_Product_Family__c'] = data['Roll_Up_Product_Family__c'].fillna(data['Roll_Up_Product_Family__c'].mode()[0])
data['roll_up_product_category__c'] = data['roll_up_product_category__c'].fillna(data['roll_up_product_category__c'].mode()[0])

data['TotalOpportunityQuantity'] = data['TotalOpportunityQuantity'].fillna(data['TotalOpportunityQuantity'].mean())
data['ExpectedRevenue'] = data['ExpectedRevenue'].fillna(data['ExpectedRevenue'].mean())
data['Amount'] = data['Amount'].fillna(data['Amount'].mean())
data['Total_Spigit_Consulting_old__c'] = data['Total_Spigit_Consulting_old__c'].fillna(data['Total_Spigit_Consulting_old__c'].mean())
data['Total_Spigit_Subscriptions_old__c'] = data['Total_Spigit_Subscriptions_old__c'].fillna(data['Total_Spigit_Subscriptions_old__c'].mean())

data['Sales_Channel__c'] = data['Sales_Channel__c'].fillna(data['Sales_Channel__c'].mode()[0])
data['PushCount__c'] = data['PushCount__c'].fillna(data['PushCount__c'].mode()[0])


##correaltion matrix
#corr = OneHotCodedData.corr()
#corr.style.background_gradient(cmap = 'coolwarm')

################################################################################################
#dividing the data based on open and closed deals
#predict the Open Opportunity (Deals) outcome based on the historical data of closed opportunities.
train = data.loc[data['IsClosed'] == True]
test = data.loc[data['IsClosed'] == False]

#output var to be predicted
y = train['IsWon']
#label encoding Y
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#contain only false - just to check, the value we are trying to predict
y1 = test['IsWon']

X = train.drop('IsWon', axis = 1)
test = test.drop('IsWon', axis = 1)

#one hot code the categorical vars
categoricalVar = X.select_dtypes(exclude=['int', 'float']).columns
OneHotCodedData = pd.get_dummies(X, columns = categoricalVar, drop_first=True)


#split Train Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(OneHotCodedData, y, test_size=0.25)

#Logistics Regression
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state = 42)
lr_clf = lr_clf.fit(X_train, y_train)
prediction = lr_clf.predict(X_test)

lr_score = accuracy_score(prediction, y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, prediction)
#########################################################################################
##Random Forest
#from sklearn.ensemble import RandomForestClassifier
#rf_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
#rf_clf = rf_clf.fit(X_train, y_train)
#prediction = rf_clf.predict(X_test)
#
#rf_score = accuracy_score(prediction, y_test)
#
#
##XGboost
#from xgboost import XGBClassifier
#xg_clf = XGBClassifier()
#xg_clf.fit(X_train, y_train)
#prediction = xg_clf.predict(X_test)
#
#xg_score = accuracy_score(prediction, y_test)

##########################################################################################
#saving the files
trainData = X_train.copy()
validationData = X_test.copy()

trainData = trainData.reset_index(drop=True)
validationData = validationData.reset_index(drop=True)

#adding the output var to file
train['IsWon'] = pd.Series(le.inverse_transform(y_train))
validationData['IsWon'] = pd.Series(le.inverse_transform(prediction))

#saving the test data set
train.to_csv("Data/train.csv", index=False)
#saving the test data set
validationData.to_csv("Data/validation.csv", index=False)


##########################################################################################
data = data.drop('IsWon', axis = 1)
OneHotCodedFullData = pd.get_dummies(data, columns = categoricalVar, drop_first=True)
IsClosedCol = [col for col in OneHotCodedFullData.columns if 'iswon' in col.lower()]

#dividing after encoding
train = OneHotCodedFullData.loc[OneHotCodedFullData['IsClosed_True'] == 1]
test = OneHotCodedFullData.loc[OneHotCodedFullData['IsClosed_True'] == 0]

#predicting on Test Data
lr_clf = lr_clf.fit(train, y)
prediction_final = lr_clf.predict(test)


#saving the test data set
test['IsWon'] = pd.Series(le.inverse_transform(prediction_final))
test.to_csv("Data/test.csv", index=False)






