# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,2:-1].values
y = dataset.iloc[:, -1].values

from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

print("************************************************************************")

# Taking care of missing data so we must fit it
""" this fiting for for non_numerical data like 'Geography' and 'Gender'
SimpleImputer = da el class elly  besa3dna fe enna nhandel el missing data"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant')
imputer.fit(X[:, 0:2])
X[:, 0:2] = imputer.transform(X[:, 0:2])
# fiting for missing data in balance and tenure of client 
""" hena lama 3mlnaha b el 'mean' elnatega kanet 3alia 2wy 3n ba2y elntaig
 3lshan keda estkhdemna el 'meadian'"""
imputer = SimpleImputer(missing_values=0, strategy='most_frequent')
imputer.fit(X[:, 4:5])
X[:, 4:5] = imputer.transform(X[:, 4:5])

imputer = SimpleImputer(missing_values=0, strategy='median')
imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])
print(X)

print("*************************************************************************")
# Encoding categorical data
""" hena 3yzen n7ly el categorical data tb2a numbers 3lshan n3rf nt3amel m3aha so , e7na hn3mel encoding 
3la data 3lshan n7olha l numbers 3lshan n2der nt3amel ma3aha"""
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0]) # da el Encoder bta3 'Geoghraphy'
X[:,1]=labelencoder_X.fit_transform(X[:,1]) # # da el Encoder bta3 'Gender'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X) 
""""fe el7eta bta3t 'ct' de keda e7na 7olna el 'Geography' men categorical data l binary data
3lshan n2der nt3amel m3aha'  +=+ ...'ct' de ya3ny categorical """

# # Encoding the Dependent Variable ********************************************
"""el code da befahmna ezay n7ol el dependent variables men categorical l numerical or binary 
ya3ny lw 'Exited' kant yes w no he7olha l 0 , 1"""
# from sklearn.preprocessing import LabelEncoder
# labelencoder_y = LabelEncoder()
# y = le.fit_transform(y)
# print(y)

# Splitting the dataset into the Training set and Test set
"""e7na hena han2sem el data l 2 parts wa7ed l 'training', wa7ed l 'testing' 
 after we train the model in our dataset in training part 'X_train' ,'y_train' 
 we must to  test our model by giving it the 'X_test' and it predict the result
 which must be similar to 'y_test'
 so, we will compare the results of model with 'y_test' to ensure that model has a good accuracy """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature Scaling
""""""
from sklearn.preprocessing import StandardScaler
sc_x =  StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
"""e7na kan mmkn n3ml feature scaling l y-train laken he hena mesh mehtaga feature scaling 
kan heb2a ze keda 
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) """
# sc = StandardScaler()
# X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# X_test[:, 3:] = sc.transform(X_test[:, 3:])
# print(X_train)
# print(X_test)
#########################################################################################
# Support Vector Machine (SVM)

# fitting SVM to the training model 

from sklearn.svm import SVC
classifier = SVC( kernel ='rbf' , shrinking = True , C = 1.0 , degree = 3 , coef0 = 0.001 ,
gamma='scale' ,probability = True , random_state = 0)
classifier.fit(X_train,y_train)

# predicting the test set resultsof SVM

y_pred = classifier.predict(X_test)

# making the confusion matrix of SVM

"""de b2a elly bet3rfna el 'true predicted results' , 'false predicted results'"""

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm =  confusion_matrix(y_test , y_pred)

sns.heatmap(cm,center=True)
plt.show
#################################################
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred)*100

print(f"Accuracy SVM model equals : {accuracy}")

######################################################3



#الحمد لله 













