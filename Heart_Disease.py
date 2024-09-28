import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

##  01.Data Collection
df=pd.read_csv('hearts.csv')
# print(df)  ## Printing the DataFrame

## 02.Data Preprocessing
from sklearn.preprocessing import LabelEncoder  ## Label Encoding used to convert the labels into numbers(String to number conversion)
le=LabelEncoder()

df['Sex']=le.fit_transform(df['Sex'])
df['ChestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingECG']=le.fit_transform(df['RestingECG'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])

# print(df) ## Printing the DataFrame after preprocessing

X=df.drop(columns=['HeartDisease']) #Input 
y=df['HeartDisease'] #Output

# print("XXXX" ,X)
# print("YYYY" ,y)

## 03.Choosing the Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) #Splitting the data into training and testing data

#X_train - 80% of the input data
#X_test - 20% of the input data
#y_train - 80% of the output data
#y_test - 20% of the output data

# print("DF" , df.shape) ##Shape show the number of rows and columns
# print("X" , X_train.shape)
# print("y" , y_train.shape)
# print("X_test" , X_test.shape)
# print("y_test" , y_test.shape)

## 04.Model Training
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)

## 05.Model Evaluation

y_pred=model.predict(X_test)

print("y_pred" , y_pred)
print("y_test" , y_test)

from sklearn.metrics import accuracy_score  # metrics is sub module of sklearn
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of the model is" , accuracy)
