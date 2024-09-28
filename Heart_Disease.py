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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#X_train - 80% of the input data
#X_test - 20% of the input data
#y_train - 80% of the output data
#y_test - 20% of the output data

