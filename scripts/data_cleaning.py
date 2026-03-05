import pandas as pd 
import numpy as np 
 
def clean_data(df): 
    """Clean Titanic dataset""" 
    data = df.copy() 
 
    # Fill Age by Pclass & Sex 
    data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median())) 
 
    # Fill Embarked with mode 
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) 
 
    # Create HasCabin instead of using Cabin 
    data['HasCabin'] = data['Cabin'].notna().astype(int) 
 
    # Fill Fare by Pclass 
    data['Fare'] = data.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median())) 
 
    return data 
 
if __name__ == "__main__": 
    train = pd.read_csv('data/train.csv') 
    cleaned = clean_data(train) 
    print("? Data cleaning complete!") 
    print(f"Missing Age values: {cleaned['Age'].isnull().sum()}") 
    print(f"Missing Embarked values: {cleaned['Embarked'].isnull().sum()}") 
    print(cleaned.head()) 
