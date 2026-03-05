import pandas as pd 
import numpy as np 
from data_cleaning import clean_data 
 
def engineer_features(df): 
    """Create new features""" 
    data = df.copy() 
 
    # Extract Title from Name 
     data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
 
    # Group rare titles 
    rare = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'] 
    data['Title'] = data['Title'].replace(rare, 'Rare') 
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss') 
    data['Title'] = data['Title'].replace('Mme', 'Mrs') 
 
    # Family features 
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1 
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int) 
 
    # Log transform Fare 
    data['FareLog'] = np.log1p(data['Fare']) 
 
    # Age bins 
    data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Elder']) 
 
    # Drop unused columns 
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1, errors='ignore') 
 
    return data 
 
if __name__ == "__main__": 
    train = pd.read_csv('data/train.csv') 
    cleaned = clean_data(train) 
    engineered = engineer_features(cleaned) 
    print("? Feature engineering complete!") 
    print(engineered[['Title', 'FamilySize', 'IsAlone', 'FareLog', 'AgeBin']].head()) 
