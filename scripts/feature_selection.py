import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import RFECV 
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import LabelEncoder 
from data_cleaning import clean_data 
from feature_engineering import engineer_features 
 
def encode_features(df): 
    data = df.copy() 
    le = LabelEncoder() 
    for col in ['Sex', 'Embarked', 'Title', 'AgeBin']: 
        if col in data.columns: 
            data[col] = le.fit_transform(data[col].astype(str)) 
    return data 
 
def correlation_analysis(X, threshold=0.8): 
    corr_matrix = X.corr().abs() 
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)] 
    return to_drop 
 
def get_feature_importance(X, y): 
    rf = RandomForestClassifier(n_estimators=100, random_state=42) 
    rf.fit(X, y) 
    importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False) 
    return importance 
 
def rfe_selection(X, y): 
    rf = RandomForestClassifier(n_estimators=100, random_state=42) 
    rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring='accuracy', min_features_to_select=5) 
    rfecv.fit(X, y) 
    selected = X.columns[rfecv.support_].tolist() 
    return selected, rfecv.n_features_ 
 
if __name__ == "__main__": 
    train = pd.read_csv('data/train.csv') 
    cleaned = clean_data(train) 
    engineered = engineer_features(cleaned) 
    X = engineered.drop(['Survived', 'PassengerId'], axis=1, errors='ignore') 
    y = engineered['Survived'] 
    X_encoded = encode_features(X) 
    print("=== Correlation Analysis ===") 
    print("Features to drop:", correlation_analysis(X_encoded)) 
    print("\\n=== Feature Importance ===") 
    print(get_feature_importance(X_encoded, y).head(10)) 
    print("\\n=== RFE Selection ===") 
    selected, n = rfe_selection(X_encoded, y) 
    print(f"Optimal features ({n}): {selected}") 
