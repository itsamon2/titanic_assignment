# Titanic Survival Prediction - Assignment 2

## Overview

Predict passenger survival using feature engineering and selection techniques on the Titanic dataset.

## Dataset

- **Source:** Kaggle - Titanic: Machine Learning from Disaster
- **Link:** https://www.kaggle.com/c/titanic
- **Files:**
  - `data/train.csv` - Training data with target variable `Survived` (891 passengers, 12 columns)
  - `data/test.csv` - Test data for final predictions (optional)

## Project Structure

```
titanic_assignment/
├── data/
│   ├── train.csv              # Original training dataset
│   └── test.csv               # Test dataset for predictions
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb  # Main analysis notebook with visualizations
├── scripts/
│   ├── data_cleaning.py       # Data preprocessing and missing value handling
│   ├── feature_engineering.py # Feature creation and transformations
│   └── feature_selection.py   # Feature selection methods and justification
├── README.md
├── requirements.txt
└── feature_selection_report.txt  # Detailed feature selection justification
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Scripts (Command Line)
```bash
# Part 1: Data Cleaning (10 Marks)
python scripts/data_cleaning.py

# Part 2: Feature Engineering (30 Marks)
python scripts/feature_engineering.py

# Part 3: Feature Selection (10 Marks)
python scripts/feature_selection.py
```

### 3. Jupyter Notebook
```bash
jupyter notebook
# Open: notebooks/Titanic_Feature_Engineering.ipynb
# Run cells with Shift + Enter
```

---

## Part 1: Data Cleaning

### Missing Value Handling
| Column | Missing | Strategy |
|--------|---------|----------|
| Age | 177 (19.9%) | Filled with median by Pclass & Sex |
| Embarked | 2 (0.2%) | Filled with mode ('S') |
| Cabin | 687 (77.1%) | Converted to binary `HasCabin` feature |
| Fare | 0 | Filled with median by Pclass |

### Outlier Handling
- Detected outliers in `Fare` and `Age`
- Applied log transformation to `Fare` to reduce skewness (4.79 → 0.63)

### Data Consistency
- Fixed `Sex` values (male/female)
- No duplicates found
- Cleaned dataset saved as `train_cleaned.csv`

---

## Part 2: Feature Engineering

### 5 New Features Created

| Feature | Description | Formula | Impact |
|---------|-------------|---------|--------|
| `Title` | Extracted from Name (Mr/Mrs/Miss/Master/Rare) | `Name.str.extract(r' ([A-Za-z]+)\.')` | Social status proxy |
| `FamilySize` | Total family members on board | `SibSp + Parch + 1` | Family support indicator |
| `IsAlone` | Binary: traveling alone | `1 if FamilySize == 1 else 0` | Solo traveler flag |
| `FareLog` | Log-transformed fare | `log(Fare + 1)` | Reduced skewness |
| `HasCabin` | Binary: cabin number known | `1 if Cabin not null else 0` | Wealth proxy |

### Categorical Encoding
- One-hot encoded: `Sex`, `Embarked`, `Title`
- Ordinal: `Pclass` (1, 2, 3)
- Binned: `AgeBin` (Child/Teen/Adult/Middle/Elder)

### Feature Transformations
- **Log transform:** `Fare` → `FareLog` (skewness reduced from 4.79 to 0.63)
- **Standardization:** Applied for distance-based models

---

## Part 3: Feature Selection

### Three Methods Applied

#### 1. Correlation Analysis
- **Threshold:** 0.8
- **Dropped:** `FamilySize` (highly correlated with `SibSp`/`Parch`)
- **Reason:** `FamilySize = SibSp + Parch + 1` (redundant)

#### 2. Random Forest Feature Importance
| Rank | Feature | Importance | Decision |
|------|---------|------------|----------|
| 1 | **Sex** | 18.6% | ✅ Keep - strongest predictor |
| 2 | **Age** | 17.4% | ✅ Keep - "children first" policy |
| 3 | **FareLog** | 13.8% | ✅ Keep - wealth indicator |
| 4 | **Fare** | 14.0% | ❌ Drop - redundant with FareLog |
| 5 | **Title** | 11.0% | ✅ Keep - social status |
| 6 | **Pclass** | 5.9% | ✅ Keep - passenger class |
| 7 | **FamilySize** | 4.4% | ⚠️ Drop - correlated |
| 8 | **HasCabin** | 3.5% | ✅ Keep - wealth proxy |

#### 3. Recursive Feature Elimination (RFE)
- **Optimal features:** 9
- **Cross-validation:** 5-fold StratifiedKFold
- **Selected:** `['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'HasCabin', 'Title', 'FamilySize', 'FareLog']`

### Final Selected Features (8)

| Feature | Reason |
|---------|--------|
| **Sex** | Strongest predictor - gender policy |
| **Age** | Children prioritized for rescue |
| **FareLog** | Wealth indicator (log-transformed) |
| **Title** | Social status/age/gender proxy |
| **Pclass** | Passenger class - wealth indicator |
| **HasCabin** | Cabin availability - 1st class proxy |
| **FamilySize** | Family support vs burden |
| **Embarked** | Port of embarkation |

### Dropped Features
| Feature | Reason |
|---------|--------|
| `PassengerId` | ID only, no predictive value |
| `Name` | Extracted Title, dropped raw name |
| `Ticket` | Random alphanumeric strings |
| `Cabin` | 77% missing, converted to HasCabin |
| `Fare` | Replaced by FareLog (less skewed) |
| `SibSp`, `Parch` | Combined into FamilySize |
| `AgeBin` | Less informative than continuous Age |
| `IsAlone` | Redundant with FamilySize |

---

## Key Findings

1. **Sex is the strongest predictor** (18.6% importance) - reflects "women and children first" policy
2. **Age matters significantly** (17.4% importance) - children had higher survival rates
3. **Wealth indicators** (FareLog, Pclass, HasCabin) combined account for ~23% importance
4. **Title extraction** improved model by capturing social status and gender/age patterns
5. **Family size** has non-linear relationship - medium families (3-4) had best survival rates
6. **Log transformation** of Fare reduced skewness from 4.79 to 0.63

---

## Visualizations Included

- Survival overview (count and percentage)
- Gender analysis (survival rates by sex)
- Passenger class analysis
- Age distribution and survival by age groups
- Fare distribution (before/after log transform)
- Correlation heatmap
- Title extraction analysis
- Family size impact
- Feature importance ranking
- Summary dashboard

---

## Technologies Used

- Python 3.12
- pandas 3.0.1
- numpy 2.4.2
- scikit-learn 1.8.0
- matplotlib 3.10.8
- seaborn 0.13.2
- Jupyter Notebook

---

## Assignment Requirements Met

✅ **Part 1: Data Cleaning (10 Marks)**
- Missing value handling with justification
- Outlier detection and transformation
- Data consistency checks
- Cleaned dataset saved as `train_cleaned.csv`

✅ **Part 2: Feature Engineering (30 Marks)**
- 5 derived features created
- Categorical encoding (one-hot and ordinal)
- Interaction features
- Feature transformations (log, standardization)
- Visualizations to justify transformations

✅ **Part 3: Feature Selection (10 Marks)**
- Correlation analysis
- Random Forest feature importance
- Recursive Feature Elimination (RFE)
- Justification for kept/dropped features

✅ **GitHub Submission**
- Clean folder structure
- README.md with approach and findings
- requirements.txt
- Proper commit messages

---

## Author

**Amon Aiyabei Sawe**  
S13/02934/23  
April 2025
