import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.neighbors import KNeighborsClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

#drive.mount('/content/drive')

#ls

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Data Credit Risk

dataset = pd.read_csv('credit_risk_dataset.csv')
dataset

"""## features Selection """

plt.figure(figsize = (15,7))

sns.heatmap(dataset.assign(person_home_ownership = dataset.person_home_ownership.astype('category').cat.codes,
                           loan_intent = dataset.loan_intent.astype('category').cat.codes,
                           loan_grade = dataset.loan_grade.astype('category').cat.codes,
                           cb_person_default_on_file = dataset.cb_person_default_on_file.astype('category').cat.codes).corr(),
            annot = True, cmap ='RdYlGn', vmin = -1, vmax = 1, linewidths = 0.5)


"""Drop features"""

features_drop = ['cb_person_cred_hist_length', 'cb_person_default_on_file','loan_percent_income', 'loan_int_rate',]

for col in features_drop:
  dataset = dataset.drop(col,axis=1)

"""dropna Values and unknow data"""

dataset = dataset.dropna()
dataset = dataset.drop(dataset[dataset['person_emp_length'] == 123].index)
dataset = dataset.drop(dataset[(dataset['person_age'] == 144) | (dataset['person_age'] == 123)].index)



"""# Machine Learning

## Outlier
"""

dataset

def plot_boxplot(dataset):
  column_outlier = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']

  for col in column_outlier:
    sns.set(style = 'whitegrid')
    plt.figure(figsize = (10,6))
    sns.boxplot(x=dataset[col])
    plt.title(f'Boxplot Outlier {col}')
    

def remove_outlier(dataset):
  column_outlier = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']

  for col in column_outlier:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (0.4 * IQR)

    dataset = dataset[(dataset[col] > lower_bound) & (dataset[col] < upper_bound)]

    return dataset

plot_boxplot(dataset)

dataset = remove_outlier(dataset)

"""## Encode and Normalization"""

dataset['loan_grade']  = dataset['loan_grade'].replace(['D','E','F','G'], 'D')
dataset

dataset = dataset.dropna()

dataset = dataset.sort_values('loan_grade')
# encode kategorikal features
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade']
# person_home_ownership = {rent : 3, own : 2, mortgage : 0, other : 1}
# loan_intet = {Debtconsolidation : 0, Venture : 5, Medical : 3, Education : 1, Homeimprovement : 2, Personal : 4}
#loan_grade = {A : 0, B : 1, C : 2, D : 3, E : 4, F : 5, G : 6}

for col in categorical_features:
  le = LabelEncoder()
  dataset[col] = le.fit_transform(dataset[col])

numerical_features = ['person_age', 'person_income', 'person_emp_length','loan_amnt']

for col in numerical_features:
  scaler  = MinMaxScaler()
  dataset[col] = scaler.fit_transform(dataset[[col]])

dataset

"""## Train test Split"""

# split dataset into train and test set
X = dataset.drop(['loan_status', 'loan_grade'], axis=1)
y_loan_status = dataset['loan_status']
y_loan_grade = dataset['loan_grade']
X_train, X_test, y_train_loan_status, y_test_loan_status = train_test_split(X, y_loan_status, test_size=0.2, random_state=42)
X_train, X_test, y_train_loan_grade, y_test_loan_grade = train_test_split(X, y_loan_grade, test_size=0.2, random_state=42)

"""## Resampling Imbalance Data

###Variabel Target Loan_status
"""

data_plot_y_loan = y_train_loan_status.reset_index().drop('index',axis = 1)
data_plot_y_loan = data_plot_y_loan.groupby('loan_status')['loan_status'].count().reset_index(name = 'count')
data_plot_y_loan

plot_loan_imbelance = px.bar(data_plot_y_loan, x = 'loan_status', y = 'count')


# Lakukan SMOTE pada subset training
# Lakukan Random Oversampling pada subset training
ros = RandomOverSampler(random_state=42)
X_train_resampled_loan, y_train_resampled_loan = ros.fit_resample(X_train, y_train_loan_status)
X_train_resampled_loan

data_plot_y_loan_resampled = y_train_resampled_loan.reset_index().drop('index', axis = 1)
data_plot_y_loan_resampled = data_plot_y_loan_resampled.groupby('loan_status')['loan_status'].count().reset_index(name = 'count')
data_plot_y_loan_resampled

"""###Variabel Target Grade"""

data_plot_y_grade = y_train_loan_grade.reset_index().drop('index',axis = 1)
data_plot_y_grade = data_plot_y_grade.groupby('loan_grade')['loan_grade'].count().reset_index(name = 'count')
data_plot_y_grade

plot_loan_imbelance = px.bar(data_plot_y_grade, x = 'loan_grade', y = 'count')


# Lakukan SMOTE pada subset training
# Lakukan Random Oversampling pada subset training
ros = RandomOverSampler(random_state=42)
X_train_resampled_grade, y_train_resampled_grade = ros.fit_resample(X_train, y_train_loan_grade)
X_train_resampled_grade

data_plot_y_grade_resampled = y_train_resampled_grade.reset_index().drop('index', axis = 1)
data_plot_y_grade_resampled = data_plot_y_grade_resampled.groupby('loan_grade')['loan_grade'].count().reset_index(name = 'count')
data_plot_y_grade_resampled
