# -*- coding: utf-8 -*-
"""Credit Risk Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rdzhubOgRkQDec57DqNt6RGjwbC0sR9v

# Preprocessing

## Import library and load data
"""


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

"""# Analysist Data

## Analysist Features Person Age
"""

dataset.sort_values('person_age', ascending = False)

sns.set(style = 'whitegrid')
plt.figure(figsize = (10,6))
sns.boxplot(x = dataset['person_age'])
plt.title('Boxplot Distribusi Age')
plt.xlabel('Age')



def plot_dis(dataset):
  kde = gaussian_kde(dataset)
  x = np.linspace(dataset.min(), dataset.max())
  y = kde(x)

  plot_dis = px.histogram(x = dataset, nbins = 50, opacity = 0.5)

  plot_dis.add_trace(go.Scatter(x = x,
                                y = y,
                                name = 'Density',
                                yaxis = 'y2',
                                line = dict(color = 'red', width = 3)))
  plot_dis.update_layout(bargroupgap = 0.2)
  plot_dis.update_layout(yaxis2 = dict(title = 'Density',
                                       title_font = dict(family = 'arial'),
                                       overlaying = 'y', side = 'right'))
  return plot_dis
  

plot_dis = plot_dis(dataset['person_age'])



data_age = dataset

bins = [20,30,50,60,85]
labels = ['20-29', '30-49', '50-59','60-84']

data_age['age_group'] = pd.cut(dataset['person_age'], bins= bins, labels = labels)

data_age_loan = data_age.groupby('age_group').agg({'loan_amnt': 'mean'}).reset_index()

plot_age_loan = px.bar(data_age_loan, x = 'age_group', y = 'loan_amnt')


data_age_annual = data_age.groupby('age_group').agg({'person_income': 'mean'}).reset_index()
plot_age_annual = px.bar(data_age_annual, x = 'age_group', y = 'person_income')

data_age

data_age_loan_status = data_age.groupby(['loan_status', 'age_group'])['age_group'].count().reset_index(name = 'count')
data_age_loan_status

# Membuat contoh data frame
# Create the bar chart
data_age_loan_status = data_age_loan_status.sort_values('count')
plot_age_loan_status = px.bar(data_age_loan_status, x = 'age_group', y = 'count', barmode = 'group', color = 'loan_status', template = 'ggplot2')


data_age_home = data_age.groupby(['age_group', 'person_home_ownership'])['person_home_ownership'].count().reset_index(name = 'count')
data_age_home

plot_age_home = px.bar(data_age_home, x = 'person_home_ownership', y = 'count', color = 'age_group', barmode = 'group')


data_age

"""## Analysist Person Home Ownership"""

data_home_ownership = dataset
data_home_ownership = dataset.groupby('person_home_ownership')['person_home_ownership'].count().reset_index(name = 'count')
data_home_ownership

plot_home_owner = px.bar(data_home_ownership, x= 'person_home_ownership', y = 'count', color = 'person_home_ownership')


data_home_income = dataset.groupby('person_home_ownership')['person_income'].mean().reset_index()
data_home_income

plot_home_income = px.bar(data_home_income, x= 'person_home_ownership', y = 'person_income', color = 'person_home_ownership')


data_home_loan = dataset.groupby('person_home_ownership')['loan_amnt'].mean().reset_index()
data_home_loan

sns.catplot(data = dataset, kind = 'bar', x = 'person_home_ownership', y = 'person_income', hue = 'loan_status')

data_home_loan_status = dataset.groupby(['person_home_ownership', 'loan_status'])['loan_status'].count().reset_index(name = 'count')
data_home_loan_status

def catplot_home_owner_loan(dataset):
  plt.figure(figsize = (12,12))
  sns.catplot(data= dataset, kind= 'violin', x = 'person_home_ownership', y = 'loan_amnt', hue= 'loan_status',split = True)
  

catplot_home_owner_loan = catplot_home_owner_loan(dataset)

"""##Analysist Loan Intent"""



plt.figure(figsize = (12,15))
ax =sns.catplot(data = dataset, x = 'loan_intent', y  = 'loan_amnt', hue = 'loan_status',kind = 'bar')

# rotate x-axis tick labels
plt.xticks(rotation=90)

"""## Analysist Loan Grade"""

data_grade_loan = dataset.groupby(['loan_grade','loan_status'])['loan_amnt'].mean().reset_index()
data_grade_loan['loan_status']  = data_grade_loan['loan_status'].replace([1], 'Charge Off')
data_grade_loan['loan_status']  = data_grade_loan['loan_status'].replace([0], 'Fully Pain')
data_grade_loan

plot_grade_loan = px.bar(data_grade_loan, x = 'loan_grade', y = 'loan_amnt', color = 'loan_status', barmode = 'group')


dataset

data_grade_age = dataset.groupby(['loan_grade', 'age_group'])['age_group'].count().reset_index(name = 'count')
data_grade_age

plot_grade_age = px.bar(data_grade_age, x = 'loan_grade', y = 'count', color = 'age_group', barmode = 'group')


data_grade_income_status = dataset.groupby(['loan_grade', 'loan_status'])['person_income'].mean().reset_index()
data_grade_income_status

plot_grade_income_status = px.bar(data_grade_income_status,x = 'loan_grade', y = 'person_income', color = 'loan_status', barmode = 'group')


data_grade_home = dataset.groupby(['loan_grade', 'person_home_ownership'])['person_home_ownership'].count().reset_index(name = 'count')
data_grade_home

plot_grade_home = px.bar(data_grade_home, x = 'person_home_ownership', y = 'count', color = 'loan_grade', barmode = 'group')

data_income = dataset.groupby('loan_status')['person_income'].mean().reset_index()
data_income['loan_status']  = data_income['loan_status'].replace([1], 'Charge Off')
data_income['loan_status']  = data_income['loan_status'].replace([0], 'Fully Pain')
plot_income = px.bar(data_income, x = 'person_income', y = 'loan_status', color = 'loan_status')

"""## Analysist Loan_status"""

dataset
def kde_plot(dataset):
  plt.figure(figsize = (15,7))

  ax = sns.kdeplot(data = dataset, x = 'loan_amnt', hue = 'loan_status')
  ax.set_xlabel('Loan_amnt', color='white')
  ax.set_ylabel('Count', color='white')
  ax.set_xticklabels(ax.get_xticklabels(), color='white')
  ax.set_yticklabels(ax.get_yticklabels(), color='white')
  ax.tick_params(axis='x', colors='white')
  ax.tick_params(axis='y', colors='white')
  
  
  

kde_plot = kde_plot(dataset)

data_loan_status = dataset.groupby('loan_status')['loan_status'].count().reset_index(name = 'count')
data_loan_status

# membuat rentang kelompok loan_amnt
bins = [0, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 25000, 30000, 35000 ]

# memberi nama pada setiap rentang kelompok
labels = ['1000', '2000', '3000', '4000', '5000', '7000', '10000', '25000', '30000', '35000']


data = dataset
data['loan_amnt_range'] = pd.cut(data['loan_amnt'], bins = bins, labels = labels)

data = data.groupby(['loan_amnt_range', 'loan_status'])['loan_amnt_range'].count().reset_index(name = 'count')
data





plot_line_loan = px.line(data,x = 'loan_amnt_range', y = 'count',color = 'loan_status' ,markers = True)

plot_line_loan.update_traces(name='Fully Paid', selector=dict(name='0'))
plot_line_loan.update_traces(name='Charge Off', selector=dict(name='1'))


pie_loan_status = px.pie(data_loan_status, values = 'count' ,names = 'loan_status', color= 'loan_status')
pie_loan_status.update_traces(name='Fully Paid', selector=dict(name='0'))
pie_loan_status.update_traces(name='Charge Off', selector=dict(name='1'))


