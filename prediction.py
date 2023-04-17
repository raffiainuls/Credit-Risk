import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

def load_models():
    model_ann_loan = load_model('model/model_ann.h5')
    model_decision_loan = joblib.load('model/Decision_tree_model_loan.joblib')
    model_decision_grade = joblib.load('model/Decision_tree_model_grade.joblib')
    model_knn_loan = joblib.load('model/KNN_model_loan_status.joblib')
    model_knn_grade = joblib.load('model/KNN_model_loan_grade.joblib')
    model_xgboost_loan = joblib.load('model/Xgboost_model_loan_status.joblib')
    model_xgboost_grade = joblib.load('model/Xgboost_model_loan_grade.joblib')
    model_forest_loan = joblib.load('model/Random_forest_model_loan_status.joblib')
    model_forest_grade = joblib.load('model/Random_forest_model_loan_grade.joblib')
    
    return model_ann_loan, model_decision_loan, model_decision_grade, model_knn_loan, model_knn_grade, model_xgboost_loan, model_xgboost_grade, model_forest_loan, model_forest_grade


def input_preprocessing(data):
  home_owner = {'Rent' : 3, 'Own' : 2, 'Mortagage' : 0, 'Other' : 1}

  for category,value in home_owner.items():
    if (data['person_home_ownership'] == category).all():
      data['person_home_ownership'] = data['person_home_ownership'].map(home_owner)
      data.loc[0, 'person_home_ownership'] = value
      break
  
  loan_intent = { 'Debtconsolidation' : 0,  'Venture' : 5, 'Medical' : 3, 'Education' : 1, 'Home improvement' : 2, 'Personal' : 4}

  for category, value in loan_intent.items():
    if(data['loan_intent'] == category).all():
      data['loan_intent'] = data['loan_intent'].map(loan_intent)
      data.loc[0,'loan_intent'] = value
      break

  min_age = 20
  max_age = 32
  min_income = 4080
  max_income = 1200000	
  min_emp_length = 0
  max_emp_length = 17
  min_loan_amnt = 500
  max_loan_amnt = 35000

  data['person_age'] = (data['person_age'] - min_age) / (max_age - min_age)
  data['person_income'] = (data['person_income'] - min_income) / (max_income - min_income)
  data['person_emp_length'] = (data['person_emp_length'] - min_emp_length) / (max_emp_length - min_emp_length) 
  data['loan_amnt'] = (data['loan_amnt'] - min_loan_amnt) / (max_loan_amnt - min_loan_amnt)

  return data



def prediction(person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_amnt):
  data = pd.DataFrame({
      'person_age'            : [person_age],
      'person_income'         : [person_income],
      'person_home_ownership' : [person_home_ownership],
      'person_emp_length'     : [person_emp_length],
      'loan_intent'           : [loan_intent],
      'loan_amnt'             : [loan_amnt]    

  })

  data = input_preprocessing(data)

  model_ann_loan, model_decision_loan, model_decision_grade, model_knn_loan, model_knn_grade, model_xgboost_loan, model_xgboost_grade, model_forest_loan, model_forest_grade = load_models()

  #predict for loan_status

  y_pred_loan = (model_ann_loan.predict(data) + model_decision_loan.predict(data) + model_xgboost_loan.predict(data) + model_forest_loan.predict(data) )/4
  y_pred_loan = np.round(y_pred_loan, 0)

  #predict for loan_grade

  y_pred_grade = (model_decision_grade.predict(data) + model_knn_grade.predict(data) + model_xgboost_grade.predict(data) + model_forest_grade.predict(data)) / 4
  y_pred_grade = np.round(y_pred_grade, 0)
  
  # result predict
  
  if  y_pred_loan == [0] and y_pred_grade == [0]:
    return 'Anda Layak'
  elif y_pred_loan == [0] and y_pred_grade == [1]:
    return 'Anda Layak'
  elif y_pred_loan == [0] and y_pred_grade == [2]:
    return 'Anda Layak'
  elif y_pred_loan == [0] and y_pred_grade == [3]:
    return 'Anda Belum Layak'
  elif y_pred_loan == [1] and y_pred_grade == [0]:
    return 'Anda Tidak Layak'
  elif y_pred_loan == [1] and y_pred_grade == [1]:
    return 'Anda Tidak Layak'
  elif y_pred_loan == [1] and y_pred_grade == [2]:
    return 'Anda Tidak Layak'
  elif y_pred_loan == [1] and y_pred_grade == [3]:
    return 'Anda Tidak Layak'
  
