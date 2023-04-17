from prediction import load_models
from preprocessing import X_test, y_test_loan_status
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import numpy as np
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import joblib
import pandas as pd
from tensorflow.keras.models import load_model


def load_models():
    model_ann_loan = load_model('model/model_ann.h5')
    model_decision_loan = joblib.load('model/Decision_tree_model_loan.joblib')
    model_knn_loan = joblib.load('model/SVM_model_loan.joblib')
    model_xgboost_loan = joblib.load('model/Xgboost_model_loan_status.joblib')
    model_forest_loan = joblib.load('model/Random_forest_model_loan_status.joblib')

    return model_ann_loan, model_decision_loan, model_knn_loan, model_xgboost_loan, model_forest_loan

def prediction(X_test, y_test, model): 
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    accuracy = (accuracy_score(y_test,y_pred)*100)
    f1 = (f1_score(y_test,y_pred)*100)
    recall = (recall_score(y_test, y_pred)*100)
    precision  = (precision_score(y_test, y_pred)*100)

    return accuracy,f1, recall, precision

def confusion_matrixx(X_test, y_test,model):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    cm = confusion_matrix(y_test, y_pred)
    return cm




def performance_model(model, X_test = X_test, y_test = y_test_loan_status):

    model_ann_loan, model_decision_loan, model_knn_loan, model_xgboost_loan, model_forest_loan = load_models()

    if model == 'Artificial Neural Network':
        model = model_ann_loan
    elif model == 'Decision Tree':
        model = model_decision_loan
    elif model== 'Suport Vector Machine':
        model = model_knn_loan
    elif model == 'Xgboost':
        model = model_xgboost_loan
    elif model == 'Random Forest':
        model = model_forest_loan

    accuracy,f1,recall,precision = prediction(X_test, y_test, model)
    
    cm = confusion_matrixx(X_test, y_test,model)
    data_confusion = pd.Series({
    'True Positive'     : cm[0][0],
    'False Positive'    : cm[0][1],
    'False Negative'    : cm[1][0],
    'True Negative'     : cm[1][1] 
    })  

    

    return accuracy,f1,recall,precision, data_confusion,cm

