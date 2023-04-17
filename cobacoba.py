import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk menampilkan grafik Confusion Matrix
def show_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5, cbar=False)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    st.pyplot(fig)

# Fungsi untuk menampilkan Classification Report
def show_classification_report(classification_rep):
    st.text_area('Classification Report', classification_rep, height=200)

# Ambil data input dari user
y_true = st.text_input('Masukkan nilai y_true (dalam format array atau list) : ')
y_pred = st.text_input('Masukkan nilai y_pred (dalam format array atau list) : ')

# Konversi string input menjadi array/list
y_true = list(map(int, y_true.split()))
y_pred = list(map(int, y_pred.split()))

# Hitung Confusion Matrix dan Classification Report
conf_matrix = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)

# Tampilkan Confusion Matrix dan Classification Report
show_confusion_matrix(conf_matrix)
show_classification_report(classification_rep)
