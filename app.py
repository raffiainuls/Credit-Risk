import streamlit as st
from prediction import prediction
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from simple_data_analysist import plot_dis, plot_home_owner, plot_line_loan, pie_loan_status,plot_grade_loan,plot_income
from model_performance import performance_model
import seaborn as sns
import matplotlib.pyplot as plt


# Set page config
st.set_page_config(page_title="Page Title", page_icon=":guardsman:", layout="wide")

# Add margin between components
st.markdown("""<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>""",unsafe_allow_html=True)
st.markdown("""<style>div.row-widget.stRadio > div{margin-left:20px;margin-right:20px;}</style><br>""",unsafe_allow_html=True)



def main():
    # Buat dua halaman menggunakan st.sidebar
    st.sidebar.title("Navigation")
    pages = ["Analysist Data", "Performance Model", "App Credit Risk"]
    selection = st.sidebar.radio("Go to", pages)

    # Halaman Home
    if selection == "Analysist Data":
        
        header_image = Image.open("image/header.jpg")
        header_image = header_image.resize((1010, 400))
        st.markdown("<h1 style = 'text-align : center; color : white; font_size : 40 px; font-family : Arial'><b>Application Credit Risk<b></h1>", unsafe_allow_html= True)
        st.image(header_image,  use_column_width=False)
        st.markdown("------")
        st.markdown("Created by [Raffi Ainul Afif](linkedin.com/in/raffi-ainul-afif-9811a411b/)")
        
        #analysist age
        st.markdown("<h1 style = 'text-align : left; color : white; font-size : 20px; font-family : Arial'> <b>Data Analysist Umur Peminjam<b></h1>", unsafe_allow_html= True)
        paragraf_age = ''' <p style = 'font-size : 14px; font-family : Arial; color : white;'> Jika dilihat pada grafik ternyata peson yang melakukan pinjaman banyak dikuasai 
        oleh 22 sampai dengan 25 atau generasi milenials. Mungkin dengan perkembangan jaman dan teknologi membuat generasi milenials yang juga secara ekomi belum stabil ini dituntut untuk trendi dengan keadaan ekonomi yang suli, 
        dan memaksa mereka untuk melakukan pinjaman untuk memenuhi kebutuhan mereka'''
        st.markdown(paragraf_age, unsafe_allow_html= True)
        st.plotly_chart(plot_dis, use_container_width=True, height=500)

        #analysist home owner 
        st.markdown("<h1 style = 'text-align : left; color : white; font-size : 20px; font-family : Arial;' <b>Data Analysis Home Ownership<b></h1>", unsafe_allow_html= True)
        paragraf_home = '''<p style = 'font-size: 14px; font-family : family; color : white;'> Ternyata peminjam dengan kepemilikan rumah rental dan mortagage adalah yang paling banyak dan perminjam yang memiliki rumah sendiri justru sedikit. 
        Hal ini membuktikan bahwa jika dilihat dari kepemilikan rumah secara ekonomi memang orang yang sudah memiliki rumah sendiri ekonominya lebih baik dari peminjam dengna kepemilikan rumah rental dan mortagage, karena hal tersebut menyebabkan 
        banyak dari mereka melakukan peminjaman'''
        st.markdown(paragraf_home, unsafe_allow_html=True)
        st.plotly_chart(plot_home_owner,use_container_width= True, height = 500)

        #analysist loan_grade
        st.markdown("<h1 style = 'text-align : left; color : white; font-size : 20px; font-family : Arial;'<b>Data Analysist Loan Grade<b></h1>", unsafe_allow_html= True)
        paragraf_grade = '''<p style = 'font-size : 14px; font-family : Arial; color : white;'> Jika dilihat pada grafik Grade dibawah terlihat memang peminjam dengan grade G banyak yang tidak dapat membayar peminjamannya, tentunya hal seperti ini 
        perlu dihindari dan perlu identifikasi untuk menjegah peminjam yang gagal melakukan pembayaran.</p>'''
        st.markdown(paragraf_grade, unsafe_allow_html= True)
        st.plotly_chart(plot_grade_loan, use_container_width= True, height = 500)


        #analysist person_income
        st.markdown("<h1 style = 'text-align : left; color : white; font-size : 20px; font-family : Arial;'<b>Data Analysis Person Income<b></h1>", unsafe_allow_html= True)
        paragraf_income = ''' <p style = 'font-size: 14px; font-family : Arial; color : white;'> Peminjam yang dapat menulasi peminjaman dan tidak dapat menulasi peminjam karakteristiknya juga dapat dilihat dari income yang diperoleh jika dilihat dari grafik dibawah
        peminjam yang berhasil menulasi peminjamannya memiliki rata rata pendapatan pertahun lebih banyak dari pada yang gagal untuk menulasi peminjamannya. </p>'''
        st.markdown(paragraf_income, unsafe_allow_html= True)
        st.plotly_chart(plot_income, use_container_width=True, height = 500)

        #analysit loan
        st.markdown("<h1 style = 'text-align : left; color : white; font-size : 20px; font-family : Arial;'<b>Data Analysist Loan Amount<b></h1>", unsafe_allow_html=True)
        st.plotly_chart(plot_line_loan, use_container_width= True, height = 500)
        paragraf_loan = ''' <p style = 'font-size : 14px; font-family : Arial; color : white;'> Jika kita lihat pada line chart diatas ternyata baynyak peminjam yang meminjam diangka 10.000 sampai dengan 30.000'''
        st.markdown(paragraf_loan, unsafe_allow_html= True)

        st.markdown("<h1 style = 'text-align : left; color : white; font-size: 20px; font-family: Arial;'<b> Data Analysist Loan Status<b></h1>", unsafe_allow_html= True)
        paragraf_loan_status = ''' <p style = 'font-size : 14px; font-family : Arial; color : white;'> Dari data yang ada ada sekitar 20 persen dari peminjam yang gagal melakukan pembayaran. hal ini tentu saja merugikan perusahaan peminjam. Dari data yang ada kita bisa 
        mengindentifikasi peminjam yang gagal melakukan pembayaran. Dengan menggunakan bantuan teknologi Machine learning kita bisa membuat model untuk mengetahui peminjam yang berpotensi gagal melakukan peminjaman atau tidak dari karakteristik data yang telah ada. Saya telah
        mencoba beberapa model Machine learning seperti algoritma Random Forest, Decision Tree, Xgboost, KNN dan saya juga mencoba menggunakan algoritma deeplearning yaitu Atificial Neuron Network. Hasil dari Akurasi model dapat dilihat di 'Performance Model' dan untuk implementasi applikasinya dapat menuju ke page 'App Credit Risk'.</p> '''
        st.markdown(paragraf_loan_status, unsafe_allow_html= True)
        st.plotly_chart(pie_loan_status, use_container_width = True, height = 500)
        st.markdown("<h1 style = 'text-align : left; color : white; font-size: 14px; font-family: Arial;'<b><i> Disclimer :Karena waktu yang kurang dan device yang kurang mempuni wkwkk beberapa model akurasinya masih belum optimal, namun saya akan coba terus mengimprove akurasi model-model yang saya buat<i><b></h1>", unsafe_allow_html=True)
        


        

    # Halaman About
    elif selection == "Performance Model":
        st.title("Performance Model")
        option = st.selectbox(
            'Select Model',
            ('Random Forest', 'Decision Tree', 'Suport Vector Machine', 'Xgboost', 'Artificial Neural Network')
        )
        accuracy,f1,recall,precision, data_confusion, cm = performance_model(option)
# Create the dashboard layout
        with st.container():
            col1, col2, col3, col4= st.columns(4)
            with col1:
                st.write("<h1 style='font-align : center; font-size : 60; color :white'> Accuracy</h1>",unsafe_allow_html=True)
                st.write("<h1 style='font-align : center; font-size : 60; color :#2F58CD'>{:.2f}%</h1>".format(accuracy), unsafe_allow_html=True)

            with col2:
                st.write("<h1 style='font-align : center; font-size : 60; color :white'> F1 Score</h1>",unsafe_allow_html=True)
                st.write("<h1 style='font-align : center; font-size : 60; color :#2F58CD'>{:.2f}%</h1>".format(f1), unsafe_allow_html=True)
            with col3:
                st.write("<h1 style='font-align : center; font-size : 60; color :white'> Precision</h1>",unsafe_allow_html=True)
                st.write("<h1 style='font-align : center; font-size : 60; color :#2F58CD'>{:.2f}%</h1>".format(precision), unsafe_allow_html=True)
            with col4:
                st.write("<h1 style='font-align : center; font-size : 60; color :white'> Recall</h1>",unsafe_allow_html=True)
                st.write("<h1 style='font-align : center; font-size : 60; color :#2F58CD'>{:.2f}%</h1>".format(recall), unsafe_allow_html=True)
        with st.container():
            col1,col2= st.columns((1,2))
            with col1:
                # Create heatmap using Plotly
                plot_cm = go.Figure(data=go.Heatmap(
                   z=cm,
                   text=cm, 
                   colorscale="Blues", 
                   hovertemplate='x=%{x}<br>y=%{y}<br>z=%{text}'))
                plot_cm.update_layout( coloraxis_colorbar=dict(title='Value'))
                plot_cm.update_xaxes(showticklabels = False)
                plot_cm.update_yaxes(showticklabels=False)
                plot_cm.update_traces(texttemplate='%{text:.2s}')
                st.plotly_chart(plot_cm,use_container_width = True, height = 1500)
                
            with col2:
                plot_pie = px.pie(names=data_confusion.index, values=data_confusion, color = data_confusion.index, hole = 0.5)
                plot_pie.update_layout(showlegend = False)
                st.plotly_chart(plot_pie,use_container_width = True, height = 1000)
        
    
    
    elif selection == "App Credit Risk":
        st.header('Input Data')
        loan_amnt = st.number_input('Jumlah Pinjaman', min_value = 0, key = 'loan_amnt')
        person_age = st.number_input('Umur', min_value = 20, key = 'person_age')
        person_emp_length = st.number_input('Lama bekerja', min_value = 0)
        person_income = st.number_input('Pendapatan Per tahun', min_value = 0)
        loan_intent = st.selectbox('Tujuan Punjaman', ['Debtconsolidation', 'Venture', 'Medical', 'Education', 'Home improvement', 'Personal'])
        person_home_ownership = st.selectbox('Status Kepemilikan Rumah', ['Rent', 'Own', 'Mortagage', 'Other'])


        if st.button('Prediksi'):

            result = prediction(person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_amnt)
            st.subheader('Hasil Prediksi')
            st.write(f'Kredit {result}')
        

if __name__ == '__main__':
    main()


