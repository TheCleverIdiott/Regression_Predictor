import streamlit as st
import numpy as np
import pandas as pd

data = "https://raw.githubusercontent.com/TheCleverIdiott/Score_Predictor/main/Dataset"   
students_data = pd.read_csv(data)

# X contains independent variable and y contains dependent variable
X = students_data.iloc[:, :-1].values
y = students_data.iloc[:, 1].values

#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#TRAINING THE MODEL
from sklearn.linear_model import LinearRegression
sp_model=LinearRegression()
sp_model.fit(X_train, y_train)


def predict_score(hours):
    input = np.asarray(hours)
    input_data_reshaped=input.reshape(1,-1)
    prediction = sp_model.predict(input_data_reshaped)
    print(type(prediction))
    return float(prediction)

def main():
    st.title("Predict Marks Based on Hours Studied")
    html_temp = '''
    <div style = "background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center">Score Predictor App</h2>
    </div>
    '''
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    hours = st.number_input("Enter Hours Studied")
    hours = float(hours)
    
    if st.button("Predict"):
        output = predict_score(hours)
        st.success("Your predicted score is {}".format(output))
    
if __name__ == '__main__':
    main()
    
    
    
