import streamlit as st
import numpy as np # for performing mathematical calculations behind ML algorithms
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for handling and cleaning the dataset
import seaborn as sns # for visualization
import sklearn
string = "Startup's Profit Prediction"
# setup page config — dynamic web page
st.set_page_config(page_title=string, page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)
# st.title is a widget element
st.title (string, anchor=None)

from PIL import Image 
image = Image.open('startup.png') #load image
st.image(image) # st.image — image widget/placeholder

from PIL import Image
image = Image.open('startup.png')

st.image(image)


dataset = pd.read_csv("50_Startups.csv")

# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)

training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)

rnd_cost = st.sidebar.number_input('Insert R&D Spend')
st.write('The current number is ', rnd_cost)

Administration_cost = st.sidebar.number_input('Insert Administration cost Spend')
st.write('The current number is ', Administration_cost)

Marketing_cost_Spend = st.sidebar.number_input('Insert Marketing cost Spend')
st.write('The current number is ', Marketing_cost_Spend)


y_pred = model.predict([[Marketing_cost_Spend,Administration_cost,rnd_cost]])

if st.button('Predict'):
    st.success('The Profit must be  {} '.format(y_pred))
else:
     st.write('Please fill all the important details')


fig = plt.figure()

X = ['Toal cost Spend']
x_value = [rnd_cost+Administration_cost+Marketing_cost_Spend]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, x_value, 0.4, label = 'cost')
plt.bar(X_axis + 0.2, y_pred, 0.4, label = 'profit')
  
plt.xticks(X_axis, X)
plt.xlabel("RS")
plt.title("Profit vs Toal cost spend")
plt.legend()
plt.show()

st.pyplot(fig)


