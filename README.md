# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
To develop a NN regression model we need dataset . I created a dataset with one input and output values. The data will be divided into two categories one testing the data and next training the data. The model consists of one input layer , 8 neurons in the  hidden layer  with relu activation and 10 neurons in the hidden layer with relu activation which connected with the one output layer. The model is complied with optimizer and loss 


## Neural Network Model

![d1](https://user-images.githubusercontent.com/81132849/226178966-749dd928-b2e4-4186-a9e6-109f4a5bffde.jpg)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```python3
import pandas as pd
df = pd.read_csv("student_scores.csv")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df.head()
x = df[['Hours']].values
y = df[['Scores']].values
x
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state =33)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train1 = Scaler.transform(x_train)
x_train1
ai_brain  = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
x_n1 = [[4]]
x_n11_1 = Scaler.transform(x_n1)
ai_brain.predict(x_n11_1)



```

## Dataset Information

![d2](https://user-images.githubusercontent.com/81132849/226179112-0df79155-7c6a-4534-ad59-602a23f9ef00.jpg)

## OUTPUT

### Training Loss Vs Iteration Plot

![d3](https://user-images.githubusercontent.com/81132849/226179334-f94db0ac-826c-4fa2-ad34-03793074211c.jpg)


### Test Data Root Mean Squared Error

![d4](https://user-images.githubusercontent.com/81132849/226179528-03360be6-82af-47ff-afc6-7b6ac87db6b7.jpg)


### New Sample Data Prediction

![d5](https://user-images.githubusercontent.com/81132849/226181357-e52689c1-7294-4750-8bee-892c90f786a2.jpg)


## RESULT

Thus,the neural network regression model for the given dataset is developed.
