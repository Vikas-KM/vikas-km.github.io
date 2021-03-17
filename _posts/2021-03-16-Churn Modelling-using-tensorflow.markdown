---
layout: post
title: "Churn Model Prediction using TensorFlow"
date: 2021-03-16
---

<p class="intro"><span class="dropcap">I</span>n this post we will implement Churn Model Prediction System using the Bank Customer data.</p>

Using the Bank Customer Data, we can develop a ML Prediction System which can predict if a customer will leave the Bank or not, In Finance this is known as <strong>Churning.</strong> Such ML Systems can help Bank to take precautionary steps to ensure the customer stays with the Bank.

### Importing the Necessary Libraries

Lets start by importing the necessary libraries needed to execute this project.
{%- highlight python -%}
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
{%- endhighlight -%}

The problem that we are trying to classify is a <strong>binary classification</strong> problem. <br/> In every ML Problem we need to clean the data, preprocess the data and split the data for train and validation, so lets import necessary libraries for those tasks.

{%- highlight python -%}
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
{%- endhighlight -%}

### Loading the Data

We will use google colab for this project, let's load the dataset from kaggle on to colab, [follow the link][colab_kaggle] if you dont know how to load kaggle data onto colab.

Once the data is loaded we can use the panda's head function to look at the data and info function of panda's to have a glance at the column types and if there are any null values present.
Luckily this dataset is clean and there are no null values present.

<figure>
	<img src="{{ '/assets/img/churn-model/info.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig1. - Panda's Info function</figcaption>
</figure>

### Preprocessing of data

It is very important to shuffle the data before we begin with preprocessing of the data.<br/>
The shuffle function of scikit-learn will shuffle the data for us.
{%- highlight python -%}
data = shuffle(data)
{%- endhighlight -%}

#### Preprocessing involves steps like

- Checking for null values
- What columns to be used and what to be dropped
- Converting categorical values to numerical values
- Normalization or Standardisation of the column data

#### Check for NULL Values

using the below pandas functions we can get the number of null values present in each column of the dataset.

{%- highlight python -%}
data.isna().sum()
{%- endhighlight -%}
As you can see from the below graph, All the columns report zero, indication there are no null values in the dataset.

<figure>
	<img src="{{ '/assets/img/churn-model/null.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig2. - No NULL Values Present</figcaption>
</figure>

#### Selecting important columns

Pandas columns attribute will list the column names, the CustomerId, Surname and RowNumber are not very helpful for our model.
The Exited column is a target column
{%- highlight python -%}
data.columns
X = data.drop(['RowNumber','CustomerId','Surname','Exited'], axis=1)
y = data['Exited']
{%- endhighlight -%}
We will split the dataset into X and y, where X is input and y is target variable.

#### Categorical to Numerical

The pandas dtype returns a series with the data type of each column. We can see that the 'Geography' and 'Gender' are categorical columns.
{%- highlight python -%}
X.dtypes
X['Geography'].unique()
X['Gender'].unique()
X = pd.get_dummies(X, prefix='Geography', drop_first=True, columns=['Geography'])
X = pd.get_dummies(X, prefix='sex', drop_first=True, columns=['Gender'])
{%- endhighlight -%}
The pandas unique function return a series listing the unique values of each column. We will use pandas get_dummies function to convert the categorical columns into numerical columns.

#### Applying standardisation

We will apply standardisation, so no column has more effect on the output than other columns. It scales all values between -1 and 1
{%- highlight python -%}
scalar = StandardScaler()
X = scalar.fit_transform(X)
{%- endhighlight -%}

#### Creation of Training and Validation datasets

Using the train_test_split we will create training and validation datasets, until the model is final we should not use the test datasets.
{%- highlight python -%}
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
{%- endhighlight -%}

### NN Model Building

Now that the data is ready, we can build a NN (Neural Network) model, we can start simple and use a 1 or 2 hidden layers. In this for demostration purpose we will build a 4 hidden layers model. You can change the model and fine tune it to learn better using more layers and experimenting with different nodes in each layer.
{%- highlight python -%}
model = tf.keras.models.Sequential([
Dense(256, activation='relu', input_shape=x_train.shape),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(1, activation='sigmoid'),
])
{%- endhighlight -%}
We are using Sequential model here and the dense layers, the activation function used is relu, since this is binary classification problem we have used sigmoid as the activation function in the last layer.<br/>
We can use the summary function to have a look at our model and its network
{%- highlight python -%}
model.summary()
{%- endhighlight -%}

<figure>
	<img src="{{ '/assets/img/churn-model/summary.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig3. - Model Summary</figcaption>
</figure>

#### Compile and fit the model

{%- highlight python -%}
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
history = model.fit(X_train, y_train, verbose=1, epochs=50, batch_size=32, validation_data=(x_test, y_test))
)
{%- endhighlight -%}

#### Plotting the model accuracy and loss plots

<figure>
	<img src="{{ '/assets/img/churn-model/accuracy.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig4. - Model Accuracy</figcaption>
</figure>
<figure>
	<img src="{{ '/assets/img/churn-model/loss.png' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig5. - Model Loss</figcaption>
</figure>

From the graphs we can see that the validation accuracy is around >80% not bad for a simple model.

### Conclusion

In this problem we have used a simple 4 layer model to come up with a prediction system, the model is certainly overfitting ( the validation loss keeps increasing). There is lot of scope to improve this model further.

#### Further Improvements

- We can add dropout layers to make model less overfit
- Use callbacks to stop model learning once the accuracy stops getting better
- Experimenting with different nodes and hidden layers.

### Different Dataset same problem

Use the [telco dataset][kaggle_telco] and see if you can use the same model to predict if a customer will leave their telecom network or not.

#### Source Code

Checkout the full [Source Code][source_url] of the above project.

### References

- [Artificial Neural Network using Tensorflow 2][book_url] Book.
- [TensorFlow][tensorflow_url] website

[colab_kaggle]: https://www.kaggle.com/general/74235
[kaggle_url]: https://www.kaggle.com/shrutimechlearn/churn-modelling
[kaggle_telco]: https://www.kaggle.com/c/customer-churn-prediction-2020
[book_url]: https://www.apress.com/in/book/9781484261491
[tensorflow_url]: https://www.tensorflow.org/tutorials
[source_url]: https://github.com/Vikas-KM/tensorflow-learning/blob/master/binary_classification.ipynb
