
import gradio as gr
from gradio.outputs import Label 
import cv2
import tensorflow as tf
import matplotlib as mpl 
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train_full,y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_valid, x_train = x_train_full[:5000] / 255.0 , x_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]
x_test = x_test / 255.0

print("X_train.shape: ", x_train.shape)
print("X_valid.shape: ", x_valid.shape)
print("X_test.shape: ", x_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_valid.shape: ", y_valid.shape)
print("y_test.shape: ", y_test.shape)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

"""# Task 2"""

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape((28,28)))
    label_index = int(y_train[i])
    plt.title(class_names[label_index])
plt.show()

"""# Task 3"""

y=y_train

plt.hist(y, bins=20)
plt.title("Bar plot for training labels ")

y=y_test

plt.hist(y, bins=20)
plt.title("Bar plot for testing labels ")

y= y_valid

plt.hist(y, bins=20)
plt.title("Bar plot for testing labels ")

"""#Task 4

Feature Scaling - Coverting 3D tensor to 2D tensor
"""

#Flattening the training set
n, width, height = x_train.shape[0], x_train.shape[1], x_train.shape[2]
x_train_flat = x_train.reshape(n, width*height)
x_train_flat.shape

#Flattening the testing set
n, width, height = x_test.shape[0], x_test.shape[1], x_test.shape[2]
x_test_flat = x_test.reshape(n, width*height)
x_test_flat.shape

#Flattening the validation set
n, width, height = x_valid.shape[0], x_valid.shape[1], x_valid.shape[2]
x_valid_flat = x_valid.reshape(n, width*height)
x_valid_flat.shape

"""## KNN Classfier

4.1.1
1. **'n_neighbors' -** This parameter is used to specify 
the value of n neighbors which the model needs to be run i.e the number of neighbors for each instance in order to predict the output.
2. **'metric' -** This parameter is used to calculate the distance metrics of the model. Defaultly, minkowski metrics is calculated.
3. **'p' -** This parameter is used to specify the type of Minkowski distance metric. if p=1 , Manhattan distance is used. When p=2, Euclidean distance is used.
4. **'n_jobs' -** This parameter specifies the total number of parallel jobs that can run during the neighbour search. We can toggle this parameter depending upon the computing resource that we have for faster execution.

4.1.2

1. **fit(X,y)-** This function is used to fit the k-nearest neighbours from the training dataset. The k-value will be specified in the KNeighborsClassifier() function.
2. **predict(X)-** This function is used to predict the class labels for the provided input data i.e X .
3. **predict_proba(X)-** This function will calculate the probability estimation for the given input test data X.
4. **score(X,y)-** This function will return the mean accuracy of the test data by comparing it with the labels.

4.1.3
"""

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_flat,y_train)

start = time.time()
y_valid_predicted = knn.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))

"""4.1.4"""

s1 = pickle.dumps(knn)
knn = pickle.loads(s1)

"""4.1.5"""

prediction_summary = pd.DataFrame({'predicted_label':y_valid_predicted, 'actual_label':y_valid})
prediction_summary

"""4.1.6"""

def get_accuracy():
  c=0
  for i in range(5000):
    if y_valid_predicted[i]==y_valid[i]:
      c+=1
  return c/5000

print("Accuracy using user defined function :",get_accuracy())

print("Accuracy of the model using sklearn metrics :", metrics.accuracy_score(y_valid, y_valid_predicted))

"""4.1.7"""

def get_per_class_accuracy(y, y_pred):
  actual_count={'0':0, '1':0, '2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9': 0}
  pred_count={'0':0, '1':0, '2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9': 0}
  for i in range(5000):
    actual_count[str(y[i])]+=1
    if y_pred[i]==y[i]:
      pred_count[str(y[i])]+=1
  t=[]
  for i in actual_count.keys():
    t.append(pred_count[i]/actual_count[i])
  for i in range(10):
    print(f'The validation accuracy of {class_names[i]} is : ', t[i], ' ------->', round(t[i]*100,2),"%" )

get_per_class_accuracy(y_valid, y_valid_predicted)

"""4.1.8"""

ConfusionMatrixDisplay.from_predictions(y_valid, y_valid_predicted)
plt.title("Classification Confusion matrix")
plt.show()

"""4.1.9"""

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_test_flat,y_test)
start = time.time()
y_test_predicted = knn.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy = ", metrics.accuracy_score(y_test, y_test_predicted))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_test_flat,y_test)
start = time.time()
y_test_predicted = knn.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy = ", metrics.accuracy_score(y_test, y_test_predicted))

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_test_flat,y_test)
start = time.time()
y_test_predicted = knn.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy = ", metrics.accuracy_score(y_test, y_test_predicted))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_test_flat,y_test)
start = time.time()
y_test_predicted = knn.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy = ", metrics.accuracy_score(y_test, y_test_predicted))

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_test_flat,y_test)
start = time.time()
y_test_predicted = knn.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy = ", metrics.accuracy_score(y_test, y_test_predicted))

"""From the above results it is evident that for the value of K=2 , the model gives better accuracy.

4.1.10
"""

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_test_flat,y_test)
start = time.time()
y_test_predicted = knn.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy of test set = ", metrics.accuracy_score(y_test, y_test_predicted))

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_valid_flat,y_valid)
start = time.time()
y_valid_predicted = knn.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
ConfusionMatrixDisplay.from_predictions(y_valid, y_valid_predicted)
plt.title("Classification Confusion matrix")
plt.show()
print("Overall Accuracy of validation set = ", metrics.accuracy_score(y_valid, y_valid_predicted))

"""## 4.2 Linear Discriminant Analysis

4.2.1

Bayes classification rule for multi-class classfication is done by calculating the posterior probability P of event or value X belonging to class Y.
i.e 𝑷(𝒀 = 𝒋 |𝑿 = 𝒙𝒊) where 𝟏 ≤ 𝒋 ≤ 𝑲

4.2.2
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda= LinearDiscriminantAnalysis()
lda.fit(x_train_flat, y_train)

start = time.time()
predicted_labels = lda.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))

"""The running time of KNN for validation set is 1.07 seconds whereas it takes 0.032 seconds for Linear Discriminant Analysys. Hence, Linear Discriminant Analysis runs faster for validation set.

4.2.3
"""

s2 = pickle.dumps(lda)
lda = pickle.loads(s2)

"""4.2.4"""

#Training Set Accuracy
start = time.time()
y_train_pred = lda.predict(x_train_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_train, y_train_pred))

#Testing Set Accuracy
start = time.time()
y_test_pred = lda.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_test, y_test_pred))

#Validation Set Accuracy
start = time.time()
y_valid_pred = lda.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_valid, y_valid_pred))

"""4.2.5"""

get_per_class_accuracy(y_test,y_test_pred)

"""## 4.3 Quadratic Discriminant Analysis

4.3.1
"""

qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train_flat, y_train)

start = time.time()
predicted_labels = qda.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))

"""The running time of KNN for validation set is 1.07 seconds whereas it takes 2.29 seconds for Quadratic Discriminant Analysis. Hence, KNN runs faster for validation set.

4.3.2
"""

s3 = pickle.dumps(qda)
qda = pickle.loads(s3)

"""4.3.3"""

#Training Set Accuracy
start = time.time()
y_train_pred = qda.predict(x_train_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_train, y_train_pred))

#Testing Set Accuracy
start = time.time()
y_test_pred = qda.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_test, y_test_pred))

#Validation Set Accuracy
start = time.time()
y_valid_pred = qda.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_valid, y_valid_pred))

"""4.3.4"""

get_per_class_accuracy(y_test,y_test_pred)

"""## 4.4 Gaussian Naive Bayes Classifier

4.4.1
"""

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train_flat, y_train)

start = time.time()
predicted_labels = gnb.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))

"""The running time of KNN for validation set is 1.07 seconds whereas it takes 0.315 seconds for Gaussian Naive Bayes. Hence, Gaussian Naive bayes model runs faster for validation set.

4.4.2
"""

s4 = pickle.dumps(gnb)
gnb = pickle.loads(s4)

"""4.4.3"""

#Training Set Accuracy
start = time.time()
y_train_pred = gnb.predict(x_train_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_train, y_train_pred))

#Testing Set Accuracy
start = time.time()
y_test_pred = gnb.predict(x_test_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_test, y_test_pred))

#Validation Set Accuracy
start = time.time()
y_valid_pred = gnb.predict(x_valid_flat)
end = time.time()
time_duration = end-start
print("Program finishes in {} seconds:".format(time_duration))
print("Accuracy of training Set: ", metrics.accuracy_score(y_valid, y_valid_pred))

"""4.4.4"""

get_per_class_accuracy(y_test,y_test_pred)

"""##Gradio Implementation"""


def caption(image,input_module1):
  class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] 
  image=image.reshape(1,28*28)
  if input_module1=="KNN":
    output1=knn.predict(image)[0]
    predictions=knn.predict_proba(image)[0]
    
  elif input_module1==("Linear discriminant analysis"):
    output1=lda.predict(image)[0]
    predictions=lda.predict_proba(image)[0]
    
  elif input_module1==("Quadratic discriminant analysis"):
    output1=qda.predict(image)[0]
    predictions=qda.predict_proba(image)[0]
    
  elif input_module1=="Naive Bayes classifier":
    output1=gnb.predict(image)[0]
    predictions=gnb.predict_proba(image)[0]
  output2 = {}
  for i in range(len(predictions)):
    output2[class_names[i]] = predictions[i]
  return output1 ,output2

input_module = gr.inputs.Image(label = "Input Image",image_mode="L",shape=(28,28))
input_module1 = gr.inputs.Dropdown(choices=["KNN","Linear discriminant analysis", "Quadratic discriminant analysis","Naive Bayes classifier"], label = "Method")
output1 = gr.outputs.Textbox(label = "Predicted Class")
output2=gr.outputs.Label(label= "probability of class")
gr.Interface(fn=caption, inputs=[input_module,input_module1], outputs=[output1,output2]).launch(debug=True)

