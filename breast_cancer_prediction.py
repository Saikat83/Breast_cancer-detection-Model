import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import csv
data=load_breast_cancer()
def normalize(X):
    val1=np.mean(X)
    val2=np.std(X)
    return (X-val1)/val2
def sigmoid(z):
    return 1/(1+np.exp(-z))
def initialize_parameter(dim):
    W=np.zeros(shape=(dim,1))
    b=0
    return W,b
def propagate(W,b,X,Y):
    m=Y.shape[1]
    Z=np.dot(W.T,X)+b
    A=sigmoid(Z)
    cost=(-1/m)*(np.sum(Y * np.log(A)+ (1 - Y) * (np.log(1 - A))))
    dZ=A-Y
    dW=(1/m)*np.dot(X,dZ.T)
    db=(1/m)*np.sum(dZ)
    grads={"dw":dW,"db":db}
    return grads,cost
def Gradient_Descent(W,b,X,Y,learning_rate,no_of_iteration):
    costs=[]
    for i in range(no_of_iteration): 
        grads,cost=propagate(W,b,X,Y)
        dW=grads["dw"]
        db=grads["db"]
        W=W-(learning_rate)*dW
        b=b-(learning_rate)*db
        costs.append(cost)
    params={"W":W,"b":b}
    return params,costs
def model(X_train,Y_train,learning_rate,no_of_iteration):
    dim=X_train.shape[0]
    W,b=initialize_parameter(dim)
    params,costs=Gradient_Descent(W,b,X_train,Y_train,learning_rate,no_of_iteration)
    Weight=params["W"]
    Bias=params["b"]
    return Weight,Bias,costs
Y_data=data.target
X_data=data.data
for i in range(X_data.shape[1]):
    X_data[:,i]=normalize(X_data[:,i])
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=0)
X_train=X_train.T
Y_train=Y_train.reshape(X_train.shape[1],1).T
X_test=X_test.T
Y_test=Y_test.reshape(X_test.shape[1],1).T
W,b,costs=model(X_train,Y_train,0.2,500)
print(W,b)
plt.figure()
plt.plot(np.arange(len(costs)),costs)
plt.xlabel("No of iteration")
plt.ylabel("Cost Function")
plt.show()
y_pred=sigmoid(np.dot(W.T,X_test)+b)
m=X_test.shape[1]
for i in range(m):
    if y_pred[0,i]>=0.5:
        y_pred[0,i]=1
    else:
        y_pred[0,i]=0
count=0
for i in range(m):
    if y_pred[0,i]==Y_test[0,i]:
        count+=1
y_pred.astype(int)
csvdata=[['Actual Value','Predicted Value']]
for i in range(m):
    csvdata.append([Y_test[0,i],y_pred[0,i]])
with open('Result2.csv','w') as csvFile:
    w=csv.writer(csvFile)
    w.writerows(csvdata)
csvFile.close()
print((count*1.0/m)*100.0)
