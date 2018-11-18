from __future__ import division
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from math import sqrt
from scipy import sparse


def ridgeReg(X,y,l):
    print l
    one=np.ones(shape=(1,X.shape[1]))          
    X=np.vstack((X,one))
    X_trans=X.transpose()                          
    identity=np.identity(X.shape[0]-1)                #kxk identity matrix
    zero=np.zeros(shape=(X.shape[0]-1,1))             #kx1 zero matrix
    identity=np.hstack((identity,zero))
    identity=np.vstack((identity,np.append((np.transpose(zero)),0)))
    C=np.dot(X,X_trans)
    t=np.multiply(l,identity)
    C+=t
    d=np.dot(X,y)
    C_inv=inv(C)
    w=np.dot(C_inv,d)           #weight matrix when trained on entire training data
    temp=np.dot(X_trans,w) -y
    w_trans=np.transpose(w)
    obj=np.multiply(l,np.dot(w_trans,w)) - l*(w.item(X.shape[0]-1)**2)+ np.dot(np.transpose(temp),temp)
    cvErrs=np.empty(shape=(X.shape[1],1))
    for i in range(0,X.shape[1]):
        x_i=X[:,i]
        error=(np.dot(w_trans,x_i)-y.iat[i,0])/(1-np.dot(np.transpose(x_i),np.dot(C_inv,x_i)))
        cvErrs=np.append(cvErrs,error)
    b=w.item(X.shape[0]-1)
    w=np.delete(w,X.shape[0]-1,0)
    return w,obj,b,cvErrs



X_t=pd.read_csv('trainData.csv')
y_t=pd.read_csv('trainLabels.csv')

X_v=pd.read_csv('valData.csv')
y_v=pd.read_csv('valLabels.csv')

X_t=X_t.drop(X_t.columns[0],axis=1)
y_t=y_t.drop(y_t.columns[0],axis=1)

X_v=X_v.drop(X_v.columns[0],axis=1)
y_v=y_v.drop(y_v.columns[0],axis=1)

rmvalues_t=[]
rmvalues_v=[]
cverr_t=[]
obj_values=[]
w_array=[]
#cverr_v=[]

l=[0.01,0.1,1,10,100,1000]
for each in l:
    weights_t,obj_cost_t,bias_t,cverror_t=ridgeReg(X_t.transpose(),y_t,each)
    rmse_train= sqrt(mean_squared_error(y_t, np.add(np.dot(X_t,weights_t),bias_t)))
    rmse_val= sqrt(mean_squared_error(y_v, np.add(np.dot(X_v,weights_t),bias_t)))
    cv_t=sqrt(np.sum(np.square(cverror_t))/5000)
    w_array.append(weights_t)
    rmvalues_t.append(rmse_train)
    rmvalues_v.append(rmse_val)
    cverr_t.append(cv_t)
    obj_values.append(obj_cost_t)
    print l
    print rmse_train
    print rmse_val
    print cv_t


l_correct=cverr_t.index(min(cverr_t))
w_temp=w_array[l_correct]
print "For best lambda:"
print ("Cost of objective function ",obj_values[l_correct])
print ("RMSE training value ",rmvalues_t[l_correct])
print ("Regularization term ",l[l_correct]*(np.dot(np.transpose(w_temp),w_temp)))


max_weights=[]
max_weights_index=[]

min_weights=[]
min_weights_index=[]
w_temp=w_temp.tolist()
w_temp = [item for sublist in w_temp for item in sublist]
w_temp=[abs(number) for number in w_temp]
w_temp_copy=[item for item in w_temp]
for i in range(0,10):
    max_weights.append(max(w_temp))
    max_weights_index.append(w_temp.index(max(w_temp)))
    w_temp[max_weights_index[i]]=0
    min_weights.append(min(w_temp_copy))
    min_weights_index.append(w_temp.index(min(w_temp_copy)))
    w_temp_copy[min_weights_index[i]]=10000

print ("max weights are ",max_weights)
print ("max weight indices are ",max_weights_index)
print ("min weights are ",min_weights)
print ("min weight indices are ",min_weights_index)

l=np.log10(l)
mp.plot(l,rmvalues_t,marker='o',markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label="RMS Train")
mp.plot( l, rmvalues_v, marker='', color='olive', linewidth=2,label="RMS Val")
mp.plot( l, cverr_t, marker='', color='olive', linewidth=2, linestyle='dashed', label="CV error")
mp.legend()
mp.show()

