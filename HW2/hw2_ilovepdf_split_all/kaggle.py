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
    #C=C.toarray()
    t=np.multiply(l,identity)
    C+=t
    #C=C.todense()
    d=np.dot(X,y)
    C_inv=inv(C)
    w=np.dot(C_inv,d)           #weight matrix when trained on entire training data
    temp=np.dot(X_trans,w) -y
    w_trans=np.transpose(w)
    obj=np.multiply(l,np.dot(w_trans,w)) + np.dot(np.transpose(temp),temp)
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

'''X_v=pd.read_csv('valData.csv')
y_v=pd.read_csv('valLabels.csv')'''

X_t=X_t.drop(X_t.columns[0],axis=1)
y_t=y_t.drop(y_t.columns[0],axis=1)
#X_new = SelectKBest(mutual_info_regression, k=100).fit_transform(X_t, y_t)
X_test=pd.read_csv('testData.csv')
X_test=X_test.drop(X_test.columns[0],axis=1)
print X_test.shape

'''X_v=X_v.drop(X_v.columns[0],axis=1)
y_v=y_v.drop(y_v.columns[0],axis=1)
'''
rmvalues_t=[]
rmvalues_v=[]
cverr_t=[]
obj_values=[]
#cverr_v=[]

l=[0.7]
weight_max=0.0
predictions=np.empty(shape=(1,X_t.shape[0]))
for each in l:
    weights_t,obj_cost_t,bias_t,cverror_t=ridgeReg(X_t.transpose(),y_t,each)
    print sqrt(np.sum(np.square(cverror_t))/5000)
    predictions=np.add(np.dot(X_test,weights_t),bias_t)
    weight_max=max(weights_t)


frame=pd.DataFrame(data=predictions)
frame.to_csv('predTestLabels.csv',encoding='utf-8',index=True)



