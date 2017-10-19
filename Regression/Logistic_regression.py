# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/31 上午10:55
# @Author  : huaixian
# @File    : Logistic_regression.py
# @Software: pycharm

import numpy as np
from sklearn.preprocessing import *
from sklearn import metrics
from sklearn.linear_model import *
import numpy.matlib
import math

#评分函数
def auc(model, X, y, posLab=1):
    pred = np.array([ proba[1] for proba in model.predict_proba(X)])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=posLab)
    return metrics.auc(fpr, tpr)
#区分数值型特征与类别行特征
def loadFeatType(fname):
    featType = {}
    with open(fname) as f:
        header = f.readline()
        for line in f.readlines():
            k,v=line.strip().split(',')
            if k[:2] == '\"x':
                featType[k] = 0 if v == '\"numeric\"' else 1 #0: numeric, 1: categorical
    return featType
#预处理数据
def loadX(fname, featType, imp4Num=None, imp4Cat=None, scaler=None, enc=None):
    uids = []
    featVecLst4Num = []
    featVecLst4Cat = []
    typLst = []

    with open(fname) as f:
        header = f.readline()
        typLst += [featType[field] for field in header.strip().split(',')[1:]]
        for line in f.readlines():
            toks = line.strip().split(',')
            uids.append(toks[0])
            featVec4Num=[]
            featVec4Cat=[]
            for typ, v in zip(typLst, toks[1:]):
                v = v.replace('\"', '')
                if typ == 0:
                    featVec4Num.append(np.nan if v == '-1' else float(v))
                    #featVec4Num.append(float(v))
                else:
                    featVec4Cat.append(np.nan if v[0] == '-' else int(v))
                    #featVec4Cat.append(int(v))
            featVecLst4Num.append(np.array(featVec4Num))
            featVecLst4Cat.append(np.array(featVec4Cat))
    featMat4Num = np.array(featVecLst4Num)
    featMat4Cat = np.array(featVecLst4Cat)

    catCount = sum(typLst)
    numCount = len(typLst) -catCount
    typLst = [0] * numCount + [1] * catCount

    #first deal with missing value
    if imp4Num == None:
        imp4Num = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp4Num.fit(featMat4Num)
    featMat4NumImp = imp4Num.transform(featMat4Num)

    if imp4Cat == None:
        imp4Cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        imp4Cat.fit(featMat4Cat)
    featMat4CatImp = imp4Cat.transform(featMat4Cat)

    #second deal with scaling
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(featMat4NumImp)
    featMat4NumScale = scaler.transform(featMat4NumImp)

    #third one-hot encoding
    if enc == None:
        enc = OneHotEncoder()
        enc.fit(featMat4CatImp)
    featMat4CatEnc = enc.transform(featMat4CatImp).toarray() # scipy.sparse.csr.csr_matrix

    #concatenate numeric and categorical features
    newFeatMat = np.concatenate((featMat4NumScale,featMat4CatEnc), axis=1)
    #newFeatMat = np.concatenate((featMat4NumImp, featMat4CatImp), axis=1)
    #return uids, newFeatMat, imp4Num, imp4Cat, scaler, enc
    return uids, newFeatMat, imp4Num, imp4Cat, scaler, enc, typLst
#处理标签数据
def loadY(fname, uids):
    labels = {}
    with open(fname) as f:
        header = f.readline()
        for line in f.readlines():
            uid, lab = line.strip().split(',')
            labels[uid] = float(lab)
    return np.array([labels[uid] for uid in uids])
#sigmoid函数
def logisticFun(x, w): #x: Fx1, w: Fx1
    return 1/(1+math.exp(-w.T*x))
#损失函数
def lossFun(X, y, w): #X: NxF, y:Nx1, w:Fx1
    numSamples = len(y)
    loss = 0
    c=1
    for i in range(numSamples):
        x_i = X[i,:] #x: 1xF
        y_i = y[i]

        loss += (y_i*x_i*w - math.log(1+math.exp(x_i*w)))
    loss/= numSamples

    return -loss + c*math.sqrt(w.T*w),c
#梯度下降法求解最优参数
def lossGrad(X, y, w,c): #X: NxF, y:Nx1, w:Fx1
    numSamples, F = X.shape
    grad = np.matlib.zeros((F, 1))
    for i in range(numSamples):
        x_i = X[i, :].T #x: 1xF
        y_i = y[i]
        y_pred = logisticFun(x_i, w)
        grad += (y_i - y_pred) * x_i
    grad /= numSamples
    return -grad + 2*c*w
#调用logisticFun,lossFun,lossGrad训练模型参数
def train(X, y,c):
    numSamples, F = X.shape
    w = np.matlib.zeros((F, 1))
    #w = np.matlib.randn((F, 1))
    maxIter = 30
    loss = lossFun(X, y, w)
    print 'init: ', loss
    for it in range(maxIter):
        grad = lossGrad(X, y, w,c)
        alpha = 0.01
        for j in range(10):
            w_tmp = w - alpha * grad
            loss_tmp = lossFun(X, y, w_tmp)

            if loss_tmp < loss:
                w = w_tmp
                loss = loss_tmp
                break
            else:
                alpha /= 2.
        print 'iteration '+str(it)+": "+str(loss)
    return w
#对模型打分
def score(X, y, w):
    numSamples = len(y)
    hits = 0.
    for i in range(numSamples):
        x_i = X[i, :].T  # x: 1xF
        y_i = y[i]
        y_pred = logisticFun(x_i, w)
        y_pred = 0 if y_pred < 0.5 else 1
        if y_i == y_pred:
            hits += 1.
    return hits/numSamples

if __name__ == "__main__":
    #featType = loadFeatType("E:/DC/RenPin/features_type.csv")
    featType = loadFeatType("/Users/huanghuaixian/ipython/renpindasai/features_type.csv")
    uids, dataX, imp4Num, imp4Cat, scaler, enc, typLst = loadX("/Users/huanghuaixian/ipython/renpindasai/train_x.csv", featType)
    datay = loadY("/Users/huanghuaixian/ipython/renpindasai/train_y.csv", uids)
    splitPoint = int(0.8*datay.size)
    trainX = np.matrix(dataX[:splitPoint])
    trainy = datay[:splitPoint]
    devX = np.matrix(dataX[splitPoint:])
    devy = datay[splitPoint:]
    c=0.1
    w = train(trainX, trainy,c)
    print 'score on train ', score(trainX, trainy, w)
    print 'score on dev ', score(devX, devy, w)
    model = LogisticRegression()
    #model = DecisionTreeClassifier(min_samples_leaf=4)
    model.fit(trainX , trainy)
    print 'score on train ', model.score(trainX, trainy)
    print 'score on dev ', model.score(devX, devy)

    print 'auc on train ', auc(model, trainX, trainy, posLab=1)
    print 'auc on dev ', auc(model, devX, devy, posLab=1)