import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as spio

def todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict

def loadmat(path):
    data = todict(spio.loadmat(path, struct_as_record=False, squeeze_me=True)['data'])
    return data["train"]["x"], data["train"]["y"], data["test"]["x"], data["test"]["y"]

def accuracy(y, pred):
    count = 0.0
    for i in range(0,200):
        if pred[i] == y[i]:
            count += 1
    return (count / 200.0)

def scoreFeatures(x, y):
    score = np.zeros((28,28)).astype(int)
    pred = np.zeros((28,28)).astype(int)
    for i in range(0, 28):
        for j in range(0, 28):
                sc1 = 0.0
                sc2 = 0.0
                for k in range(0,x.shape[2]):
                    if x[i,j,k] == 1:
                        if y[k] == 3:
                            sc1 += 1
                        else:
                            sc2 += 1
                    else:
                        if y[k] == 3:
                            sc2 += 1
                        else:
                            sc1 += 1
                if(sc1>sc2):
                    score[i,j] = sc1
                    pred[i,j] = 3
                else:
                    score[i,j] = sc2
                    pred[i,j] = 8
    return score, pred

# def applyDT(points, pred_vals, depth, x, y, predictions):
#     if depth = 1:

#         return predictions
#     i = points[0][1]
#     j = points[0][0]
#     for n in range(0,200):
#         if x[i,j,n] == 1:

# def split():


def applyDT1(p, pred, x, y):
    predictions = np.zeros(200)
    pred_opp = (pred-5)%10
    for n in range(0,200):
        if x[p[1],p[0],n] == 1:
            predictions[n] = pred
        else:
            predictions[n] = pred_opp
    acc = accuracy(y, predictions)
    return acc

def applyDT2(p1, p2, pred2, p3, pred3, x, y):
    predictions = np.zeros(200)
    pred2_opp = (pred2-5)%10
    pred3_opp = (pred3-5)%10
    for n in range(0,200):
        if x[p1[1],p1[0],n] == 1:
            if x[p2[1],p2[0],n] == 1:
                predictions[n] = pred2
            else:
                predictions[n] = pred2_opp
        else:
            if x[p3[1],p3[0],n] == 1:
                predictions[n] = pred3
            else:
                predictions[n] = pred3_opp
    acc = accuracy(y, predictions)
    return acc

#Function test
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = loadmat('data.mat')
    score1,pred1 = scoreFeatures(x_train, y_train)
    # plt.imshow(score1, cmap='gray')
    # plt.show()

    a = np.max(score1)
    print(a)
    (i,j) = (np.where(score1 == a)[1][0], (np.where(score1 == a)[0][0]))
    p1 = (i,j)
    print(p1)

    testDepth1 = applyDT1((i,j), pred1[j, i], x_test, y_test)
    print(testDepth1)

    ind1 = np.where(x_train[j,i,:]==1)
    ind2 = np.where(x_train[j,i,:]==0)

    sub1x = x_train[:,:,ind1[0]]
    sub1y = y_train[ind1[0]]

    sub2x = x_train[:,:,ind2[0]]
    sub2y = y_train[ind2[0]]

    score2,pred2 = scoreFeatures(sub1x,sub1y)

    b = np.max(score2)
    (k,l) = (np.where(score2 == b)[1][0], (np.where(score2 == b)[0][0]))
    p2 = (k,l)
    print(p2)

    score3,pred3 = scoreFeatures(sub2x,sub2y)
    c = np.max(score3)
    (m,n) = (np.where(score3 == c)[1][0], (np.where(score3 == c)[0][0]))
    p3 = (m,n)
    print(p3)

    testDepth2 = applyDT2(p1, p2, pred2[l, k], p3, pred3[n, m], x_test, y_test)
    print(testDepth2)
