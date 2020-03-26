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


#data.mat is a dict with the following structure:
#   {train:
#       {x: array with raw data (image height, image width, number of images)}
#       {y: array with corresponding labels}},
#   {test:
#       {x: array with raw data (image height, image width, number of images)}
#       {y: array with corresponding labels}}
def loadmat(path):
    return todict(spio.loadmat(path, struct_as_record=False, squeeze_me=True)['data'])

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

def applyDT1(p, pred, x, y):
    predictions = []
    pred_opp = (pred-5)%10
    for n in range(0,200):
        if x[p[1],p[0],n] == 1:
            predictions.append(pred)
        else:
            predictions.append(pred_opp)
    acc = accuracy(y, predictions)
    return acc

def applyDT2(p1, p2, pred2, p3, pred3, x, y):
    predictions = []
    pred2_opp = (pred2-5)%10
    pred3_opp = (pred3-5)%10
    for z in range(0,200):
        if x[p1[1],p1[0],z] == 1:
            if x[p2[1],p2[0],z] == 1:
                predictions.append(pred2)
            else:
                predictions.append(pred2_opp)
        else:
            if x[p3[1],p3[0],z] == 1:
                predictions.append(pred3)
            else:
                predictions.append(pred3_opp)
    acc = accuracy(y, predictions)
    return acc

#Function test
if __name__ == '__main__':
    data = loadmat('data.mat')
    x = data["train"]["x"]
    y = data["train"]["y"]
    x1 = data["test"]["x"]
    y1 = data["test"]["y"]
    score1,pred1 = scoreFeatures(x,y)
    # plt.imshow(score1, cmap='gray')
    # plt.show()

    a = np.max(score1)
    (i,j) = (np.where(score1 == a)[1][0], (np.where(score1 == a)[0][0]))
    print(np.where(score1 == a))
    print(i,j)

    testDepth1 = applyDT1((i,j), pred1[j, i], x1, y1)
    print(testDepth1)

    ind1 = np.where(x[j,i,:]==1)
    ind2 = np.where(x[j,i,:]==0)

    sub1x = x[:,:,ind1[0]]
    sub1y = y[ind1[0]]

    sub2x = x[:,:,ind2[0]]
    sub2y = y[ind2[0]]

    score2,pred2 = scoreFeatures(sub1x,sub1y)

    b = np.max(score2)
    (r,t) = (np.where(score2 == b)[1][0], (np.where(score2 == b)[0][0]))
    print(r,t)

    score3,pred3 = scoreFeatures(sub2x,sub2y)
    c = np.max(score3)
    (q,w) = (np.where(score3 == c)[1][0], (np.where(score3 == c)[0][0]))
    print(q,w)

    p1 = (i,j)
    p2 = (r,t)
    p3 = (q,w)
    testDepth2 = applyDT2(p1, p2, pred2[t, r], p3, pred3[w, q], x1, y1)
    print(testDepth2)