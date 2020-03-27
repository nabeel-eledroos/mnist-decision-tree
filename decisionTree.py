import numpy as np
import matplotlib.pyplot as plt

class decisionTree:

    def accuracy(self, y, pred):
        count = 0.0
        for i in range(0,200):
            if pred[i] == y[i]:
                count += 1
        return (count / 200.0)

    def scoreFeatures(self, x, y):
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

    def get_point(self, score):
        m = np.max(score)
        (i,j) = (np.where(score == m)[1][0], (np.where(score == m)[0][0]))
        return (i, j)

    def split(self, points, pred_vals, pred_opps, depth, x, y, predictions):
        i = points[0][1]
        j = points[0][0]
        for n in range(0,200):
            if x[i,j,n] == 1:
                if depth>1:
                    split(points[1:], pred_vals, pred_opps, depth-1, x, y, predictions)
                else:
                    predictions[n] = pred_vals[points[0]]
            else:
                if depth>1:
                    split(points[2:], pred_vals, pred_opps, depth-1, x, y, predictions)
                else:
                    predictions[n] = pred_opps[points[0]]
        return predictions

    def applyDT(self, points, pred_vals, depth, x, y):
        predictions = np.zeros(200)
        pred_opps = {}
        for p in points:
            pred_opps[p] = ((pred_vals[p]-5)%10)
        return self.split(points, pred_vals, pred_opps, depth, x, y, predictions)

    def applyDT2(self, p1, p2, pred2, p3, pred3, x, y):
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
