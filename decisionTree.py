import numpy as np
# import matplotlib.pyplot as plt
import queue

class decisionTree:
    pred_vals = {}
    pred_opps = {}
    points = []
    depth = 0

    def __init__(self, depth=1, verbose=False):
        self.depth = depth
        self.predictions = np.zeros(200)
        self.verbose=verbose

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

    def accuracy(self, y, pred):
        count = 0.0
        for i in range(0,200):
            if pred[i] == y[i]:
                count += 1
        return (count / 200.0)

    def find_max(self, score):
        m = np.max(score)
        max_points = np.where(score == m)
        for x in range(0,len(max_points[0])):
            if (max_points[1][x], max_points[0][x]) not in self.points:
                return (max_points[1][x], max_points[0][x])
        return (-1,-1)

    def split(self, point, x, y):
        (i, j) = point
        ones = np.where(x[j,i,:]==1)

        ones_set_x = x[:,:,ones[0]]
        ones_set_y = y[ones[0]]

        ones_score, ones_pred = self.scoreFeatures(ones_set_x,ones_set_y)
        (k,l) = self.find_max(ones_score)
        self.pred_vals[(k,l)] = ones_pred[l, k]
        self.points.append((k,l))

        zeros = np.where(x[j,i,:]==0)

        zeros_set_x = x[:,:,zeros[0]]
        zeros_set_y = y[zeros[0]]

        zeros_score, zeros_pred = self.scoreFeatures(zeros_set_x, zeros_set_y)
        (m,n) = self.find_max(zeros_score)
        self.pred_vals[(m,n)] = zeros_pred[n, m]
        self.points.append((m,n))

        return (k,l), (m,n)

    def fit(self, x, y):
        self.root_score, self.root_pred = self.scoreFeatures(x, y)
        (i, j) = self.find_max(self.root_score)
        self.pred_vals[(i, j)] = self.root_pred[j, i]
        self.points.append((i, j))
        if self.depth > 1:
            nodes = queue.Queue()
            nodes.put((i, j))
            num_splits = (2**(self.depth-1)) - 1
            split_count = 0
            while split_count<num_splits:
                p = nodes.get()
                p1, p2 = self.split(p, x, y)
                split_count += 1
                if self.verbose:
                    print("Splitting " + str(p))
                    print("\tinto " + str(p1) + " and " + str(p2))
                nodes.put(p1)
                nodes.put(p2)

    def decide(self, n, points, depth, x, y, predictions):
        i = points[0][1]
        j = points[0][0]
        prediction = 0
        if x[i,j,n] == 1:
            if depth>1:
                prediction = self.decide(n, points[1:], depth-1, x, y, predictions)
            else:
                prediction = self.pred_vals[points[0]]
        else:
            if depth>1:
                prediction = self.decide(n, points[2:], depth-1, x, y, predictions)
            else:
                prediction = self.pred_opps[points[0]]
        return prediction

    def predict(self, x, y):
        for p in self.points:
            self.pred_opps[p] = ((self.pred_vals[p]-5)%10)
        for n in range(200):
            self.predictions[n] = self.decide(n, self.points, self.depth, x, y, self.predictions)
        return self.predictions
