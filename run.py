import scipy.io as spio
import numpy as np
from decisionTree import decisionTree

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

#Function test
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = loadmat('data.mat')

    tree1 = decisionTree(depth=1)
    tree1.fit(x_train, y_train)
    testDepth1 = tree1.predict(x_test, y_test)
    print(tree1.accuracy(y_test, testDepth1))


    # ind1 = np.where(x_train[j,i,:]==1)
    # ind2 = np.where(x_train[j,i,:]==0)

    # sub1x = x_train[:,:,ind1[0]]
    # sub1y = y_train[ind1[0]]

    # sub2x = x_train[:,:,ind2[0]]
    # sub2y = y_train[ind2[0]]

    # score2,pred2 = tree.scoreFeatures(sub1x,sub1y)

    # (k,l) = tree.get_point(score2)
    # p2 = (k,l)
    # pred_vals[p2] = pred2[l, k]
    # points.append(p2)
    # print(p2)


    # score3,pred3 = tree.scoreFeatures(sub2x,sub2y)
    # (m,n) = tree.get_point(score3)
    # p3 = (m,n)
    # pred_vals[p3] = pred3[n, m]
    # points.append(p3)
    # print(p3)

    # testDepth2 = tree.applyDT(points, pred_vals, 2, x_test, y_test)

    # print(tree.accuracy(y_test, testDepth2))
