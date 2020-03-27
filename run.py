import scipy.io as spio
import decisionTree
from decisionTree import accuracy
from decisionTree import applyDT
from decisionTree import scoreFeatures
from decisionTree import get_point

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
    score1,pred1 = scoreFeatures(x_train, y_train)
    # plt.imshow(score1, cmap='gray')
    # plt.show()
    pred_vals = {}
    points = []

    (i,j) = get_point(score1)
    p1 = (i,j)
    pred_vals[p1] = pred1[j, i]
    points.append(p1)
    print(p1)

    testDepth1 = applyDT(points, pred_vals, 1, x_test, y_test) #applyDT1((i,j), pred1[j, i], x_test, y_test)
    print(accuracy(y_test, testDepth1))


    # ind1 = np.where(x_train[j,i,:]==1)
    # ind2 = np.where(x_train[j,i,:]==0)

    # sub1x = x_train[:,:,ind1[0]]
    # sub1y = y_train[ind1[0]]

    # sub2x = x_train[:,:,ind2[0]]
    # sub2y = y_train[ind2[0]]

    # score2,pred2 = scoreFeatures(sub1x,sub1y)

    # b = np.max(score2)
    # (k,l) = (np.where(score2 == b)[1][0], (np.where(score2 == b)[0][0]))
    # p2 = (k,l)
    # print(p2)

    # score3,pred3 = scoreFeatures(sub2x,sub2y)
    # c = np.max(score3)
    # (m,n) = (np.where(score3 == c)[1][0], (np.where(score3 == c)[0][0]))
    # p3 = (m,n)
    # print(p3)

    # testDepth2 = applyDT2(p1, p2, pred2[l, k], p3, pred3[n, m], x_test, y_test)
    # print(testDepth2)
