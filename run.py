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

    # tree2 = decisionTree(depth=2)
    # tree2.fit(x_train, y_train)
    # testDepth2 = tree2.predict(x_test, y_test)
    # print(tree2.accuracy(y_test, testDepth2))

    # tree3 = decisionTree(depth=3)
    # tree3.fit(x_train, y_train)
    # testDepth3 = tree3.predict(x_test, y_test)
    # print(tree3.accuracy(y_test, testDepth3))

