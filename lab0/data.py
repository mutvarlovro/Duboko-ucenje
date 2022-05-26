import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Random2DGaussian:

    d0min = 0
    d0max = 10
    d1min = 0
    d1max = 10
    scalecov = 5

    def __init__(self):
        dw0, dw1 = self.d0max - self.d0min, self.d1max - self.d1min
        mean = (self.d0min, self.d1min)
        mean += np.random.rand(2)*(dw0, dw1) #mean += np.random.random_sample(2) * (dw0, dw1) ...tak su oni stavili u rj
        #u slijedeca 3 retka defincija svojstvenih vektora, neznam zasto bas tako..
        eigvals = np.random.random_sample(2)#logicno, jer radimo s 2D podacima
        eigvals *= (dw0 / self.scalecov, dw1 / self.scalecov)
        eigvals **= 2
        D = np.diag(eigvals)
        theta = np.random.rand() * 2*np.pi
        R = [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
        Sigma = np.dot(np.dot(np.transpose(R), D), R)
        self.get_sample = lambda n: np.random.multivariate_normal(mean, Sigma, n)#probaj ju izvadit iz konstr


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)
    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])



def graph_data(X, Y_, Y, special=[]):
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s')


def sample_gauss_2d(nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(nclasses):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_

def eval_perf_binary(Y, Y_):
    #tocnost = (tp + tn) / all
    #preciznost = tp / (tp + fp)
    #odziv = tp / (tp + fn)
    c_m = confusion_matrix(Y_, Y)
    tn = c_m[0, 0]
    fp = c_m[0, 1]
    fn = c_m[1,0]
    tp = c_m[1,1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, recall, precision

def eval_perf_multi(Y,Y_):
    c_m = confusion_matrix(Y_, Y)
    acc = accuracy_score(Y_, Y)
    prec = precision_score(Y_, Y, average='micro')
    rec = recall_score(Y_, Y, average='micro')
    return c_m, acc, prec, rec

# def eval_AP(Yr):#to je moj eval_AP, ali ne radi skroz isto
#     n = len(Yr)
#     pos = np.sum(Yr)
#     ap = 0
#     for i in range(len(Yr)):
#         y_temp = np.copy(Yr)
#         y_temp[i:] = 1
#         r, prec, a = eval_perf_binary(y_temp, Yr)
#         if Yr[i]:
#             ap += prec
#     return ap/pos

def eval_AP(ranked_labels):
    """Recovers AP from ranked labels"""

    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0
    # IPython.embed()
    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if x:
            sumprec += precision

        # print (x, tp,tn,fp,fn, precision, recall, sumprec)
        # IPython.embed()

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__ == '__main__':
    np.random.seed(100)
    G = Random2DGaussian()
    X = G.get_sample(100)
    print(X)
    plt.scatter(X[:,0], X[:,1])
    plt.show()

# if __name__ == "__main__":
#     np.random.seed(100)
#
#     # get the training dataset
#     X, Y_ = sample_gauss_2d(2, 100)
#
#     # get the class predictions
#     Y = myDummyDecision(X) > 0.5
#
#     # graph the data points
#     graph_data(X, Y_, Y)
#
#     # show the results
#     plt.show()