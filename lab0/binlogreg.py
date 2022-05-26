import data
import numpy as np
import matplotlib.pyplot as plt


def binlogreg_train(X, Y_, param_niter=1000, param_delta=0.01):#PROVJERI OVAJ CIJELI KOD...
    N, d = X.shape #dimenzija podataka
    w = np.random.randn(d)
    b = 0

    for i in range(param_niter):
        # klasifikacijske mjere, Nx1
        scores = np.dot(X, w) + b

        # vjerojatnosti razreda c_1, Nx1
        aposteriori = 1 / (1 + np.exp(-scores))

        # gubitak
        loss = -1/N * (np.sum(np.log(aposteriori[Y_ == 1])) + np.sum(np.log(1 - aposteriori[Y_ == 0])))

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = aposteriori - Y_
        # gradijenti parametara
        grad_w = 1/N * np.matmul(dL_dscores, X)
        grad_b = 1/N * np.sum(dL_dscores)
        # poboljšani parametri
        w = w - param_delta*grad_w
        b = b - param_delta*grad_b

    return w, b

def binlogreg_classify(X, w, b):
    # klasifikacijske mjere, Nx1
    scores = np.dot(X, w) + b
    # vjerojatnosti razreda c_1, Nx1
    probs = 1 / (1 + np.exp(-scores))
    return probs

def binlogreg_decfun(w,b):
    def classify(X):
        return binlogreg_classify(X, w,b)
    return classify

if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs >= 0.5).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print('accuracy:{}, recall:{}, precision:{:.3f}, AP:{:.3f}'.format(accuracy, recall, precision, AP))

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
