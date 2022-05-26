import data
import numpy as np
import matplotlib.pyplot as plt

def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh

def logreg_train(X, Y_, param_niter=2000, param_delta=0.01):

    c = max(Y_) + 1           #broj klasa
    N, d = X.shape            #dimenzije podataka
    W = np.random.randn(c, d) #inicijalne tezine
    b = np.zeros(c)           #inicijalizacija vektora b

    for i in range(param_niter):
        scores = np.matmul(X, np.transpose(W)) + b # N x C
        expscores = ... # N x C

        # nazivnik sofmaksa
        sumexp = ...  # N x 1
        softmax = [np.exp(i - max(i)) / np.sum(np.exp(i-max(i))) for i in scores] #ovo je zapravo probs ja mslm

        # logaritmirane vjerojatnosti razreda
        probs = ...  # N x C
        logprobs = ...  # N x C

        # gubitak
        loss = 0
        for j in range(N):
            loss += np.log(softmax[j][Y_[j]])
        loss = -1/N * loss # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        Y_oh = class_to_onehot(Y_)
        dL_ds = softmax - Y_oh  # N x C

        # gradijenti parametara
        grad_W = 1/N * np.matmul(np.transpose(dL_ds), X)  # C x D (ili D x C)
        grad_b = np.sum(np.transpose(dL_ds), axis=1)  # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def logreg_classify(X, W, b):
    # klasifikacijske mjere
    scores = np.matmul(X, np.transpose(W)) + b
    # vjerojatnosti razreda NxC
    probs = np.array([np.exp(i - max(i)) / np.sum(np.exp(i - max(i))) for i in scores])

    return probs

def logreg_decfun(w,b):
    def classify(X):
        return logreg_classify(X, w,b)[:, 0]
    return classify

if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    W, b = logreg_train(X, Y_)
    #print(w, b)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = [np.argmax(i) for i in probs] #1xN ili Nx1

    # report performance
    confusion_mat, acc, prec, rec = data.eval_perf_multi(Y, Y_)
    print(confusion_mat)
    print('accuracy:{:.3f}, recall:{:.3f}, precision:{:.3f}'.format(acc, rec, prec))

    # graph the decision surface -- CRTANJE NE RADI DOBRO
    decfun = logreg_decfun(W,b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()




