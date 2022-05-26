import numpy as np
import data
import matplotlib.pyplot as plt

def fcann2_train(X, Y_, hidden_dim, param_niter, param_delta, param_lambda=0):
    c = max(Y_) + 1  # broj klasa
    N, d = X.shape   # dimenzije ulaznih podataka
    W0 = np.random.randn(d, hidden_dim)
    b0 = np.zeros(hidden_dim)
    W1 = np.random.randn(hidden_dim, c)
    b1 = np.zeros(c)

    for i in range(param_niter):

        s1 = np.matmul(X, W0) + b0
        h1 = np.maximum(s1, 0)
        s2 = np.matmul(h1, W1) + b1
        probs = [np.exp(i - max(i)) / np.sum(np.exp(i-max(i))) for i in s2]

        #loss
        Yoh = data.class_to_onehot(Y_)
        filtered_probs = probs * Yoh #tu se prebaci nazad u numpy array
        filtered_probs = np.sum(filtered_probs, axis=1)
        loss = - np.mean(np.log(filtered_probs)) + param_lambda*(np.linalg.norm(W1) + np.linalg.norm(W1)) #REGULARIZACIJA

        if i % 100 == 0:
            print('Step:{}, loss:{}'.format(i, loss))

        # derivacije
        dL_ds2 = probs - Yoh
        dL_ds2 = 1/N * dL_ds2

        # layer 1 grad
        w1_grad = np.matmul(np.transpose(h1), dL_ds2) # HxC mora bit tako jer je W1 HxC
        b1_grad = np.sum(dL_ds2, axis=0)

        # dL/ds
        dL_ds1 = np.matmul(dL_ds2, np.transpose(W1)) # NxC * CxH => NxH
        dL_ds1[s1 <= 0] = 0

        # layer 0 grad
        w0_grad = np.matmul(np.transpose(X), dL_ds1) # DxH mora bit tako jer je W0 DxH
        b0_grad = np.sum(dL_ds1, axis=0)

        # adjusting parameters, including REGULARIZATION!
        W1 = W1*(1-param_delta*param_lambda) - param_delta*w1_grad
        b1 = b1 - param_delta*b1_grad
        W0 = W0*(1-param_delta*param_lambda) - param_delta*w0_grad
        b0 = b0 - param_delta*b0_grad

    return W1, b1, W0, b0


def fcann2_classify(X, W1, b1, W0, b0):
    s1 = np.matmul(X, W0) + b0
    h1 = np.maximum(s1, 0)
    s2 = np.matmul(h1, W1) + b1
    num = np.exp(s2)
    denum = np.sum(num, axis=1)
    denum = denum.reshape(-1, 1)
    return num/denum


def fcann2_decfun(X, W1, b1, W0, b0):
    def classify(X):
        #return np.argmax(fcann2_classify(X, W1, b1, W0, b0), axis=1) za viseklasnu, ali ruznije izgleda
        return fcann2_classify(X, W1, b1, W0, b0)[:,0]
    return classify

if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(6,2,10)

    # train the model
    W1, b1, W0, b0 = fcann2_train(X, Y_, hidden_dim=5, param_niter=10**4, param_delta=0.05, param_lambda=10**-3)
    probs = fcann2_classify(X, W1, b1, W0, b0)
    Y = np.argmax(probs, axis=1)

    decfun = fcann2_decfun(X, W1, b1, W0, b0)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y, special=[])
    plt.show()



