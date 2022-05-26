import data
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model.fit(X,Y_)

    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

    def support(self):
        return self.model.support_


def svm_decfun(model:KSVMWrap, X):
    def classify(X):
        return model.get_scores(X)[:,1]
    return classify

if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    ksvm = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')
    Y = ksvm.predict(X)
    print('Accuracy:{:.3f}'.format(accuracy_score(Y_, Y)))
    print('Precision:{:.3f}'.format(precision_score(Y_, Y)))
    print('Recall:{:.3f}'.format(recall_score(Y_, Y)))
    print('Average precision:{:.3f}'.format(average_precision_score(Y_, Y)))

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decfun = svm_decfun(ksvm, X)
    data.graph_surface(decfun, bbox, offset=0.5)

    support_vectors = ksvm.support()
    data.graph_data(X, Y_, Y, special=support_vectors)
    plt.show()





