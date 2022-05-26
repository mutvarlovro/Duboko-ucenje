import numpy as np
import torch
from torch import nn
import torch.optim as optim
import data
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        super().__init__()
        self.W = nn.Parameter(torch.randn((D, C)), requires_grad=True) #tenzor DxC sa sluc. vrij. iz distribucije N(0,1)
        self.b = nn.Parameter(torch.zeros(C), requires_grad=True)

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        o = torch.mm(X, self.W) + self.b
        aposteriori_probs = torch.softmax(o, dim=1)
        return aposteriori_probs

    def get_loss(self, X, Yoh_):
        aposteriori_probs = self.forward(X)
        # formulacija gubitka
        # koristiti: torch.log, torch.mean, torch.sum
        true_probs = aposteriori_probs * Yoh_
        true_probs_sum = torch.sum(true_probs, dim=1)
        log_loss = - torch.mean(torch.log(true_probs_sum))
        return log_loss


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0):
    # inicijalizacija optimizatora
    optimizer = optim.SGD(params=model.parameters(), lr=param_delta, weight_decay=param_lambda)

    # petlja učenja
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_) + param_lambda*(torch.linalg.norm(model.W)) # regularizacija
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Step: {}, loss:{}'.format(i, loss))

        optimizer.zero_grad()


def eval(model, X):
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X = torch.tensor(X, dtype=torch.float)
    probs = model.forward(X).detach()
    return probs.numpy()

def logreg_decfun(model, X):
    def classify(X):
        return np.argmax(eval(model, X), axis=1) #za viseklasnu ali onda nema probabilisticku interpretaciju boja
        #return eval(model, X)[:, 0]
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(3, 3, 100) #ncomponents, nclasses, nsamples
    Yoh_ = data.class_to_onehot(Y_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, torch.tensor(X, dtype=torch.float), torch.tensor(Yoh_, dtype=torch.float),
          param_niter=3000, param_delta=0.1, param_lambda=0) # probaj za razlicite lambde!

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y_pred = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, M = data.eval_perf_multi(Y_pred, Y_)
    print('Accuracy:{}'.format(accuracy))
    print('Recall, precision (po razredima):\n{}'.format(pr))
    print('Matrica konfuzije:\n{}'.format(M))

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decfun = logreg_decfun(ptlr, X)
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y_pred, special=[])
    plt.show()