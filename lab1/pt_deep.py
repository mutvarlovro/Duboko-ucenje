import numpy as np
import torch
from torch import nn
import torch.optim as optim
import data
import matplotlib.pyplot as plt

class PTDeep(nn.Module):
    def __init__(self, layers, activation_function):
        super().__init__()

        self.af = activation_function
        weights_list = []
        biases_list = []

        for i in range(len(layers)-1):
            W = nn.Parameter(torch.randn((layers[i], layers[i+1])), requires_grad=True)
            b = nn.Parameter(torch.zeros(layers[i+1]), requires_grad=True)
            weights_list.append(W)
            biases_list.append(b)

        self.weights = nn.ParameterList(weights_list)
        self.biases = nn.ParameterList(biases_list)

    def count_params(self):
        cnt = 0
        for name, param in ptdeep.named_parameters():
            cnt += torch.numel(param)
            print('Ime:{}, dimenzije:{}'.format(name, param.shape))
        print('Ukupan broj parametara:{}'.format(cnt))
        return

    def forward(self, X):
        next = X
        i = 0
        for W, b in zip(self.weights, self.biases):
            next = torch.mm(next, W) + b
            if i < (len(self.weights)-1):
                next = self.af(next)
            i += 1

        #nakraju se primjeni softmax
        return torch.softmax(next, dim=1)

    def get_loss(self, X, Yoh_):
        aposteriori_probs = self.forward(X)
        true_probs = aposteriori_probs * Yoh_
        true_probs_sum = torch.sum(true_probs, dim=1)
        log_loss = - torch.mean(torch.log(true_probs_sum))
        return log_loss

def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0, ispis=False):
    # inicijalizacija optimizatora
    optimizer = optim.SGD(params=model.parameters(), lr=param_delta, weight_decay=param_lambda)

    # petlja učenja
    losses = []
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_)
        if ispis:
            losses.append(loss.detach())

        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Step: {}, loss:{}'.format(i, loss))
        optimizer.zero_grad()
    if ispis:
        return losses


def eval(model, X):
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float)
    probs = model.forward(X).detach()
    return probs.numpy()

def logreg_decfun(model, X):
    def classify(X):
        #return np.argmax(eval(model, X), axis=1) #za viseklasnu ali ruznije crta
        return eval(model, X)[:, 0]
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    #X, Y_ = data.sample_gmm_2d(3, 3, 30)
    #X, Y_ = data.sample_gmm_2d(4, 2, 40)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y_)

    # definiraj model:
    #layers = [2,3]
    #layers = [2,4,5,4,2]
    #layers = [2,2]
    #layers = [2,10,2]
    layers = [2,10,10,2]
    ptdeep = PTDeep(layers, torch.nn.ReLU())
    #ptdeep = PTDeep(layers, torch.nn.Sigmoid())

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptdeep, torch.tensor(X, dtype=torch.float), torch.tensor(Yoh_, dtype=torch.float),
          param_niter=2*10**4, param_delta=0.01, param_lambda=10**-3) # probaj za razlicite lambde!

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptdeep, X)
    Y_pred = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, M = data.eval_perf_multi(Y_pred, Y_)
    print('Accuracy:{}'.format(accuracy))
    print('Recall, precision (po razredima):\n{}'.format(pr))
    print('Matrica konfuzije:\n{}'.format(M))

    ptdeep.count_params()

    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decfun = logreg_decfun(ptdeep, X)
    data.graph_surface(decfun, bbox, offset=0.5, width=1024, height=1024)

    # graph the data points
    data.graph_data(X, Y_, Y_pred, special=[])
    plt.show()