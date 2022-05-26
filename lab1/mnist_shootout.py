import torch
import torchvision
import matplotlib.pyplot as plt
import pt_deep
import data
import numpy as np
from torch.nn.functional import normalize
import torch.optim as optim


def get_data():
    #dataset_root = '/tmp/mnist'  # change this to your preference
    dataset_root = 'data/'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    return x_train, y_train, x_test,y_test

def zad1(x_train, y_train,param_niter, param_delta, param_lambda):

    #priprema podataka za model
    x_train_flat = torch.flatten(x_train, start_dim=1)
    Yoh_train = data.class_to_onehot(y_train)

    layers = [784, 10]

    model = pt_deep.PTDeep(layers, torch.nn.ReLU())
    pt_deep.train(model, x_train_flat, torch.tensor(Yoh_train, dtype=torch.float), param_niter=param_niter, param_delta=param_delta, param_lambda=param_lambda)

    probs = pt_deep.eval(model, x_train_flat)
    Y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(Y_pred, y_train)
    print('Accuracy:{}'.format(accuracy))

    for w in model.weights:
        w = w.detach()
        for i in range(w.shape[1]):
            plt.imshow(w[:, i].reshape(28, 28), cmap=plt.get_cmap('gray'))
            plt.title(str(i))
            plt.show()
    return

def zad2(x_train, y_train, x_test, y_test, layers, param_niter, param_delta, param_lambda):
    # priprema podataka za model
    x_train_flat = torch.flatten(x_train, start_dim=1)
    x_test_flat = torch.flatten(x_test, start_dim=1)

    if len(layers) > 2:
        x_train_flat = normalize(x_train_flat, p=1)
        x_test_flat = normalize(x_test_flat, p=1)

    Yoh_train = data.class_to_onehot(y_train)

    model = pt_deep.PTDeep(layers, torch.nn.ReLU())
    loss_log = pt_deep.train(model, x_train_flat, torch.tensor(Yoh_train, dtype=torch.float), param_niter=param_niter,
                             param_delta=param_delta, param_lambda=param_lambda, ispis=True)
    #train
    probs = pt_deep.eval(model, x_train_flat)
    Y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(Y_pred, y_train)
    pr = np.array(pr)
    recall = np.mean(pr[:, 0])
    precision = np.mean(pr[:, 1])
    print('Train accuracy:{}'.format(accuracy))
    print('Train recall:{}'.format(recall))
    print('Train precision:{}'.format(precision))

    #test
    probs = pt_deep.eval(model, x_test_flat)
    Y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(Y_pred, y_test)
    pr = np.array(pr)
    recall = np.mean(pr[:, 0])
    precision = np.mean(pr[:, 1])
    print('Test accuracy:{}'.format(accuracy))
    print('Test recall:{}'.format(recall))
    print('Test precision:{}'.format(precision))

    #vizualiziraj kako se gubitak krece
    plt.plot([i for i in range(len(loss_log))], loss_log)
    plt.show()
    return

def zad3(x_train, y_train, x_test, y_test, param_niter, param_lambda):
    # priprema podataka za model
    x_train_flat = torch.flatten(x_train, start_dim=1)
    #x_train_flat = normalize(x_train_flat, p=1)
    x_test_flat = torch.flatten(x_test, start_dim=1)
    #x_test_flat = normalize(x_test_flat, p=1)
    Yoh_train = data.class_to_onehot(y_train)

    layers = [784, 10]

    model = pt_deep.PTDeep(layers, torch.nn.ReLU())
    pt_deep.train(model, x_train_flat, torch.tensor(Yoh_train, dtype=torch.float), param_niter=param_niter, param_delta=0.1,
                  param_lambda=param_lambda)

    # train
    probs = pt_deep.eval(model, x_train_flat)
    Y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(Y_pred, y_train)
    pr = np.array(pr)
    recall = np.mean(pr[:, 0])
    precision = np.mean(pr[:, 1])
    print('Train accuracy:{}'.format(accuracy))
    print('Train recall:{}'.format(recall))
    print('Train precision:{}'.format(precision))

    # test
    probs = pt_deep.eval(model, x_test_flat)
    Y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(Y_pred, y_test)
    pr = np.array(pr)
    recall = np.mean(pr[:, 0])
    precision = np.mean(pr[:, 1])
    print('Test accuracy:{}'.format(accuracy))
    print('Test recall:{}'.format(recall))
    print('Test precision:{}'.format(precision))
    return


def zad67(x_train, y_train, param_niter, task):
    #priprema podataka za model
    x_train_flat = torch.flatten(x_train, start_dim=1)
    Yoh_train = data.class_to_onehot(y_train)

    layers = [784, 10]
    model = pt_deep.PTDeep(layers, torch.nn.ReLU())

    def train(model, X, Yoh_, param_niter):
        # inicijalizacija optimizatora
        optimizer = optim.Adam(params=model.parameters(), lr=10**-2) #originalno 10**-4 al mi onda sporo krene konvergencija, ovak se dobro vidi razlika
        if task == 7:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-10**-4)

        # petlja učenja
        for i in range(param_niter):
            loss = model.get_loss(X, Yoh_)

            loss.backward()
            optimizer.step()
            if task == 7:
                scheduler.step()

            if i % 100 == 0:
                print('Step: {}, loss:{}'.format(i, loss))
            optimizer.zero_grad()

    train(model, x_train_flat, torch.tensor(Yoh_train, dtype=torch.float),
          param_niter)

    # train
    probs = pt_deep.eval(model, x_train_flat)
    Y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(Y_pred, y_train)
    pr = np.array(pr)
    recall = np.mean(pr[:, 0])
    precision = np.mean(pr[:, 1])
    print('Train accuracy:{}'.format(accuracy))
    print('Train recall:{}'.format(recall))
    print('Train precision:{}'.format(precision))
    return

def zad8(x_train, y_train):
    # priprema podataka za model
    x_train_flat = torch.flatten(x_train, start_dim=1)
    Yoh_train = data.class_to_onehot(y_train)

    layers = [784, 10]
    model = pt_deep.PTDeep(layers, torch.nn.ReLU())
    loss = model.get_loss(x_train_flat, torch.tensor(Yoh_train, dtype=torch.float))
    print('Gubitak slučajno inicijaliziranog modela:{}'.format(loss))


if __name__ == '__main__':
    # dohvacanje podataka i dimenzija
    x_train, y_train, x_test, y_test = get_data()
    N = x_train.shape[0] # broj podataka za ucenje
    D = x_train.shape[1] * x_train.shape[2] # dimenzija jednog ulaznog podatka 28x28
    C = y_train.max().add_(1).item() # broj klasa
    #zad1
    #zad1(x_train, y_train, param_niter=1500, param_delta=0.05, param_lambda=0)

    #zad2
    #layers = [784, 10] # za ovakvu strukturu treba maknut normalizaciju podataka
    #layers = [784, 100, 10]
    #zad2(x_train, y_train, x_test, y_test, layers, param_niter=2000, param_delta=0.1, param_lambda=0)

    #zad3
    #zad3(x_train, y_train, x_test, y_test, param_niter=2000, param_lambda=0.1)

    #zad6
    #zad67(x_train, y_train, param_niter=2000, task=6)

    #zad7
    #zad67(x_train, y_train, param_niter=2000, task=7)

    #zad8 - interpretaciju znas
    #zad8(x_train, y_train)
