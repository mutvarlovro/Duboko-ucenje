import torch
import torch.nn as nn
import torch.optim as optim

def original_linreg():
    #y = a * x + b
    # podaci i parametri, inicijalizacija parametara
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    X = torch.tensor([1, 2])
    Y = torch.tensor([3, 5])

    # optimizacijski postupak: gradijentni spust
    optimizer = optim.SGD([a, b], lr=0.1)

    for i in range(100):
        # afin regresijski model
        Y_ = a*X + b

        diff = (Y-Y_)
        # kvadratni gubitak
        loss = torch.sum(diff**2)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()

        #ispis
        print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')


def modified_linreg(X, Y, epochs=100, lr=0.1):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    # optimizacijski postupak: gradijentni spust
    optimizer = optim.SGD([a, b], lr=lr)

    for i in range(epochs):
        Y_ = a*X + b

        diff = (Y - Y_)

        # kvadratni gubitak
        loss = torch.mean(diff ** 2)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        print(f'Step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
        print('Pytorch: gradijent gubitka po a:{:.3f}, gradijent gubitka po b:{:.3f}'.format(a.grad.item(), b.grad.item()))
        my_a = 2 * torch.mean((Y_ - Y)*X)
        my_b = 2 * torch.mean((Y_ - Y)*1)
        print('Analitcki: gradijent gubitka po a:{:.3f}, gradijent gubitka po b:{:.3f}'.format(my_a, my_b))

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()

if __name__ == "__main__":
    #originalni kod
    #original_linreg()

    #modificirani kod
    X = torch.tensor([0, 4, 5, 7])
    Y = torch.tensor([0, 4, 5, 7])

    modified_linreg(X, Y, 200, 0.01)