from pathlib import Path
import os
import time
import math
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io


DATA_DIR = Path(__file__).parent / "datasets" / "MNIST"
SAVE_DIR = Path(__file__).parent / "out_3"


class CovolutionalModel(nn.Module):
    def __init__(self, in_channels=1, conv1_width=16, conv2_width=32, fc1_width=512, class_count=10):
        super().__init__()

        # prvi CONV = > POOL = > RELU
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_width, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # drugi CONV = > POOL = > RELU
        self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_width, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        #7x7
        # potpuno povezani slojevi
        self.fc1 = nn.Linear(in_features=conv2_width*7*7, out_features=fc1_width, bias=True)
        self.fc_logits = nn.Linear(in_features=fc1_width, out_features=class_count, bias=True) # logits

        # parametri su inicijalizirani pozivima Conv2d i Linear, ali možemo ih drugačije inicijalizirati
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.maxpool1(h)
        h = torch.relu(h)

        h = self.conv2(h)
        h = self.maxpool2(h)
        h = torch.relu(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        logits = self.fc_logits(h) #PAZI ovo jos nije softmax!

        return logits

def train(model: CovolutionalModel, train_loader, test_loader, lr, weight_decay):
    max_epochs = 6

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()

    draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)

    train_losses = [0 for i in range(max_epochs)]
    for epoch in range(1, max_epochs+1):
        epoch_loss = 0
        cnt_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model.forward(data) #output bez softmaxa, za accuracy softmax napravit
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            cnt_correct += (pred == target).sum().item()

            if batch_idx % 10 == 0:
                print('epoch: {} [{}/{}], batch loss:{:.5f}'.format(epoch, batch_idx*len(data),
                                                                    len(train_loader.dataset), loss.item()))
            epoch_loss += loss.item()
        #--kraj epohe--
        scheduler.step()

        train_losses[epoch-1] = epoch_loss / len(train_loader)
        draw_conv_filters(epoch, batch_idx, model.conv1.weight.detach().numpy(), SAVE_DIR)

        print('epoch {} train accuracy: {:.4f}'.format(epoch, cnt_correct / len(train_loader.dataset))) #accuracy epohe
        test(model, test_loader, epoch)

    plt.plot([i for i in range(1, len(train_losses)+1)], train_losses)
    save_path = os.path.join(SAVE_DIR, 'training_plot.png')
    plt.savefig(save_path)

    return

def test(model: CovolutionalModel, test_loader, epoch):
    cnt_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model.forward(data)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            cnt_correct += (pred == target).sum().item()

    print('epoch: {} test accuracy: {:.4f}'.format(epoch, cnt_correct / len(test_loader.dataset)))
    return


def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)


if __name__ == "__main__":
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)
    batch_size = 50
    max_epochs = 8

    train_loader = torch.utils.data.DataLoader(
        MNIST(DATA_DIR, train=True, download=True,
              transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNIST(DATA_DIR, train=False, download=True,
              transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=batch_size, shuffle=True)

    model = CovolutionalModel()

    train(model, train_loader, test_loader, lr=1e-3, weight_decay=1e-3)
