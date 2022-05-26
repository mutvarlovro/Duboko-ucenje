from pathlib import Path
import os
import time
import math
import numpy as np
import pickle
import torch
from torch import nn
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
import ssl
from sklearn.metrics import confusion_matrix

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = Path(__file__).parent / "datasets" / "CIFAR10"
SAVE_DIR = Path(__file__).parent / "out_4"


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


def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


class CovolutionalModel(nn.Module):
    def __init__(self, in_channels=3, conv1_width=16, conv2_width=32, fc1_width=256, fc2_width=128, class_count=10):
        super().__init__()

        # prvi CONV = > POOL
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_width, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        # drugi CONV = > POOL
        self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_width, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        #7x7
        # potpuno povezani slojevi
        self.fc1 = nn.Linear(in_features=conv2_width*7*7, out_features=fc1_width, bias=True) #32x7x7 -> 256
        self.fc2 = nn.Linear(in_features=fc1_width, out_features=fc2_width, bias=True) #256 -> 128
        self.fc_logits = nn.Linear(in_features=fc2_width, out_features=class_count, bias=True) # 128 -> 10

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
        #mozda ces trebatcivat u torch.tenzor, ako nije tenzor...?
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.maxpool1(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.maxpool2(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)

        h = self.fc2(h)
        h = torch.relu(h)

        logits = self.fc_logits(h) #PAZI ovo jos nije softmax!!!!

        return logits


def train(model: CovolutionalModel, train_loader, valid_loader, batch_size, lr, weight_decay):
    max_epochs = 10

    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()

    #print(model.conv1.weight.detach().numpy().shape)
    draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)

    for epoch in range(1, max_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model.forward(data)  # output bez softmaxa, za accuracy softmax napravit
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('epoch: {} [{}/{}], batch loss:{:.5f}'.format(epoch, batch_idx * len(data),
                                                                    len(train_loader.dataset),
                                                                    loss.item()))

        # --kraj epohe--
        draw_conv_filters(epoch, batch_idx, model.conv1.weight.detach().numpy(), SAVE_DIR)

        train_loss, train_acc = evaluate(model, train_loader, batch_size)
        val_loss, val_acc = evaluate(model, valid_loader, batch_size)
        print('epoch {} train accuracy:{:.4f}, valid accuracy:{:.4f}'.format(epoch, train_acc, val_acc))

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [scheduler.get_last_lr()]
        scheduler.step()

    plot_training_progress(SAVE_DIR, plot_data)
    return

def evaluate(model: CovolutionalModel, dataset, batch_size):
    cnt_correct = 0
    loss_avg = 0
    loss_function = nn.CrossEntropyLoss()
    num_of_batches = len(dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataset):
            output = model.forward(data)
            loss = loss_function(output, target)
            loss_avg += (loss.item() / num_of_batches)

            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            cf_matrix = confusion_matrix(target.numpy(), pred.numpy())
            cnt_correct += np.sum(np.diagonal(cf_matrix))

        acc = cnt_correct / (len(dataset) * batch_size)
    return loss_avg, acc


if __name__ == "__main__":

    batch_size = 50
    valid_size = 5000

    #mean: 0.49139968, 0.48215827, 0.44653124
    #std: 0.24703233, 0.24348505, 0.26158768
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=transform)

    trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])

    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                           download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True)

    model = CovolutionalModel()

    train(model, train_loader, valid_loader, batch_size, lr=1e-3, weight_decay=1e-4)
