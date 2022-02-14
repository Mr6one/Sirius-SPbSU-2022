import numpy as np
import matplotlib.pyplot as plt

import torch

import time
from tqdm.notebook import tqdm
from IPython.display import clear_output
from collections import defaultdict


def train_model(model, n_epochs, optimizer, criterion, train_dataloader, val_dataloader, device):
    model.to(device)

    history = defaultdict(list)
    for epoch in range(n_epochs):
        start = time.time()

        model.train()
        epoch_loss = 0
        accuracy = []
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            accuracy.append((output.argmax(axis=1) == labels).cpu().numpy() * 1)

        accuracy = np.concatenate(accuracy)
        history['train_loss'].append(epoch_loss / len(train_dataloader))
        history['train_accuracy'].append(accuracy.mean())

        model.eval()
        epoch_loss = 0
        accuracy = []
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                epoch_loss += loss.item()
                accuracy.append((output.argmax(axis=1) == labels).cpu().numpy() * 1)

        accuracy = np.concatenate(accuracy)
        history['val_loss'].append(epoch_loss / len(val_dataloader))
        history['val_accuracy'].append(accuracy.mean())

        if (epoch + 1) % 10 == 0:
            x = (epoch + 1) / 10
            optimizer.param_groups[0]['lr'] *= x / (x + 1)

        clear_output()
        print('epoch number: {}'.format(epoch + 1))
        print('time per epoch: {}s'.format(np.round(time.time() - start, 2)))
        print('val epoch accuracy: {}'.format(np.round(history['val_accuracy'][-1], 2)))
        print('train epoch accuracy: {}'.format(np.round(history['train_accuracy'][-1], 2)))

        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        plt.plot(np.arange(epoch + 1) + 1, history['train_loss'], label='train', lw=2)
        plt.plot(np.arange(epoch + 1) + 1, history['val_loss'], label='validation', lw=2)
        plt.xlabel('Номер эпохи')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(epoch + 1) + 1, history['train_accuracy'], label='train', lw=2)
        plt.plot(np.arange(epoch + 1) + 1, history['val_accuracy'], label='validation', lw=2)
        plt.xlabel('Номер эпохи')
        plt.ylabel('Accuracy')
        plt.legend()
    
        plt.show()

    return model, history
