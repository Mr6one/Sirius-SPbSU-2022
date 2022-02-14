import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import torch

from sklearn import model_selection
from sklearn.metrics import confusion_matrix


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def process_images(images, mb_kernel_size=-1, gb_kernel_size=-1):

    if mb_kernel_size == -1 and gb_kernel_size == -1:
        return images

    processed_images = []
    for image in images:

        if mb_kernel_size != -1:
            image = cv2.medianBlur(image, mb_kernel_size)
        if gb_kernel_size != -1:
            image = cv2.GaussianBlur(image, gb_kernel_size)

        processed_images.append(image)

    processed_images = np.array(processed_images)
    return processed_images


def train_test_split(data, test_size=0.2, random_state=47):
    images, labels = data['images'], data['labels']

    train_images, test_images, train_labels, test_labels = model_selection.train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    train_data = {'images': train_images, 'labels': train_labels}
    test_data = {'images': test_images, 'labels': test_labels}

    return train_data, test_data


def plot_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    

def predict_on_dataloader(model, dataloader):
    device = next(iter(model.parameters()))[0].device

    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            output = model(images)
            y_pred.append(output.argmax(axis=1).detach().cpu().numpy())

    y_pred = np.concatenate(y_pred)
    return y_pred


def stack_models(models, dataloader, path, device):
    predictions = []
    for path_to_weight in listdir_fullpath(path):
        model_name = path_to_weight.split(os.sep)[-1].split('_')[0]
        if model_name in models.keys():
            model = models[model_name]
            model.load(path_to_weight)
            model.to(device)

            y_pred = predict_on_dataloader(model, dataloader)
            predictions.append(y_pred[:, None])

    predictions = np.concatenate(predictions, axis=1)
    return predictions


def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id, Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)
