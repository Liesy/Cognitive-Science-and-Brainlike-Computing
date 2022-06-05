import logging
import pickle
from argparse import ArgumentParser
from os import path
from typing import Tuple

import joblib
import numpy as np
import torch
from sklearn import svm, metrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from hmax import HMax

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

universal_patch_set = "./universal_patch_set.mat"


def load_data() -> Tuple[DataLoader, DataLoader]:
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])
    # Dataset
    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   transform=im_transform,
                                   download=True)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  transform=im_transform)
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    return train_loader, test_loader


def create_svm(dataMat, dataLabel, decision='ovr'):
    clf = svm.SVC(decision_function_shape=decision)
    clf.fit(dataMat, dataLabel)
    return clf


def main():
    train_dataloader, test_dataloader = load_data()  # img:(64,1,28,28)  label:(64)
    model = HMax(universal_patch_set).to(device)

    # train
    train_data, train_label = [], []
    for idx, (X, y) in enumerate(tqdm(train_dataloader)):
        s1, c1, s2, c2 = model.get_all_layers(X.to(device))
        for i in range(len(y)):
            train_data.append(c2[0][i])
            train_label.append(y[i].numpy())

    clf = create_svm(train_data, train_label, decision='ovr')
    logging.info("Model training completed !")

    clf_save_path = path.join(args.model_save_path, "clf.pkl")
    joblib.dump(clf, clf_save_path)
    logging.info(f"Model has been saved to {clf_save_path} !")

    # test
    test_data, test_label = [], []
    for idx, (X, y) in enumerate(tqdm(test_dataloader)):
        s1, c1, s2, c2 = model.get_all_layers(X.to(device))
        for i in range(len(y)):
            test_data.append(c2[0][i])
            test_label.append(y[i].numpy())

    f1 = joblib.load(clf_save_path)
    y_test_pred = f1.predict(test_data)
    ov_acc = metrics.accuracy_score(y_test_pred, test_label)
    logging.info(f"the ov_acc is {ov_acc}")

    layers_path = path.join(args.model_save_path, "output_all_layers.pkl")
    logging.info(f"Saving output of all layers to: {layers_path}")
    with open(layers_path, 'wb') as f:
        pickle.dump(dict(s1=s1, c1=c1, s2=s2, c2=c2), f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data_path', type=str, default='../')
    parser.add_argument('-model_save_path', type=str, default='./')
    parser.add_argument('-batch_size', type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

    main()
