# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Original code from GitHub: 
https://github.com/amazon-science/minimax-fair/blob/main/src/torch_wrapper.py

Credit goes to: @gillwesl
"""
from abc import ABC
import torch
import os
from tempfile import TemporaryDirectory
from torch import nn
from tqdm import tqdm

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class TorchMLP(nn.Module, ABC):
    """
    A feedforward NN in pytorch using ReLU activiation functions between all layers but the last which uses a sigmoid activiation function. Supports an arbitrary number of hidden layers.
    """

    def __init__(self, h_sizes, out_size=1, task='classification'):
        """
        :param h_sizes: input sizes for each hidden layer (including the first)
        :param out_size: defaults to 1 for binary and represents the (positive class probability?)
        :param task: 'classification' or 'regression'
        """
        super(TorchMLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = self.relu(layer(x))
        output = self.out(x)  # Sigmoid applied later in BCEWithLogitLoss, and applied automatically in predict_proba
        return output.double()


class MLPClassifier(TorchMLP):
    """
    Wrapper class so our MLP looks like an sklearn model
    """

    def __init__(self, h_sizes, lr=0.0001, momentum=0.9, weight_decay=0,
                 n_epochs=50):
        super(TorchMLP, self).__init__()
        self.model = TorchMLP(h_sizes)
        self.model.double()  # set model type to double
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.n_epochs = n_epochs

    def to(self, device):
        self.model.to(device)

    def fit(self, X, y, X_val=None, y_val=None, batch_size=128,
            loss_type='BCE', tmp_file_dir='./tmp'):
        """
        Fits the model using the entire sample data as the batch size
        """
        print("Fitting with device={}".format(device))
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).double()
        X, y = X.to(device), y.to(device)
        self.model.to(device)
        self.model.train()  # Puts model in training mode so it updates itself

        # Construct a trainloader so we can do SGD
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, batch_size=batch_size)

        # Binary Cross-Entropy Loss
        criterion = nn.BCEWithLogitsLoss()

        # If validation data is supplied, keep the weights on the best epoch
        keep_best_epoch = False
        if X_val is not None and y_val is not None:
            keep_best_epoch = True
            best_epoch = 0
            best_epoch_loss = float('inf')

        # Temporary directory for storing data during training
        self.tmp_file_dir = tmp_file_dir
        if not os.path.exists(tmp_file_dir):
            os.makedirs(tmp_file_dir)
        with TemporaryDirectory(dir=self.tmp_file_dir) as tmp_dir:
            print("Created temporary directory: {}".format(tmp_dir))
            for epoch in tqdm(range(self.n_epochs)):
                for X, y in loader:
                    self.optimizer.zero_grad()  # Set gradients to 0 before back propagation for this epoch
                    # Forward pass
                    y_pred = self.model(X)
                    # Compute Loss
                    loss = criterion(y_pred.squeeze(), y)
                    #print(f'Epoch {epoch}: train loss: {loss.item()}')

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                 # If validation data is given, compute the validation loss
                if keep_best_epoch:
                    val_loss = self.compute_val(X_val, y_val, criterion)
                    if val_loss < best_epoch_loss:
                        best_epoch = epoch
                        best_epoch_loss = val_loss
                        print("Best epoch={} with loss={}".format(epoch, best_epoch_loss))
                        self.save_weights(tmp_dir)
            
            # Done training -- load the best weights for the model again
            if keep_best_epoch:
                self.load_weights(tmp_dir)

        self.model.to(torch.device('cpu'))
        return self

    def save_weights(self, dirname):
        weights_fp = os.path.join(dirname, 'weights.pt')
        torch.save(self.state_dict(), weights_fp)
        return

    def load_weights(self, dirname):
        weights_fp = os.path.join(dirname, 'weights.pt')
        state_dict = torch.load(weights_fp)
        self.load_state_dict(state_dict)
        return

    def compute_val(self, X_val, y_val, criterion):
        """
        Computes validation loss for keeping the best epoch.
        """
        self.model.eval()
        X = torch.from_numpy(X_val)
        y = torch.from_numpy(y_val).double()
        with torch.no_grad():
            val_pred = self.model(X)
            val_loss = criterion(val_pred.squeeze(), y)

        return val_loss

    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        self.model.eval()  # Puts the model in evaluation mode so calls to forward do not update it
        with torch.no_grad():  # Disables automatic gradient updates from pytorch since we are just evaluating
            return torch.sigmoid(self.model(torch.from_numpy(X))).numpy().squeeze()  # Apply sigmoid manually

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return (self.predict_proba(X) > 0.5).astype(int)