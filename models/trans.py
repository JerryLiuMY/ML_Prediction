import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
torch.manual_seed(0)
np.random.seed(0)


def fit_transformer(train_data, valid_data, params, window_path):
    """ Fit transformer model and report evaluation metric
    :param train_data: generator of training data
    :param valid_data: generator of validation data
    :param params: dictionary of parameters
    :param window_path: path to the particular window
    :return model: fitted model
    """

    # load parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers, nhead = params["num_layers"], params["nhead"]
    d_model, dropout = params["d_model"], params["dropout"]
    epochs, lr = params["epochs"], params["lr"]

    # build model
    model = TransAm(num_layers, nhead, d_model, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(epochs):
        model.train()
        for train_X, train_y, _ in train_data:
            train_X.to(device), train_y.to(device)
            optimizer.zero_grad()
            pred_y = model(train_X)
            loss = criterion(pred_y, train_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
        scheduler.step()

    # save model
    save_path = os.path.join(window_path, "model")
    torch.save(model, os.path.join(save_path, "model.pth"))

    # model evaluation
    mse_li = []
    model.eval()
    for valid_X, valid_y, _ in valid_data:
        with torch.no_grad():
            mse_li.append(criterion(model(valid_X), valid_y))

    metric = {"MSE": np.mean(mse_li)}

    return model, metric


def pre_transformer(model, test_data):
    """ Make predictions with transformer model
    :param model: fitted model
    :param test_data: generator of testing data
    :return target: predicted target
    """

    target = pd.DataFrame(columns=["target"])
    model.eval()
    with torch.no_grad():
        for test_X, _, index in test_data:
            output = model(test_X).cpu().detach().numpy().reshape(-1)
            target = pd.concat([target, pd.DataFrame(data=output, index=index, columns=["target"])], axis=0)

    return target


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, num_layers, nhead, d_model, dropout):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        self.layer = nn.TransformerEncoderLayer(nhead=nhead, d_model=d_model, dropout=dropout)

        # build transformer
        self.embedding = nn.Linear(798, d_model)
        self.position = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.embedding(src)
        src = self.position(src)
        output = self.encoder(src, self.src_mask)
        output = torch.mean(output, dim=0)
        output = self.decoder(output)

        return output

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        return mask
