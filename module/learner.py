import module.utils as u
import torch
from torch.utils.data import TensorDataset, DataLoader
from module.ANN import ANN
import torch.nn.functional as F

VALID_PERCENTAGE = 0.33
bs = 400
lr = 0.5
epochs = 5


def get_model():
    model = ANN()
    return model, torch.optim.SGD(model.parameters(), lr=lr)


def accuracy(out, y):
    out = torch.argmax(out, dim=1)
    return (out == y).float().mean()


def get_data_for_learner():
    """Return data ready to be used in the model.

    Returns:
        train_dl: DataLoader to iterate on.
        x_valid: validation set.
        y_valid: target validation set.
        test: data for final test.
    """
    train, test = u.get_data()
    train = u.preprocess_data(train, True)
    test = u.preprocess_data(test, False)
    X_train, X_valid, y_train, y_valid = u.train_validation_split(train, VALID_PERCENTAGE)
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (X_train, y_train, X_valid, y_valid))
    test = torch.tensor(test)

    x_train = x_train.float()
    y_train = y_train.long()
    test = test.float()
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    x_valid = x_valid.float()
    y_valid = y_valid.long()

    return train_dl, x_valid, y_valid, test


def get_model_trained(model, opt, loss_func, train_dl, x_valid, y_valid):
    """Return the trained model.

    Parameters:
        model: neural network to be trained.
        opt: optimizer.
        loss_func: measure of loss function used in training.
        train_dl: training data
        x_valid, y_valid: validation data to implement an early stopping condition.
    """
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = loss_func(model(x_valid), y_valid)

        print(epoch, valid_loss, accuracy(model(x_valid), y_valid))

    return model