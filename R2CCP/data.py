from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split

def get_loaders(X, y, args):
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_cal = torch.tensor(X_cal, dtype=torch.float32)
    y_cal = torch.tensor(y_cal, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    cal_dataset = TensorDataset(X_cal, y_cal)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size)

    return train_loader, cal_loader

def get_input_and_range(X, y, args):
    train_loader, val_loader = get_loaders(X, y, args)
    X_train = train_loader.dataset.tensors[0]
    y_train = train_loader.dataset.tensors[1]

    input_size = X_train.shape[1]
    range_vals = torch.linspace(torch.min(y_train), torch.max(y_train), args.range_size)
    return input_size, range_vals


def get_train_cal_data(X, y, args):
    train_loader, cal_loader = get_loaders(X, y, args)
    X_train = train_loader.dataset.tensors[0]
    y_train = train_loader.dataset.tensors[1]
    X_cal = cal_loader.dataset.tensors[0]
    y_cal = cal_loader.dataset.tensors[1]
    return X_train, y_train, X_cal, y_cal