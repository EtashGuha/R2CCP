import torch
from R2CCP.data import get_loaders, get_input_and_range, get_train_cal_data
from R2CCP.argparser import get_parser_args
import pytorch_lightning as pl
from R2CCP.models.model import GenModule
import os
from R2CCP.cp import get_cp_lists, calc_coverages_and_lengths
from R2CCP.models.callbacks import get_callbacks
import random
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class R2CCP():
    def __init__(self, options):
        """
        Initialize a custom argument parser based on the provided keyword arguments (kwargs).

        Args:
            **kwargs: Keyword arguments representing the command-line arguments.
                - early_stopping (bool): Enable early stopping (default: False).
                - save_path (str): Where to save the model (default: None)
                - alpha (float): Alpha parameter (default: 0.1).
                - annealing (bool): Enable annealing (default: False).
                - annealing_epochs (int): Annealing epochs (default: 500).
                - weight_decay (float): Weight decay (default: 0.0).
                - ffn_activation (str): Activation function for the FFN. Choices: ['relu', 'sigmoid'] (default: 'relu').
                - ffn_hidden_dim (int): Number of nodes in FFN hidden layers (default: 256).
                - transformer_hidden_dim (int): Number of nodes in transformer hidden layers (default: 256).
                - ffn_num_layers (int): Number of layers in FFN (default: 3).
                - transformer_num_layers (int): Number of layers in transformer (default: 3).
                - lq_norm_val (float): Lq norm value (default: .5).
                - transformer_num_heads (int): Number of heads in transformer (default: 8).
                - dropout_prob (float): Dropout probability (default: 0).
                - lr_scheduler (str): Learning rate scheduler. Choices: ['cosine', 'cosine_warmup', 'linear', 'step', 'absent'] (default: 'cosine').
                - batch_size (int): Batch size (default: 32).
                - bias (bool): Use bias (default: True).
                - max_epochs (int): Maximum epochs (default: 1000).
                - seed (int): Random seed (default: 0).
                - cal_size (float): Calibration size (default: 0.2).
                - optimizer (str): Optimization algorithm (default: 'adam', options =['adam', 'adamw', 'sgd']).
                - range_size (int): Range size (default: 50).
                - lr (float): Learning rate (default: 1e-3).
                - constraint_weights (list of float): List of constraint weights.
                - num_workers (int): Number of workers (default: 4).
        """
        parser = get_parser_args()
        args = parser.parse_args([])
        for key, value in options.items():
            if key not in args:
                raise ValueError(f"Argument '{key}' is not defined in the parser.")
            arg_type = type(args.__getattribute__(key))
            if not isinstance(value, arg_type):
                raise TypeError(f"Argument '{key}' should have type {arg_type.__name__}, but got {type(value).__name__}.")
            setattr(args, key, value)
        self._args = args

    def fit(self, X, y):
        seed_everything(self._args.seed)
        self.train_X = X
        self.train_y = y
        input_size, range_vals = get_input_and_range(X, y, self._args)
        self.range_vals = range_vals

        model = GenModule(self._args, input_size, range_vals)

        total_path = self._args.model_path
        if os.path.exists(total_path):
            model.load_state_dict(torch.load(total_path))
        else:
            train_loader, cal_loader = get_loaders(X, y, self._args)
            callbacks = get_callbacks(self._args)
            if torch.cuda.is_available():
                trainer = pl.Trainer(max_epochs=self._args.max_epochs, accelerator="gpu", devices=[0], callbacks=callbacks)
            else:
                trainer = pl.Trainer(max_epochs=self._args.max_epochs, accelerator="cpu", callbacks=callbacks)

            trainer.fit(model, train_loader, cal_loader)
            torch.save(model.state_dict(), total_path)
        model.eval()
        self.model = model
    
    def get_intervals(self, X):
        X_train, y_train, X_cal, y_cal = get_train_cal_data(self.train_X, self.train_y, self._args)
        intervals = get_cp_lists(X, self._args, self.range_vals, X_cal, y_cal, self.model)
        return intervals
    
    def get_coverage_length(self, X, y):
        X_train, y_train, X_cal, y_cal = get_train_cal_data(self.train_X, self.train_y, self._args)
        intervals = get_cp_lists(X, self._args, self.range_vals, X_cal, y_cal, self.model)
        return calc_coverages_and_lengths(intervals, y, self.range_vals)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    X_train = np.random.rand(10, 1)
    Y_train = 2 * X_train + 1 + 0.1 * np.random.randn(10, 1)
    X_test = np.random.rand(10, 1)
    Y_test = 2 * X_test + 1 + 0.1 * np.random.randn(10, 1)

    model = R2CCP({'model_path':'model_paths/idk.pth', 'max_epochs':5})
    model.fit(X_train, Y_train)

    intervals = model.get_intervals(X_test)
    coverage, length = model.get_coverage_length(X_test, Y_test)
    print(f"Coverage: {np.mean(coverage)}, Length: {np.mean(length)}")
