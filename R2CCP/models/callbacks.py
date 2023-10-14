import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from pytorch_lightning.callbacks import EarlyStopping

class NanStoppingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check for NaN in the model's parameters
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                trainer.should_stop = True
                print(f"Training stopped due to NaN values in parameter: {name}")
def get_callbacks(args):
    callbacks=[NanStoppingCallback()]
    
    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # Choose the metric to monitor for early stopping
            patience=100,          # Number of epochs with no improvement after which training will be stopped
            verbose=False,        # Print early stopping messages
            mode='min'           # 'min' for minimizing the monitored metric, 'max' for maximizing
        )
        callbacks.append(early_stopping_callback)
    return callbacks