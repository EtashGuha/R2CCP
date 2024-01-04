# Overview
This is a library for generating prediction sets for machine learning regression tasks. 
We do this by first converting regression to a classification problem (divide the output space into 50 bins) and then using CP techniques for
classification to obtain a conformal set.

## Get Started
Our example file (example.py) provides a simple example of how to use our R2CCP class for conformal prediction. At a high level, the basic steps are instantiating the model class, fitting against data, and analyzing the results. 
```
# Instiantiate the model
model = R2CCP({'model_path':'model_paths/model_save_destination.pth', 'max_epochs':5})
// model_path is where to save the trained pytorch model output

# Fit against the data
model.fit(X_train, y_train)

# Analyze the results
intervals = model.get_intervals(X_test)
coverage, length = model.get_coverage_length(X_test, Y_test)
print(f"Coverage: {np.mean(coverage)}, Length: {np.mean(length)}")
```

## R2CCP Parameters
The R2CCP class can be instantiated with a variety of different parameters. Here is an overview of all the available options.
- early_stopping (bool): Enable early stopping (default: False). Uses Pytorch Lightning EarlyStopping. Set custom configuration in R2CCP/models/callbacks.py
- save_path (str): Where to save the model (default: None)
- alpha (float): Alpha parameter (default: 0.1).
- annealing (bool): Enable annealing (default: False).
- annealing_epochs (int): Annealing epochs (default: 500).
- weight_decay (float): Weight decay (default: 0.0).
- ffn_activation (str): Activation function for the FFN. Choices: ['relu', 'sigmoid'] (default: 'relu').
- ffn_hidden_dim (int): Number of nodes in FFN hidden layers (default: 256).
- transformer_hidden_dim (int): Number of nodes in transformer hidden layers (default: 256). In development
- ffn_num_layers (int): Number of layers in FFN (default: 3).
- transformer_num_layers (int): Number of layers in transformer (default: 3). In development
- lq_norm_val (float): Lq norm value (default: .5).
- transformer_num_heads (int): Number of heads in transformer (default: 8). In development
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

### Note: Transformer integration is still in development