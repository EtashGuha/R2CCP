import torch.nn as nn

# Define the MLP model
class OldMLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(OldMLPModel, self).__init__()
        self.fc1 = nn.Sequential(
          nn.Linear(input_size, 128),
          nn.LayerNorm(128),
          nn.Dropout(.25),
          nn.Linear(128, 1000),
          nn.LayerNorm(1000),
          nn.ReLU(),
          nn.Dropout(.25),
          nn.Linear(1000, 1000),
          nn.LayerNorm(1000),
          nn.ReLU(),
          nn.Dropout(.25),
          nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        
        return self.fc1(x)


class MLPModel(nn.Module):
    def __init__(self, args, input_size, num_classes):
        super(MLPModel, self).__init__()
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        activation = activations[args.ffn_activation]

        self.model = nn.Sequential()
        
        h = [args.ffn_hidden_dim] * (args.ffn_num_layers - 1)

        shapes = zip([input_size] + h, h + [num_classes])
        for i, (n, k) in enumerate(shapes):
            if i == args.ffn_num_layers - 1:
                self.model.append(nn.Linear(n, k, bias=args.bias))
            else:
                self.model.append(nn.Linear(n, k, bias=args.bias))
                self.model.append(activation())
                self.model.append(nn.Dropout(p=args.dropout_prob))

    def forward(self, x):
        return self.model(x)

