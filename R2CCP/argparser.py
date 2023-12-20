import argparse
from configargparse import Parser
import os

def parse_float_list(input_string):
    try:
        return [float(item) for item in input_string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float list format. Please provide a comma-separated list of numbers.")

def get_parser_args():
    parser = Parser()
    parser.add_argument('--model_path', type=str, default="model_paths", help='Name of the model')
    parser.add_argument('--model', type=str, default="mlp", help='Name of the model')

    parser.add_argument('--logdir', type=str, default=None, help='Name of the model')
    parser.add_argument('--early_stopping', action='store_true', help='Name of the model')
    parser.add_argument('--alpha', type=float, default=.1, help='Name of the model')
    parser.add_argument('--annealing', action="store_true", help='Name of the model')
    parser.add_argument('--annealing_epochs', type=int, default=500, help='Name of the model')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Name of the model')
    parser.add("--ffn_activation", choices=["relu", "sigmoid"], default="relu",
               help="The activation function to use in the FFN.")
    parser.add("--ffn_hidden_dim", default=256, type=int,
               help="The number of nodes in the FFN hidden layers.")
    parser.add("--transformer_hidden_dim", default=256, type=int,
               help="The number of nodes in the FFN hidden layers.")
    parser.add("--ffn_num_layers", default=3, type=int,
               help="The number of layers in the FFN.")
    parser.add("--transformer_num_layers", default=3, type=int,
               help="The number of layers in the FFN.")
    parser.add("--loss_type", default="moment", type=str,
               help="The number of layers in the FFN.")
    parser.add("--lq_norm_val", default=.5, type=float,
               help="The number of layers in the FFN.")
    parser.add("--transformer_num_heads", default=8, type=int,
               help="The number of layers in the FFN.")
    
    
    parser.add("--dropout_prob", default=0, type=float,
               help="The probability by which a node will be dropped out.")
    parser.add("--lr_scheduler", choices=["cosine", "cosine_warmup", "linear", "step", "absent"], default="cosine",
               help="The name of the LR scheduler to utilize.")
    parser.add_argument('--batch_size', type=int, default=32, help='Name of the model') 
    parser.add_argument('--bias', type=bool, default=True, help='Name of the model')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Name of the model')
    parser.add_argument('--seed', type=int, default=0, help='Name of the model')

    parser.add_argument('--test_size', type=float, default=.2, help='Name of the model')
    parser.add_argument('--optimizer', type=str, default="adamw", help='Name of the model')
    parser.add_argument('--range_size', type=int, default=50, help='Name of the model')
    parser.add_argument('--num_moments', type=int, help='Number of moments')
    parser.add_argument('--lr', type=float, default=1e-3, help='Number of moments')
    parser.add_argument('--devices', default=-1, help="Input can be an int or a list of ints")
    parser.add_argument("--entropy_weight", type=float, default=1)
    parser.add_argument("--loss_weight", type=float, default=5)
    parser.add_argument('--num_workers', type=float, default=0)
    

    return parser