import torch
import pytorch_lightning as pl
from torch import optim
from R2CCP.models.transformer import Transformer
from R2CCP.models.mlp import MLPModel
import torchvision.models as tnmodels
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR

# Define a LightningModule
class GenModule(pl.LightningModule):
    def __init__(self, args, input_size, range_vals):
        super(GenModule, self).__init__()

        if args.model == "mlp":
            model_class = MLPModel
            self.model = model_class(args, input_size, len(range_vals))
        elif args.model == "transformer":
            model_class = Transformer
            self.model = model_class(args, input_size, len(range_vals))
        elif args.model == "resnet":
            self.model = tnmodels.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features,len(range_vals))

        self.loss_type = args.loss_type
        
        self.smax = torch.nn.Softmax(dim=1)
        self.register_buffer("range_vals",range_vals)
        self.annealing = args.annealing
        if self.annealing:
            self.initial_temperature = 1
            self.annealing_epochs = args.annealing_epochs
        if self.loss_type == "cross_entropy_quantile":
            self.register_buffer("alpha", torch.tensor(args.alpha))

        self.q = args.lq_norm_val
        self.arguments = args
        self.K = args.num_moments
        self.register_buffer("entropy_weight", torch.tensor(args.entropy_weight))
        self.register_buffer("loss_weight",torch.tensor(args.loss_weight))
            
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def compute_loss(self, batch):
        x, y = batch
        pre_probs = self(x)
        probs = self.smax(pre_probs)

        if self.annealing:
            self.entropy_weight = max(0, (self.initial_temperature * (1 - self.current_epoch / self.annealing_epochs)))
        log_vals = torch.log(probs)
        log_vals[probs == 0] = 0
        neg_entropy = torch.sum(torch.sum(probs * log_vals, dim=1))
        all_losses = [neg_entropy * self.entropy_weight]

        all_losses.append(self.loss_weight * torch.sum(probs * torch.pow(torch.abs(self.range_vals.view(1, -1).expand(len(y), -1) - y.view(-1, 1)), self.q)) )
  
        
        loss = torch.sum(torch.stack(all_losses))
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.compute_loss(batch)
        self.log('val_loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        if self.arguments.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.arguments.lr, weight_decay=self.arguments.weight_decay)
        elif self.arguments.optimizer =="adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.arguments.lr, weight_decay=self.arguments.weight_decay)
        elif self.arguments.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.arguments.lr, weight_decay=self.arguments.weight_decay)

        if self.arguments.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.arguments.max_epochs,
            )
        # elif self.arguments.lr_scheduler == "cosine_warmup":
        #     scheduler = LinearWarmupCosineAnnealingLR(
        #         optimizer,
        #         self.arguments.lr_warmup_epochs,
        #         self.arguments.max_epochs,
        #     )
        elif self.arguments.lr_scheduler == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self.arguments.lr_drop,
                total_iters=self.arguments.max_epochs,
            )
        elif self.arguments.lr_scheduler == "step":
            scheduler = MultiStepLR(
                optimizer,
                self.arguments.lr_steps,
                gamma=self.arguments.lr_drop,
            )
        elif self.arguments.lr_scheduler == "absent":
            return optimizer
        
        return [optimizer],[scheduler]
