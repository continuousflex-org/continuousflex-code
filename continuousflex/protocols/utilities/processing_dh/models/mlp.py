import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, output):
        super(mlp, self).__init__()

        hidden_dims = [512,1000, 512, 128]
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                                         nn.LayerNorm(hidden_dims[i+1]),
                                         nn.GELU(),
                                         
                                         ))
        self.mlp_ = nn.Sequential(*modules)

        self.output_layer = nn.Linear(hidden_dims[-1], output)

    def forward(self, x):
        mlp_ = self.mlp_(x)
        elu = nn.ELU()
        output_layer = elu(self.output_layer(mlp_))
        return output_layer
