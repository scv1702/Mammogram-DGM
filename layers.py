import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm=None, activation='ReLU'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'Batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'Instance':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == None:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        # initialize activation
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'SELU':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'TanH':
            self.activation = nn.Tanh()
        elif activation == None:
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample=False, norm=None, shape=None, activation=None, bias=True):
        super(Conv2dBlock, self).__init__()
        # Initialize Upsample Layer
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = None
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # Initialize Normalization Layer
        if norm == 'Layer':
            self.norm = nn.LayerNorm(shape)
        elif norm == 'Instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'AdaIn':
            self.norm = AdaptiveInstanceNorm2d(out_channels)
        else:
            self.norm = None
        # Initialize Activation Layer
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm='AdaIn', activation='ReLU'),
            Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm='AdaIn', activation=None)
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResidualBlocks(nn.Module):
    def __init__(self, num_of_blocks, in_channels, out_channels):
        super(ResidualBlocks, self).__init__()
        self.model = []
        for i in range(num_of_blocks):
            self.model += [ResidualBlock(in_channels, out_channels)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)