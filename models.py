import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from layers import Conv2dBlock, LinearBlock

# Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self, norm):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            Conv2dBlock(1, 64, 7, stride=2, padding=3, norm=norm, activation='ReLU'),
            Conv2dBlock(64, 64, 4, stride=2, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation=False),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation=False),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation=False),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, activation=False)
        )

    def forward(self, image):
        latent_vector = self.model(image)
        return latent_vector

class Decoder(nn.Module):
    def __init__(self, input_size, norm):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.feautre_shapes = [
            [512, 7, 7],
            [512, 7, 7],
            [512, 7, 7],
            [512, 7, 7],
            [512, 7, 7],
            [512, 7, 7],
            [512, 7, 7],
            [512, 7, 7],
            [512, 14, 14],
            [256, 28, 28],
            [128, 56, 56],
            [64, 112, 112]
        ]
        self.model = nn.Sequential(
            Conv2dBlock(self.input_size, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[0], activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[1], activation=False),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[2], activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[3], activation=False),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[4], activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[5], activation=False),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[6], activation='ReLU'),
            Conv2dBlock(512, 512, 3, stride=1, padding=1, norm=norm, shape=self.feautre_shapes[7], activation=False),
            Conv2dBlock(512, 512, 5, stride=1, padding=2, upsample=True, norm=norm, shape=self.feautre_shapes[8], activation='ReLU'),
            Conv2dBlock(512, 256, 5, stride=1, padding=2, upsample=True, norm=norm, shape=self.feautre_shapes[9], activation='ReLU'),
            Conv2dBlock(256, 128, 5, stride=1, padding=2, upsample=True, norm=norm, shape=self.feautre_shapes[10], activation='ReLU'),
            Conv2dBlock(128, 64, 5, stride=1, padding=2, upsample=True, norm=norm, shape=self.feautre_shapes[11], activation='ReLU'),
            Conv2dBlock(64, 1, 7, stride=1, padding=3, upsample=True, norm=False, activation='Tanh'),
        )
        
    def forward(self, latent_vector):
        synthesized_image = self.model(latent_vector)
        return synthesized_image

class J(nn.Module):
    def __init__(self):
        super(J, self).__init__()
        self.decoder = Decoder(1024, norm='AdaIn')
        self.mlp = MLP(int(np.prod((512, 7, 7))), self.get_num_adain_params(self.decoder), 256, 3, norm=None, activ='ReLU')
        
    def forward(self, c_z, c_s):
        adain_params = self.mlp(torch.flatten(c_s))
        self.assign_adain_params(adain_params, self.decoder)
        reconstructed_image = self.decoder(torch.cat((c_z, c_s), 1))
        return reconstructed_image

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

# Generator
class VAE(nn.Module):
    def __init__(self, input_size, encoder_norm=None, decoder_norm=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(norm=encoder_norm)
        self.decoder = Decoder(input_size, norm=decoder_norm)
        
    def forward(self, image):
        latent_vector = self.encoder(image)
        synthesized_img = self.decoder(latent_vector)
        return synthesized_img

# Basic block
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm=None, activ='ReLU'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm=None, activation='ReLU')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.expand(1, x.shape[0]))

# Discriminator
class FCDiscriminator(nn.Module):
    def __init__(self, config):
        super(FCDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(config.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        validity = self.model(image)
        return validity

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_dim, params):
        super(MultiScaleDiscriminator, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class LSGANDiscriminator(nn.Module):
    def __init__(self):
        super(LSGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dBlock(1, 64, kernel_size=5, stride=2, padding=2, activation='LeakyReLU', bias=False),
            Conv2dBlock(64, 128, kernel_size=3, stride=2, padding=1, activation='LeakyReLU', bias=False),
            Conv2dBlock(128, 256, kernel_size=3, stride=2, padding=1, activation='LeakyReLU', bias=False), 
            Conv2dBlock(256, 512, kernel_size=3, stride=2, padding=1, activation='LeakyReLU', bias=False),
            Conv2dBlock(512, 512, kernel_size=3, stride=2, padding=1, activation='LeakyReLU', bias=False), 
            Conv2dBlock(512, 512, kernel_size=3, stride=2, padding=1, activation='LeakyReLU', bias=False), 
            Conv2dBlock(512, 512, kernel_size=3, stride=2, padding=1, activation='LeakyReLU', bias=False),
            Conv2dBlock(512, 1, kernel_size=3, stride=2, padding=1, activation=None, bias=False), 
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1,1)