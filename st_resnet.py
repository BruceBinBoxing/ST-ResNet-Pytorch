import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        #self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        #if self.has_bn:
        #    x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual # short cut

        return out

class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x

# Matrix-based fusion
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad = True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights # element-wise multiplication

        return x



class stresnet(nn.Module):
    def __init__(self, c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32),
                 t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
        '''
            C - Temporal Closeness
            P - Period
            T - Trend
            conf = (len_seq, nb_flow, map_height, map_width)
            external_dim
        '''

        super(stresnet, self).__init__()
        logger = logging.getLogger(__name__)
        logger.info('initializing net params and ops ...')

        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf

        self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]

        self.relu = torch.relu
        self.tanh = torch.tanh
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.qr_nums = len(self.quantiles)

        if self.c_conf is not None:
            self.c_way = self.make_one_way(in_channels = self.c_conf[0] * self.nb_flow)

        # Branch p
        if self.p_conf is not None:
            self.p_way = self.make_one_way(in_channels = self.p_conf[0] * self.nb_flow)

        # Branch t
        if self.t_conf is not None:
            self.t_way = self.make_one_way(in_channels = self.t_conf[0] * self.nb_flow)

        # Operations of external component
        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.external_dim, 10, bias = True)),
                ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, self.nb_flow * self.map_height * self.map_width, bias = True)),
                ('relu2', nn.ReLU()),
            ]))

    def make_one_way(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels = in_channels, out_channels = 64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter = 64, repetations = self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels = 64, out_channels = 2)),
            ('FusionLayer', TrainableEltwiseLayer(n = self.nb_flow, h = self.map_height, w = self.map_width))
        ]))

    def forward(self, input_c, input_p, input_t, input_ext):
        # Three-way Convolution
        main_output = 0
        if self.c_conf is not None:
            input_c = input_c.view(-1, self.c_conf[0]*2, self.map_height, self.map_width)
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.p_conf is not None:
            input_p = input_p.view(-1, self.p_conf[0]*2, self.map_height, self.map_width)
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.t_conf is not None:
            input_t = input_t.view(-1, self.t_conf[0]*2, self.map_height, self.map_width)
            out_t = self.t_way(input_t)
            main_output += out_t

        # parameter-matrix-based fusion
        #main_output = out_c + out_p + out_t

        # fusing with external component
        if self.external_dim != None and self.external_dim > 0:
            # external input
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.nb_flow, self.map_height, self.map_width)
            #main_output = torch.add(main_output, external_output)
            main_output += external_output

        else:
            print('external_dim:', external_dim)


        main_output = self.tanh(main_output)

        return main_output


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    model = stresnet((3, 2, 16, 8), (4, 2, 16, 8), (4, 2 , 16, 8), external_dim=8, nb_residual_unit=4)
    #print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params:', pytorch_total_params)

    model.to(device)
    #summary(model, [(3, 2, 16, 8), (4, 2, 16, 8), (4, 2, 16, 8), (8, )], batch_size=-1, device= 'cuda')
    #print(model)demo_net.py
