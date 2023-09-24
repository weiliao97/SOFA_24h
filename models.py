# from matplotlib.ticker import OldAutoLocator
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# The following implementation is from
# @article{BaiTCN2018,
# 	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
# 	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
# 	journal   = {arXiv:1803.01271},
# 	year      = {2018},
# }
# link : https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    """
    To make sure causal convolution
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConv(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256], kernel_size=2, dropout=0.2):
        super(TemporalConv, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Sequential(
        nn.Linear(num_channels[-1], 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.contiguous().transpose(1, 2)
        x = self.linear(x)
        return x

# fusion at V 
class TemporalConvStatic(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256], num_static=25, kernel_size=2, dropout=0.2, s_param=[256, 256, 256, 0.2], c_param =[256, 256, 0.2]):
        super(TemporalConvStatic, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        s_layers = []
        # s_params = [256, 256, 256, 0.2]
        for i in range(len(s_param) - 2):
            static_in = num_static if i == 0 else s_param[i-1]
            s_layers += [nn.Linear(static_in, s_param[i])]
            s_layers += [nn.ReLU()]
            s_layers += [nn.Dropout(s_param[-1])]
        s_layers += [nn.Linear(s_param[-3], s_param[-2])]
        self.static = nn.Sequential(*s_layers)

        self.network = nn.Sequential(*layers)
        
        c_layers = []
        # [256, 256, 0.2] 
        for i in range(len(c_param) - 1):
            composite_in = num_channels[-1]+s_param[-2] if i == 0 else c_param[i-1]
            c_layers += [nn.Linear(composite_in, c_param[i])]
            c_layers += [nn.ReLU()]
            c_layers += [nn.Dropout(c_param[-1])]

        c_layers += [nn.Linear(c_param[-2], 1)]
        self.composite = nn.Sequential(*c_layers)

    def forward(self, x, s):
        # (17, 200, 48) --> (17, 256, 48)
        x = self.network(x)
        # (17, 256, 48) --> (17, 48, 256)
        x = x.contiguous().transpose(1, 2)
        # (17, 25) --> (17, 256)
        s = self.static(s) 
        # (17, 256) --> (17, 48, 256)
        s = s.unsqueeze(1).repeat(1, x.size()[1], 1)
        # (17, 48, 256) + (17, 48, 256) --> (17, 48, 512)
        x = torch.cat((x, s), dim=-1)
        # (17, 48, 512) --> (17, 48, 1)
        x = self.composite(x)
        return x 

# class TemporalConvStaticRNN(nn.Module):
#     def __init__(self, num_inputs, num_channels, num_static, kernel_size=2, dropout=0.2, input_dim=128, hidden_dim=128, layer_dim=3, output_dim=1, dropout_prob=0.2416, idrop=0.2595):
#         super(TemporalConvStaticRNN, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout)]

#         self.static = nn.Sequential(
#         nn.Linear(num_static, 128),
#         nn.BatchNorm1d(128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(128, 128),
#         nn.BatchNorm1d(128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(128, 128)
#         )

#         self.composite = nn.Sequential(
#         nn.Linear(128+num_channels[-1], 128),
#         # nn.BatchNorm1d(24),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(128, 128)
#         )
#         # composite is reduced inorder to extract features, which is set to 128 right now 
#         self.network = nn.Sequential(*layers)

#         #RNN
#         self.input_dim = input_dim 
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.dropout_prob = dropout_prob
#         self.output_dim = output_dim
#         # self.nonlinearity = activation
#         self.idrop = idrop 

#         self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob)
#         self.lockdrop = LockedDropout()
#         # Fully connected layer
#         self.fc = nn.Linear(self.hidden_dim, self.output_dim)


#     def forward(self, x, s):
#         x = self.network(x)
#         x = x.contiguous().transpose(1, 2)
#         s = self.static(s)
#         s = s.unsqueeze(1).repeat(1, x.size()[1], 1)

#         x = torch.cat((x, s), dim=-1)
#         x = self.composite(x)

#         td_input = self.lockdrop(x, self.idrop)

        
#         c0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_().to(device)
#         h0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_().to(device)
#         out, (hn, cn) = self.rnn(td_input, (h0.detach(), c0.detach()))
        
#         # out = out[:, -1, :]
#         out = self.fc(out)
#         # x = x.transpose(1, 2)
#         return out

# Early fusion at I 
class TemporalConvStaticE(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256], num_static=25, kernel_size=2, dropout=0.2, c_param=[256, 256, 0.2]):
        super(TemporalConvStaticE, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        c_layers = []
        # [256, 256, 0.2] 
        for i in range(len(c_param) - 1):
            composite_in = num_channels[-1] if i == 0 else c_param[i-1]
            c_layers += [nn.Linear(composite_in, c_param[i])]
            c_layers += [nn.ReLU()]
            c_layers += [nn.Dropout(c_param[-1])]

        c_layers += [nn.Linear(c_param[-2], 1)]
        self.composite = nn.Sequential(*c_layers)

    def forward(self, x, s):
        # x (17, 200, 48)
        # s (17, 25) --> (17, 25, 48) 
        s = s.unsqueeze(-1).repeat(1, 1, x.size()[2])
        x = torch.cat((x, s), dim=1) 

        x = self.network(x)
        x = x.contiguous().transpose(1, 2)
        
        x = self.composite(x)
        return x 

# Late fusion at VI
class TemporalConvStaticL(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256], num_static=25, kernel_size=2, dropout=0.2, c_param=[256, 256, 0.2], sc_param=[256, 256, 256, 0.2]):
        super(TemporalConvStaticL, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # self.static = nn.Sequential(
        # nn.Linear(num_static, 128),
        # nn.BatchNorm1d(128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128),
        # nn.BatchNorm1d(128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128)
        # )

        # self.composite = nn.Sequential(
        # nn.Linear(num_channels[-1], 128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 1)
        # )

        # self.s_composite = nn.Sequential(
        # nn.Linear(128, 64),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(64, 32),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(32, 1)
        # )

        c_layers = []
        # [256, 256, 0.2] 
        for i in range(len(c_param) - 1):
            composite_in = num_channels[-1] if i == 0 else c_param[i-1]
            c_layers += [nn.Linear(composite_in, c_param[i])]
            c_layers += [nn.ReLU()]
            c_layers += [nn.Dropout(c_param[-1])]

        c_layers += [nn.Linear(c_param[-2], 1)]
        self.composite = nn.Sequential(*c_layers)

        sc_layers = []
        # [256, 256, 256, 0.2]
        for i in range(len(sc_param) - 2):
            s_composite_in = num_static if i == 0 else sc_param[i-1]
            sc_layers += [nn.Linear(s_composite_in, sc_param[i])]
            sc_layers += [nn.ReLU()]
            sc_layers += [nn.Dropout(sc_param[-1])]

        sc_layers += [nn.Linear(sc_param[-2], 1)]

        self.s_composite = nn.Sequential(*sc_layers)

    def forward(self, x, s):

        # (17, 200, 48) --> (17, 256, 48)
        x = self.network(x)
        x = x.contiguous().transpose(1, 2)
        # (17, 48, 256) --> (17, 48, 128)
        x = self.composite(x)
        
        # (17, 25) --> (17, 1)
        s = self.s_composite(s) 
        # (17, 1) --> (17, 48, 1)
        s = s.unsqueeze(1).repeat(1, x.size()[1], 1) 
        # (17, 48, 1) + (17, 48, 1) --> (17, 48, 1)
        x = torch.add(x, s) 
        
        return x 

class TemporalConvStaticA(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256], num_static=25, kernel_size=2, dropout=0.2, s_param =[256, 256, 256, 0.2], c_param =[256, 256, 0.2], sc_param =[256, 256, 256, 0.2]):
        super(TemporalConvStaticA, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # self.static = nn.Sequential(
        # nn.Linear(num_static, 128),
        # nn.BatchNorm1d(128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128),
        # nn.BatchNorm1d(128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128)
        # )

        # self.composite = nn.Sequential(
        # nn.Linear(num_channels[-1]+128, 128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 1)
        # )

        # self.s_composite = nn.Sequential(
        # nn.Linear(128, 64),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(64, 32),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(32, 1)
        # )

        s_layers = []
        for i in range(len(s_param) - 2):
            static_in = num_static if i == 0 else s_param[i-1]
            s_layers += [nn.Linear(static_in, s_param[i])]
            s_layers += [nn.ReLU()]
            s_layers += [nn.Dropout(s_param[-1])]
        s_layers += [nn.Linear(s_param[-3], s_param[-2])]

        self.static = nn.Sequential(*s_layers)

        
        c_layers = []
        # [256, 256, 0.2] 
        for i in range(len(c_param) - 1):
            composite_in = num_channels[-1]+s_param[-2] if i == 0 else c_param[i-1]
            c_layers += [nn.Linear(composite_in, c_param[i])]
            c_layers += [nn.ReLU()]
            c_layers += [nn.Dropout(c_param[-1])]

        c_layers += [nn.Linear(c_param[-2], 1)]
        self.composite = nn.Sequential(*c_layers)

        sc_layers = []
        # [256, 256, 256, 0.2]
        for i in range(len(sc_param) - 2):
            s_composite_in = num_static if i == 0 else sc_param[i-1]
            sc_layers += [nn.Linear(s_composite_in, sc_param[i])]
            sc_layers += [nn.ReLU()]
            sc_layers += [nn.Dropout(sc_param[-1])]

        sc_layers += [nn.Linear(sc_param[-2], 1)]

        self.s_composite = nn.Sequential(*sc_layers)

    def forward(self, x, s):

        # early 
        # (17, 25) --> (17, 25, 48)
        s_1= s.unsqueeze(-1).repeat(1, 1, x.size()[2])
        # (17, 25, 48) + (17, 200, 48) --> (17, 225, 48)
        x = torch.cat((x, s_1), dim=1) 
        # (17, 225, 48) --> (17, 256, 48)
        x = self.network(x) 
        # (17, 256, 48) --> (17, 48, 256)
        x = x.contiguous().transpose(1, 2) #(6, 24, 1024)

        # intermediate
        # (17, 25) -->  (17, 128)
        ss = self.static(s)  
        # (17, 128)  --> (17, 48, 128)
        s_2 = ss.unsqueeze(1).repeat(1, x.size()[1], 1) 
        # (17, 48, 256) + (17, 48, 128) --> (17, 48, 384)
        x = torch.cat((x, s_2), dim=-1) 
        # (17, 48, 384) --> (17, 48, 1)
        x = self.composite(x)

        # late 
        # (17, 25) --> (17, 1)
        s3 = self.s_composite(s) 
        # (17, 1) --> (17, 48, 1)
        s4 = s3.unsqueeze(1).repeat(1, x.size()[1], 1) 
        # (17, 48, 1) + (17, 48, 1) --> (17, 48, 1)
        x = torch.add(x, s4) 
        
        return x 

class TemporalConvStaticI(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256],  num_static=25, kernel_size=3, dropout=0.2, s_param=[256, 256, 256, 0.2], c_param=[256, 256, 0.2], sc_param=[256, 256, 256, 0.2]):
        super(TemporalConvStaticI, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            if i == 0:
                in_channels = num_inputs 
            else:
                in_channels = num_channels[i-1] + s_param[-2]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.TB1 = layers[0]
        self.TB2 = layers[1]
        self.TB3 = layers[2]
        self.TB4 = layers[3]

        
        s1_layers = []
        # [256, 256, 256, 0.2]
        for i in range(len(s_param) - 2):
            static_in = num_static if i == 0 else s_param[i-1]
            s1_layers += [nn.Linear(static_in, s_param[i])]
            s1_layers += [nn.ReLU()]
            s1_layers += [nn.Dropout(s_param[-1])]
        s1_layers += [nn.Linear(s_param[-3], s_param[-2])]
        self.static1 = nn.Sequential(*s1_layers)

        s2_layers = []
        for i in range(len(s_param) - 2):
            static_in = num_static if i == 0 else s_param[i-1]
            s2_layers += [nn.Linear(static_in, s_param[i])]
            s2_layers += [nn.ReLU()]
            s2_layers += [nn.Dropout(s_param[-1])]
        s2_layers += [nn.Linear(s_param[-3], s_param[-2])]

        self.static2 = nn.Sequential(*s2_layers)

        s3_layers = []
        for i in range(len(s_param) - 2):
            static_in = num_static if i == 0 else s_param[i-1]
            s3_layers += [nn.Linear(static_in, s_param[i])]
            s3_layers += [nn.ReLU()]
            s3_layers += [nn.Dropout(s_param[-1])]
        s3_layers += [nn.Linear(s_param[-3], s_param[-2])]

        self.static3 = nn.Sequential(*s3_layers)

            
        s_layers = []
        for i in range(len(s_param) - 2):
            static_in = num_static if i == 0 else s_param[i-1]
            s_layers += [nn.Linear(static_in, s_param[i])]
            s_layers += [nn.ReLU()]
            s_layers += [nn.Dropout(s_param[-1])]
        s_layers += [nn.Linear(s_param[-3], s_param[-2])]

        self.static = nn.Sequential(*s_layers)

        
        c_layers = []
        # [256, 256, 0.2] 
        for i in range(len(c_param) - 1):
            composite_in = num_channels[-1]+s_param[-2] if i == 0 else c_param[i-1]
            c_layers += [nn.Linear(composite_in, c_param[i])]
            c_layers += [nn.ReLU()]
            c_layers += [nn.Dropout(c_param[-1])]

        c_layers += [nn.Linear(c_param[-2], 1)]
        self.composite = nn.Sequential(*c_layers)

        sc_layers = []
        # [256, 256, 256, 0.2]
        for i in range(len(sc_param) - 2):
            s_composite_in = num_static if i == 0 else sc_param[i-1]
            sc_layers += [nn.Linear(s_composite_in, sc_param[i])]
            sc_layers += [nn.ReLU()]
            sc_layers += [nn.Dropout(sc_param[-1])]

        sc_layers += [nn.Linear(sc_param[-2], 1)]

        self.s_composite = nn.Sequential(*sc_layers)

    def forward(self, x, s):
        # Fusion at I
        # early  (17, 25) --> (17, 25, 48)
        s_1= s.unsqueeze(-1).repeat(1, 1, x.size()[2]) 
        # (17, 25, 48) + (17, 200, 48) --> (17, 225, 48)
        x = torch.cat((x, s_1), dim=1) 

        # start all level fusion 
        # (17, 200, 48) --> (17, 256, 48)
        x = self.TB1(x) 

        # Fusion at II 
        # (17, 25) --> (17, 256)
        ss1 = self.static1(s) 
        # (17, 256) --> (17, 256, 48)
        s1_r = ss1.unsqueeze(-1).repeat(1, 1, x.size()[-1]) 
        # (17, 256, 48) + (17, 256, 48) --> (17, 512, 48)
        x = torch.cat((x, s1_r), dim=1) 
        # (17, 512, 48) --> (17, 256, 48)
        x = self.TB2(x) 

        # Fusion at III
        # (17, 25) --> (17, 256)
        ss2 = self.static2(s) 
        # (17, 256) --> (17, 256, 48)
        s2_r = ss2.unsqueeze(-1).repeat(1, 1, x.size()[-1]) 
        # (17, 256, 48) + (17, 256, 48) --> (17, 512, 48)
        x = torch.cat((x, s2_r), dim=1) 
        # (17, 512, 48) --> (17, 256, 48)
        x = self.TB3(x) 

        # Fusion at IV
        # (17, 25) --> (17, 256)
        ss3 = self.static3(s) 
        # (17, 256) --> (17, 256, 48)
        s3_r = ss3.unsqueeze(-1).repeat(1, 1, x.size()[-1]) 
        # (17, 256, 48) + (17, 256, 48) --> (17, 512, 48)
        x = torch.cat((x, s3_r), dim=1)
        # (17, 512, 48) --> (17, 256, 48)
        x = self.TB4(x) 
        # (17, 256, 48) --> (17, 48, 256)
        x = x.contiguous().transpose(1, 2)

        # Fusion at V
        # (17, 25) --> (17, 256)
        ss = self.static(s)  
        # (17, 256) --> (17, 48, 256)
        s_2 = ss.unsqueeze(1).repeat(1, x.size()[1], 1) 
        # (17, 48, 256) + (17, 48, 256) --> (17, 48, 512)
        x = torch.cat((x, s_2), dim=-1) 
        # (17, 48, 512) --> (17, 48, 1)
        x = self.composite(x) 

        # Fusion at VI
        # (17, 25) --> (17, 1)
        ss = self.s_composite(s) 
        # (17, 1) --> (17, 48, 1)
        s3 = ss.unsqueeze(1).repeat(1, x.size()[1], 1) 
        # (17, 48, 1) + (17, 48, 1) --> (17, 48, 1)
        x = torch.add(x, s3) 
        return x 

# Transformer models
class PositionalEncoding(nn.Module):

    "Implement the PE function."
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Trans_encoder(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, d_hid, nlayers, out_dim, dropout):
        super(Trans_encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(feature_dim, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask, key_mask):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, 1]
        """
        src = src.contiguous().transpose(1, 2) #(6, 24, 182)
        src = self.encoder(src) # (6, 24, 256)
        src = self.pos_encoder(src) # (6, 24, 256)
        output = self.transformer_encoder(src, src_mask, key_mask) # (6, 24, 256)
        output = self.decoder(output) # (6, 24, 1)
        return output
    
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = ~torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        # mask = mask.float()
        # mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        # mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # tensor([[False,  True,  True,  True,  True],
        # [False, False,  True,  True,  True],
        # [False, False, False,  True,  True],
        # [False, False, False, False,  True],
        # [False, False, False, False, False]])

        return mask

# RNN 
class LockedDropout(nn.Module):
    '''
    Dropout that is consistent on the sequence dimension.
    '''
    def forward(self, x, dropout=0.5):
        if not self.training:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask/torch.max(mask)
        mask = mask.expand_as(x)
        return mask * x

class RecurrentModel(nn.Module):
    '''
    Recurrent model for time series data.
    '''
    def __init__(self, cell='RNN', input_dim=200, hidden_dim=10, layer_dim=3, output_dim=2, dropout_prob=0.2, idrop=0, activation='tanh'):
        super(RecurrentModel, self).__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_prob = dropout_prob
        self.output_dim = output_dim
        self.cell = cell
        self.nonlinearity = activation
        self.idrop = idrop 

        # Defining the number of layers and the nodes in each layer
        if isinstance(cell, str):
            if self.cell.upper() == 'RNN':
                self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob, nonlinearity=self.nonlinearity)
            elif self.cell.upper() == 'LSTM':
                self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob)
            elif self.cell.upper() == 'GRU':
                self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob)
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')

        # # encode time-incariant layer 
        self.lockdrop = LockedDropout()
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, td_input, td_lengths):
        # td should be (4, 40, 200)
        # td_inpus is (4, 200, 40), length is [32, 34, 38, 40]
        # h0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_()
        if self.idrop >0: 
            td_input = self.lockdrop(td_input, self.idrop)
        packed_td = pack_padded_sequence(td_input, td_lengths, batch_first=True, enforce_sorted=False)

        if self.cell.upper() == 'LSTM':
            # c0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.rnn(packed_td)
        else:
            out, h0 = self.rnn(packed_td)
        # padded out shoudl be (4, 40, hidden_dim)
        padded_out, _ = pad_packed_sequence(out, batch_first=True)
        # padded_out = padded_out[:, -1, :]
        # final output is (4, 40, output_classes)
        padded_out = self.fc(padded_out)
        return padded_out
