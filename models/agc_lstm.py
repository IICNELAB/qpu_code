import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
from models.graph import Graph

grandpa = [20, 20, 20, 20, 20, 20, 4, 5, 20, 20, 8, 9, 1, 0, 12, 13, 1, 0, 16, 17, 20, 7, 6, 10]
class AGC_LSTM(nn.Module):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(AGC_LSTM, self).__init__()
        feat_dim = 256
        hidden_size = 512
        self.LAMBDA = 0.01
        self.BETA = 0.001
        self.graph_layout = config['dataset']
        self.graph_strategy = 'spatial' if 'graph_strategy' not in config.keys() else config['graph_strategy']
        if self.graph_layout == 'ntu':
            self.parent = [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 20, 7, 11, 11] # ntu
        else:
            self.parent = [0, 0, 0, 0, 0, 0, 1, 6, 7, 2, 9, 10, 3, 12, 13, 4, 15, 16, 5, 18, 19] #fpha
        self.graph = Graph(layout=self.graph_layout, strategy=self.graph_strategy)
        self.drop_rate = 0.5
        device = config['device_ids'][0]
        ADJ = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False, device=device)
        self.register_buffer('ADJ', ADJ)
        self.pa = config['pa'] if 'pa' in config else 0
        self.fc = nn.Linear(in_channels*(1+self.pa), feat_dim)
        self.lstm = nn.LSTM(feat_dim * 2, hidden_size)
        self.tap1 = TAP(2, 2)
        self.agc_lstm1= AGC_LSTM_layer(hidden_size, hidden_size, self.drop_rate, ADJ)
        self.tap2 = TAP(2, 2)
        self.agc_lstm2= AGC_LSTM_layer(hidden_size, hidden_size, self.drop_rate, ADJ)
        self.tap3 = TAP(3, 2)
        self.agc_lstm3= AGC_LSTM_layer(hidden_size, hidden_size, self.drop_rate, ADJ)
        self.classifier = nn.Linear(hidden_size, num_cls)

    def forward(self, x):
        """
        x: (batch, in_  channels, num_frames, num_joints, num_body)
           (B,C,T,V,M)
        """
        # FC
        batch_size, _, num_frames, num_joints, num_body = x.shape
        x = x.permute(2, 0, 4, 3, 1)  # (T,B,M,V,C)

        if self.pa>0: 
            x_p = x[...,self.parent,:].reshape(num_frames, batch_size * num_joints * num_body, -1)
            x_c = x.reshape(num_frames, batch_size * num_joints * num_body, -1)
            if self.pa==2: 
                x_pp = x[...,grandpa,:].reshape(num_frames, batch_size * num_joints * num_body, -1)
                x = torch.cat([x_c,x_p,x_pp],-1)
            else:
                x = torch.cat([x_c,x_p],-1)
        else:
            x=x.reshape(num_frames, batch_size * num_joints * num_body, -1)

        x = self.fc(x)  # (num_frames, batch*num_joints*num_body, feat_dim)(T,B*V*M,F)
        # Feature augmentation
        x_previous = torch.cat([x[:1,...],x[:-1,...]],0)
        x = torch.cat([x,x-x_previous],-1) # (num_frames, batch*num_joints*num_body, hidden_size)(T,B*V*M,H)
        # LSTM
        x, _ = self.lstm(x)  # 
        x = x.reshape(num_frames, batch_size, num_body, num_joints, -1) # (T,B,M,V,H)

        # AGC-LSTM 1
        self.attn_weights = []
        h0 = x.new_zeros(batch_size, num_body, num_joints, self.agc_lstm1.hidden_size)
        c0 = x.new_zeros(batch_size, num_body, num_joints, self.agc_lstm1.hidden_size)
        x = self.tap1(x)
        #    x:      (T,B,M,V,H)
        #    (h, c):   (B,M,V,H)
        #    attn_w:   (T,M,V,1)
        x, _, attn_w = self.agc_lstm1(x, (h0, c0)) 
        self.attn_weights.append(attn_w)

        # AGC-LSTM 2
        x = self.tap2(x)
        x, _, attn_w = self.agc_lstm2(x, (h0, c0))
        self.attn_weights.append(attn_w)
        # AGC-LSTM 3
        x = self.tap3(x)
        x, (h, _), attn_w = self.agc_lstm3(x, (h0, c0))
        self.attn_weights.append(attn_w)
        f_glob = torch.sum(x, dim=-2).mean(dim=-2)  # (num_frames, batch, feat_dim)
        f_loc = torch.sum(self.attn_weights[-1] * h, dim=-2).mean(dim=-2)

        # Classify
        self.o_glob = self.classifier(f_glob)  # (num_frames, batch, num_cls)
        self.o_loc = self.classifier(f_loc)
        out = torch.softmax(self.o_glob[-1],dim=-1) + torch.softmax(self.o_loc[-1],dim=-1)
        return out

    def temporal_avg_pooling(self, x, kernel_size, stride):
        """
        Args:
            x: (num_frames, batch, num_joints, channels)
        Return:
            out: (num_frames_pooled, batch, num_joints, channels)
        """
        num_frames, batch_size, num_joints, channels = x.shape
        x = x.permute(1, 2, 3, 0).reshape(batch_size * num_joints, channels, num_frames)
        x = F.avg_pool1d(x, kernel_size, stride)
        x = x.reshape(batch_size, num_joints, channels, -1).permute(3, 0, 1, 2)
        return x

    def get_loss(self, output, target):
        # We dont really use output as deep supervision is used
        loss_glob = self.seq_cross_entropy(self.o_glob, target)
        loss_loc = self.seq_cross_entropy(self.o_loc, target)
        loss_attn_eq = [torch.sum((1 - x.mean(dim=0))**2) for x in self.attn_weights]
        loss_attn_eq = torch.sum(torch.Tensor(loss_attn_eq))
        loss_attn_num = [torch.mean(x.sum(dim=1)**2) for x in self.attn_weights]
        loss_attn_num = torch.sum(torch.Tensor(loss_attn_num))
        loss = loss_glob + loss_loc + self.LAMBDA * loss_attn_eq + self.BETA * loss_attn_num
        return loss

    @staticmethod
    def seq_cross_entropy(input, target):
        """
        Args:
            input: (seq_len, batch_size, num_cls)
            target: (batch_size)
        Return:
            out: (1)
        """
        seq_len, batch_size, _ = input.shape
        target = target.unsqueeze(0).expand(seq_len, -1).reshape(-1)
        input = input.reshape(seq_len * batch_size, -1)
        out = F.cross_entropy(input, target, reduction='none')
        out = out.reshape(seq_len, batch_size).sum(dim=0).mean()
        return out


class AGC_LSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, A):
        """
        Args:
            A: (num_groups, num_nodes, num_nodes)  normalized adj matrix
        """
        super(AGC_LSTM_layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.A = A
        # Input gate
        self.wxi = GConv(input_size,  hidden_size, A, 0, bias=True)
        self.whi = GConv(hidden_size, hidden_size, A, 0, bias=False)
        # Forget gate
        self.wxf = GConv(input_size,  hidden_size, A, 0, bias=True)
        self.whf = GConv(hidden_size, hidden_size, A, 0, bias=False)
        self.bf = Parameter(torch.Tensor(hidden_size))
        # Cell 
        self.wxc = GConv(input_size,  hidden_size, A, 0, bias=True)
        self.whc = GConv(hidden_size, hidden_size, A, 0, bias=False)
        # Output gate
        self.wxo = GConv(input_size,  hidden_size, A, 0, bias=True)
        self.who = GConv(hidden_size, hidden_size, A, 0, bias=False)
        # Attention
        self.attn_w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_wh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_us = nn.Linear(hidden_size, 1, bias=True)
        if self.dropout > 0:
            self.drop=nn.Dropout(self.dropout)

    def forward(self, x, hidden):
        """
        Args:
            x: (seq_len, batch, num_nodes, input_size)
            hidden: (h, c): (batch, num_nodes, hidden_size)
        Return:
            output: (seq_len, batch, hidden_size)
            (h, c): (batch, num_nodes, hidden_size)
            attn_weights: (seq_len, num_nodes, 1)
        """
        seq_len = x.shape[0]
        h = hidden[0]
        output = []
        attn_weights = []
        c = hidden[1]
        x = self.drop(x)
        # forward
        xi = self.wxi(x)
        xf = self.wxf(x)
        xo = self.wxo(x)
        xc = self.wxc(x)
        for k in range(seq_len):
            i = torch.sigmoid(xi[k] + self.whi(h))
            f = torch.sigmoid(xf[k] + self.whf(h))
            o = torch.sigmoid(xo[k] + self.who(h))
            u = torch.tanh(xc[k] + self.whc(h))
            c = f * c + i * u
            h = o * torch.tanh(c)
            attn_wt = self.compute_attn_weights(h)
            attn_weights.append(attn_wt)
            output.append(attn_wt * h + h)
        attn_weights = torch.stack(attn_weights, dim=0)
        output = torch.stack(output, dim=0)
        return output, (h, c), attn_weights

    def compute_attn_weights(self, h):
        q = torch.relu(torch.sum(self.attn_w(h), dim=-2, keepdim=True))
        h = torch.tanh(self.attn_wh(h) + self.attn_wq(q))
        attn_weights = torch.sigmoid(self.attn_us(h))
        return attn_weights

    def __repr__(self):
        return self.__class__.__name__ + '(' + \
                'input_size=' + str(self.input_size) + \
                ', hidden_size=' + str(self.hidden_size) + \
                ', dropout=' + str(self.dropout) + \
                ', A=' + str(self.A.shape) + ')'  


class TAP(nn.Module):
    """
    Time Average Pooling
    Args:
        x: (num_frames, batch, num_joints, channels)
    Return:
        out: (num_frames_pooled, batch, num_joints, channels)
    """
    def __init__(self, kernel_size, stride):
        super(TAP, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        num_frames, batch_size, num_body, num_joints, channels = x.shape
        x = x.permute(1, 2, 3, 4, 0).reshape(batch_size * num_joints * num_body, channels, num_frames)
        x = F.avg_pool1d(x, self.kernel_size, self.stride)
        x = x.reshape(batch_size, num_body, num_joints, channels, -1).permute(4, 0, 1, 2, 3)    
        return x
    def __repr__(self):
        return self.__class__.__name__ + '(' + \
                'kernel_size=' + str(self.kernel_size) + \
                'stride=' + str(self.stride) + ')'


class GConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, dropout, bias=True):
        """
        Args:
            A: (num_groups, num_nodes, num_nodes), normalized adjacent matrix.
        """
        super(GConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.num_groups = A.shape[0]
        self.dropout = dropout
        self.fc = nn.Linear(in_channels, out_channels * self.num_groups, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if self.dropout>0:
            self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x: (..., num_nodes, in_channels)
        Retrun:
            out: (..., num_nodes, out_channels)
        """
        x = self.fc(x)  # (..., num_nodes, num_groups * out_channels)
        x = x.reshape(list(x.shape[:-1]) + [self.num_groups, -1])  # (..., num_nodes, num_groups, out_channels)
        x = x.transpose(-3, -2)  # (..., num_groups, num_nodes, out_channels)
        x = torch.matmul(self.A, x)
        x = torch.sum(x, dim=-3)  # (..., num_nodes, out_channels)
        if self.bias is not None:
            x = x + self.bias
        if self.dropout > 0:
            x = self.drop(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + \
                'in_channels=' + str(self.in_channels) + \
                ', out_channels=' + str(self.out_channels) + \
                ', A=' + str(self.A.shape) + \
                ', bias=' + str(self.bias) + ')'  

if __name__ == "__main__":
    net = AGC_LSTM(5,25,48,10,None).to('cuda:0')
    x = torch.ones(2,5,48,25).to('cuda:0')
    print(net(x).shape)

