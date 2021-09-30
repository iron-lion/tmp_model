import torch.nn as nn
import torch

class GEPEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, channel_len, embedding_first, embedding_second, embedding_third):
        super(GEPEncoder, self).__init__()
        self.channel_len = channel_len
        self.layer1 = nn.Sequential(
                        nn.Linear(in_features= channel_len, out_features = embedding_first),
                        nn.BatchNorm1d(num_features = embedding_first),
                        nn.ReLU())                        
        self.layer2 = nn.Sequential(
                        nn.Linear(in_features = embedding_first, out_features = embedding_second),
                        nn.BatchNorm1d(num_features = embedding_second),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Linear(in_features = embedding_second, out_features = embedding_third),
                        nn.BatchNorm1d(num_features = embedding_third),
                        nn.ReLU())
    def forward(self,x):
        out = x.reshape(-1, self.channel_len)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = out.view(out.size(0),-1)
        return out # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, embedding_third, relation_first, relation_second):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Linear(in_features = embedding_third * 2, out_features = relation_first),
                        nn.BatchNorm1d(num_features = relation_first),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Linear(in_features = relation_first, out_features = relation_second),
                        nn.BatchNorm1d(num_features = relation_second),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Linear(in_features = relation_second, out_features = 1),
                        nn.Sigmoid())
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.5)
        m.bias.data = torch.ones(m.bias.data.size())

