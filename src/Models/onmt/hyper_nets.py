import torch.nn as nn
from Models.onmt.lorentz import Lorentz
import math
import torch

class LorentzLinear(nn.Module):
    """
        Perform the Lorentz linear transformation.

        args:
            in_features, out_features, bias: Same as nn.Linear
            dropout: Dropout rate in lorentz linear
            manifold: THe manifold that the linear layer operated in.
            nonlin: Non-linear function before the linear operation.
            merge: If set to True, it means that the input has the shape of [..., head_num, head_dim], and the output will has the shape of [..., head_num * head_dim]. The heads are merged.
            head_num: If `merge` is set to True, then head_num specifies the number of heads in input, otherwise it means that the output should be split into `head_num` heads, i.e., [..., head_num, head_dim]. If set to 0, then it is a normal lorentz linear layer.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=  None,
                 dropout=0.1,
                 manifold=Lorentz(),
                 nonlin=None,
                 head_num=0,
                 merge=False):
        super().__init__()
        self.nonlin = nonlin
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.head_num = head_num
        self.merge = merge
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)

    def forward(self, x, bias=None):
        if self.nonlin is not None:
            x = self.nonlin(x)
        if not self.merge:
            x = self.weight(self.dropout(x))
            if self.head_num > 0:
                x = x.view(x.shape[0], x.shape[1], self.head_num, -1)
        else:
            x = self.weight(
                self.dropout(x.flatten(-2)))
        # The following code has some inconsistency to Eq.7 in the paper. When calculating the time axis, 
        # we do not consider the bias in Eq.7, while we add the bias before determining time axis. 
        # It is a small bug here. However, both methods give mathematically correct results.
        # For reproducibility, we leave it unchanged here.
        if bias is not None:
            x = x + bias
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - self.manifold.k) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features // self.head_num if self.head_num > 0 else self.in_features
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)




class LorentzMultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """
    def __init__(self,
                 head_count,
                 model_dim,
                 manifold = Lorentz(),
                 dropout=0.1,
                 wid=None
                 ):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(LorentzMultiHeadedAttention, self).__init__()
        self.manifold = manifold
        self.head_count = head_count

        self.linear_keys = LorentzLinear(model_dim,
                                         head_count * self.dim_per_head,
                                         dropout=dropout,
                                         manifold=manifold,
                                         head_num=head_count)
        self.linear_values = LorentzLinear(model_dim,
                                           head_count * self.dim_per_head,
                                           dropout=dropout,
                                           manifold=manifold,
                                           head_num=head_count)
        self.linear_query = LorentzLinear(model_dim,
                                          head_count * self.dim_per_head,
                                          dropout=dropout,
                                          manifold=manifold,
                                          head_num=head_count)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(model_dim)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.wid = wid

    def generate_gauss_weight(self, props_len, width):

        center = torch.arange(props_len).cuda() / props_len
        width = width*torch.ones(props_len).cuda()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]
    def forward(self,
                key,
                value,
                query,
                mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
        """



        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, head_count, dim_per_head)
            return x.transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).reshape(batch_size,-1,head_count*dim_per_head)  #.contiguous()
            # .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        # print(f"key shape:{key.shape}")
        key = shape(key)
        value = shape(value)
        
        query = shape(query)

        attn = (2 +
                2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attn.shape[-1], self.wid)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attn = attn * gmm_mask
        if mask is not None:
            mask = (1-mask).unsqueeze(1)  # [B, 1, 1, T_values]
            mask = mask.to(torch.bool)
            attn = attn.masked_fill(mask, -1e18)
        attn = self.softmax(attn)

        context = self.manifold.mid_point(value, attn)

        context = unshape(context)


        return context

    def update_dropout(self, dropout):
        self.dropout.p = dropout