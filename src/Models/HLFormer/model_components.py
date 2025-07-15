import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.onmt.hyper_nets import LorentzMultiHeadedAttention
import ipdb

def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)
        else:
            return denominator - nominator


class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):

        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)

        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class HLFormerBlock(nn.Module):
    def __init__(self, config,manifold):
        super(HLFormerBlock, self).__init__()

        self.num_block = int (config.attention_num//2)-1
        self.e_attns = nn.ModuleList()
        self.e_attns.append(EuclideanAttentionBlock(config))
        for i in range(1,self.num_block+1):
            wid = 2 ** i 
            self.e_attns.append(EuclideanAttentionBlock(config, wid=wid))

        self.h_attns = nn.ModuleList()
        self.h_attns.append(LorentzAttentionBlock(config,manifold=manifold))

        for i in range(1,self.num_block+1):
            wid = 2 ** i 
            self.h_attns.append(LorentzAttentionBlock(config,manifold=manifold, wid=wid))


        self.ca = CrossAttention(config)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer2 = nn.Linear(config.hidden_size, config.frame_len)

        self.sft_factor = config.sft_factor

    def forward(self, input_tensor, attention_mask=None, weight_token=None):

        outputs = []
        for i in range(len(self.e_attns)):
            o = self.e_attns[i](input_tensor, attention_mask).unsqueeze(-1)
            outputs.append(o) 
        for i in range(len(self.h_attns)):  
            o = self.h_attns[i](input_tensor, attention_mask).unsqueeze(-1)
            outputs.append(o) 
        oo = torch.cat(outputs, dim=-1)


        if weight_token is None:
            mean_oo = torch.mean(oo, dim=-1)#.squeeze())
            weight_token =torch.mean(mean_oo,dim=1,keepdim=True)

        else:
            weight_token = weight_token.to(oo.device).type_as(oo).repeat(oo.shape[0], 1, 1)
        weight = []
        for i in range(oo.shape[-1]):
            temp_token = self.ca(weight_token, oo[..., i], attention_mask)
            weight.append(temp_token)

        weight = torch.cat(weight, dim=1)    
        weight = self.layer1(weight)
        weight = self.dropout(F.relu(weight))
        weight = self.layer2(weight)
        
        weight = F.softmax(weight.permute(0, 2, 1) / self.sft_factor, dim=-1)
        out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)

        return out


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x

class EuclideanAttentionBlock(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanAttentionBlock, self).__init__()
        self.self = EuclideanGaussianAttention(config, wid=wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = EuclideanGaussianAttention(config)
        self.output = FeedForward(config.hidden_size, int(1*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, query, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(query, input_tensor, input_tensor, attention_mask)

        self_output = self.dropout1(self_output)
        query = self.norm1(query + self_output)
        tmp = self.output(query)
        tmp = self.dropout2(tmp)
        query = self.norm2(query + tmp)
        return query


class EuclideanGaussianAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanGaussianAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

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

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores_ori
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attention_scores.shape[-1], self.wid)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores_ori * gmm_mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
    
class LorentzSelfAttention(nn.Module):
    def __init__(self, config,manifold,wid=None):
        super().__init__()
        self.config = config
        self.input_proj = LinearLayer(config.hidden_size,config.lorentz_dim,dropout = config.drop)
        self.scale_alpha = nn.Parameter(torch.tensor(config.lorentz_dim**-0.5).log())
        self.lorentz_multiattention = LorentzMultiHeadedAttention(dropout = config.drop,model_dim=config.lorentz_dim+1,head_count=config.num_attention_heads,manifold=manifold,wid=wid)
        self.manifold = manifold
        self.output_proj = LinearLayer(config.lorentz_dim,config.hidden_size,dropout = config.drop)

    def forward(self,x,attention_mask=None):
        x = self.input_proj(x)

        self.scale_alpha.data = torch.clamp(self.scale_alpha.data,max=0.)
        x = x*self.scale_alpha.exp()

        x = F.pad(x, pad=(1,0), value=0)
        x = self.manifold.expmap0(x)
        x = self.lorentz_multiattention(x,x,x,attention_mask)
        x = self.manifold.logmap0(x)[:,:,1:]
        x = x/self.scale_alpha.exp()
        output = self.output_proj(x)
        return output

class LorentzAttentionBlock(nn.Module):
    def __init__(self,config,manifold,wid=None):
        super().__init__()
        self.self = LorentzSelfAttention(config, manifold,wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor,attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor
