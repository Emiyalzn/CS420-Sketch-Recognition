from .basemodel import BaseModel
from neuralline.rasterize import RasterIntensityFunc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trans_utils import positional_encoding, scaled_dot_product_attention, create_masks

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(input_size, d_model)
        self.wk = nn.Linear(input_size, d_model)
        self.wv = nn.Linear(input_size, d_model)

        self.ff = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """ split the last dimension into (num_heads, depth)
        Transpose the results such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return torch.permute(x, [0,2,1,3])

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = torch.permute(scaled_attention, [0,2,1,3]) # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = torch.reshape(scaled_attention, [batch_size, -1, self.d_model]) # (batch_size, seq_len_q, d_model)

        output = self.ff(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.rate = rate

        self.mha = MultiHeadAttention(d_model, d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = F.dropout(attn_output, self.rate, training)
        out1 = self.layernorm1(x+attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = F.dropout(ffn_output, self.rate, training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TransEncoder(nn.Module):
    def __init__(self, input_size, num_layers, d_model,
                 num_heads, dff,
                 maximum_position_encoding=1000,
                 out_channels=8,
                 rate=0.1):
        super().__init__()

        self.rate = rate
        self.d_model = d_model
        self.num_layers = num_layers
        self.out_channels = out_channels

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.embedding = nn.Linear(input_size, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.attend_fc = nn.Linear(d_model, out_channels)

    def forward(self, x, training, mask):
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, ...].to(x.device)

        x = F.dropout(x, self.rate, training)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, mask)

        # TODO: May need to mask out the last padding intensities?
        intensities = torch.sigmoid(self.attend_fc(x))

        return intensities # (batch_size, input_seq_len, out_channels)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.rate = rate

        self.mha1 = MultiHeadAttention(d_model, d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = F.dropout(attn1, self.rate, training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = F.dropout(attn2, self.rate, training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = F.dropout(ffn_output, self.rate, training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class TransDecoder(nn.Module):
    def __init__(self, input_size, num_layers, d_model,
                 num_heads, dff,
                 maximum_position_encoding=1000,
                 rate=0.1):
        super(TransDecoder, self).__init__()

        self.rate = rate
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.embedding = nn.Linear(input_size, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x) # (batch_size, target_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, ...]

        x = F.dropout(x, self.rate, training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class DenseExpander(nn.Module):
    """
    Expand tensor using Dense conv
    input: (batch_size, feat_dim_in)
    output: (batch_size, seq_len, feat_dim_out)
    """
    def __init__(self, input_size, seq_len, feat_dim_out=0):
        super(DenseExpander, self).__init__()
        self.seq_len = seq_len
        self.feat_dim_out = feat_dim_out

        if self.feat_dim_out:
            self.project_layer = nn.Linear(input_size, self.feat_dim_out)
        self.expand_layer = nn.Linear(1, self.seq_len)

    def forward(self, x):
        if self.feat_dim_out:
            x = self.project_layer(x) # (batch_size, feat_dim_out)
            x = F.relu(x)
        x = torch.unsqueeze(x, 2) # (batch_size, feat_dim_out, 1)
        x = self.expand_layer(x) # (batch_size, feat_dim_out, seq_len)
        x = torch.permute(x, [0,2,1])
        return x

class Trans2CNN(BaseModel):
    def __init__(self,
                 cnn_fn, img_size, thickness,
                 num_categories, max_seq_len=226,
                 trans_input_size=5, num_layers=4,
                 d_model=128, dff=512, num_heads=8,
                 dropout=0.1, intensity_channels=8,
                 do_reconstruction=False, train_cnn=True,
                 device=None):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        self.intensity_channels = intensity_channels
        self.eps = 1e-4
        self.do_reconstruction = do_reconstruction
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.encoder = TransEncoder(trans_input_size, num_layers, d_model, num_heads, dff,
                                    out_channels=intensity_channels, rate=dropout)
        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=intensity_channels)

        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        if self.do_reconstruction:
            self.expand_layer = DenseExpander(num_fc_in_features, max_seq_len)
            self.decoder = TransDecoder(num_fc_in_features, num_layers, d_model, num_heads, dff,
                                        rate=dropout)

        nets.extend([self.encoder, self.cnn, self.fc])
        names.extend(['transencoder', 'conv', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points_offset, points, training):
        inp = tar = points_offset
        tar_inp = tar[:, :-1, ...]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        intensities = self.encoder(inp, training, enc_padding_mask)

        images = RasterIntensityFunc.apply(points, intensities, self.img_size, self.thickness, self.eps, self.device)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        cnnfeat = self.cnn(images)

        if self.do_reconstruction:
            expand_embedding = self.expand_layer(cnnfeat)
            dec_output, _ = self.decoder(tar_inp, expand_embedding, training, combined_mask, dec_padding_mask)
        else:
            dec_output = None

        logits = self.fc(cnnfeat)

        return logits, dec_output
