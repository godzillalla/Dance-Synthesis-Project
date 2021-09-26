''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(batch_size, seq_len, max_seq_len):
    one = torch.ones((batch_size, seq_len))
    zero = torch.zeros((batch_size, max_seq_len - seq_len))
    mask = torch.cat((one, zero), dim=-1)
    return mask.unsqueeze(-2)


def get_subsequent_mask(len_s, device):
    ''' For masking out the subsequent info. '''
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=device), diagonal=1)).bool()
    return subsequent_mask


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, device, batch_size=128, n_position=200, dropout=0.1):

        super().__init__()
        self.n_position = n_position
        self.batch_size = batch_size
        self.device = device

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position + 1, d_word_vec, padding_idx=0), freeze=True).to(device)
        self.position_encoding = self.position_enc(
            torch.stack([torch.arange(n_position) for i in range(batch_size)]).to(device))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_in, pad_mask, return_attns=False):

        enc_slf_attn_list = []

        if self.position_encoding.shape[0] != enc_in.shape[0]:
            self.position_encoding = self.position_enc(
                torch.stack([torch.arange(self.n_position) for i in range(enc_in.shape[0])]).to(self.device))

        # -- Forward
        enc_output = enc_in + self.position_encoding

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=pad_mask)
            if return_attns:
                enc_slf_attn_list = enc_slf_attn_list + [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, device, batch_size=128, n_position=200, dropout=0.1):

        super().__init__()
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position + 1, d_word_vec, padding_idx=0), freeze=True).to(device)
        self.position_encoding = self.position_enc(
            torch.stack([torch.arange(n_position) for i in range(batch_size)]).to(device))

        self.slf_attn_mask = get_subsequent_mask(n_position, device).expand((batch_size, n_position, n_position))

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.n_position = n_position
        self.device = device

    def forward(self, dec_in, enc_output, pad_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []
        if self.position_encoding.shape[0] != dec_in.shape[0]:
            self.slf_attn_mask = \
                get_subsequent_mask(self.n_position, self.device).expand(
                    (dec_in.shape[0], self.n_position, self.n_position))
            self.position_encoding = self.position_enc(
                torch.stack([torch.arange(self.n_position) for i in range(enc_output.shape[0])]).to(self.device))

        dec_output = dec_in + self.position_encoding

        # -- Forward
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=self.slf_attn_mask,
                dec_enc_attn_mask=pad_mask)
            dec_slf_attn_list = dec_slf_attn_list + [dec_slf_attn] if return_attns else []
            dec_enc_attn_list = dec_enc_attn_list + [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, device, d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, batch_size=128,
            dropout=0.1, n_position=200, out_dim=256):

        super().__init__()

        self.d_model = d_model
        self.n_position = n_position
        self.device = device

        self.encoder = Encoder(
            d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, device=device,
            batch_size=batch_size, n_position=n_position, dropout=dropout)

        self.decoder = Decoder(
            d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, device=device,
            batch_size=batch_size, n_position=1, dropout=dropout)

        self.converter = nn.Linear(d_model, out_dim, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq, trg_seq):

        pad_mask = get_pad_mask(src_seq.shape[0], src_seq.shape[1], self.n_position).to(self.device)
        enc_output = self.encoder(src_seq, pad_mask)
        dec_output = self.decoder(trg_seq[:, 3:4, :], enc_output[:, 3:4, :], pad_mask[..., 3:4])
        seq_logit = self.converter(dec_output)

        '''
        print("--------------------- transformer ------------------------")
        print("pad mask:", pad_mask.shape)
        print("encoder:", src_seq.shape, enc_output.shape)
        print("decoder:", trg_seq.shape, dec_output.shape)
        print("final:", seq_logit.shape)
        '''

        return seq_logit






