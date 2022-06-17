import numpy as np
import torch
import torch.nn.functional as F

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """calculate the attention weights"""
    matmul_qk = torch.matmul(q, k.transpose(2,3)) # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def create_padding_mask(seq):
    seq = torch.eq(seq[..., -1], 1).to(torch.float32) # look at last bit

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    # create an lower tri and invert it to get an upper trianguler with no diag
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = create_look_ahead_mask(tar.shape[1]).to(dec_target_padding_mask.device)
    combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def compute_reconstruction_loss(real, pred):
    pred_locations = pred[:, :, :2]
    pred_metadata = pred[:, :, 2:]
    tgt_locations = real[:, :, :2]
    tgt_metadata = real[:, :, 2:]

    location_loss = F.mse_loss(pred_locations, tgt_locations, reduction='none')
    metadata_loss = F.cross_entropy(pred_metadata, torch.argmax(tgt_metadata, dim=-1), reduction='none')

    loss_ = location_loss + metadata_loss
    mask = torch.logical_not(torch.eq(real[..., -1], 1)).to(loss_.device, dtype=loss_.dtype)

    loss_ *= mask
    return torch.mean(loss_)
