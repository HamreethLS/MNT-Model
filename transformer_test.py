import torch
from transformers_layers import MultiHeadAttention, FeedForward, PositionalEncoding, EncoderLayer, DecoderLayer, Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_multihead_attention_shapes():
    batch_size, seq_len, d_model, num_heads = 2, 5, 16, 4
    mha = MultiHeadAttention(d_model, num_heads).to(device)

    Q = torch.randn(batch_size, seq_len, d_model, device=device)
    K = torch.randn(batch_size, seq_len, d_model, device=device)
    V = torch.randn(batch_size, seq_len, d_model, device=device)

    output = mha(Q, K, V)  # (batch, seq_len, d_model)

    assert output.shape == (batch_size, seq_len, d_model)


def test_feedforward_shapes():
    batch_size, seq_len, d_model, d_ff = 2, 6, 32, 64
    ff = FeedForward(d_model, d_ff).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output = ff(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_positional_encoding_adds_signal():
    batch_size, seq_len, d_model = 2, 10, 32
    pe = PositionalEncoding(d_model, max_seq_length=50).to(device)

    x = torch.zeros(batch_size, seq_len, d_model, device=device)
    out = pe(x)

    # The output should not be all zeros (positional encoding added)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_encoder_layer_shapes():
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 7, 32, 4, 64
    enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.1).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    mask = torch.ones(batch_size, 1, 1, seq_len, device=device)

    output = enc_layer(x, mask)
    assert output.shape == (batch_size, seq_len, d_model)


def test_decoder_layer_shapes():
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 7, 32, 4, 64
    dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout=0.1).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    enc_output = torch.randn(batch_size, seq_len, d_model, device=device)
    src_mask = torch.ones(batch_size, 1, 1, seq_len, device=device)
    tgt_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=device)

    output = dec_layer(x, enc_output, src_mask, tgt_mask)
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_forward_shapes():
    batch_size, src_len, tgt_len = 2, 8, 6
    src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff = 100, 120, 32, 4, 2, 64

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, num_heads, num_layers,
        d_ff, max_seq_length=50, dropout=0.1
    ).to(device)

    src = torch.randint(0, src_vocab_size, (batch_size, src_len), device=device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len), device=device)

    output = model(src, tgt)
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size)


def test_generate_mask_shapes():
    batch_size, src_len, tgt_len = 2, 10, 7
    model = Transformer(50, 60, 16, 4, 2, 32, max_seq_length=20, dropout=0.1).to(device)

    src = torch.randint(0, 50, (batch_size, src_len), device=device)
    tgt = torch.randint(0, 60, (batch_size, tgt_len), device=device)

    src_mask, tgt_mask = model.generate_mask(src, tgt)

    assert src_mask.shape == (batch_size, 1, 1, src_len)
    assert tgt_mask.shape == (batch_size, 1, tgt_len, tgt_len)

test_decoder_layer_shapes()
test_encoder_layer_shapes()
test_feedforward_shapes()
test_multihead_attention_shapes()
test_positional_encoding_adds_signal()  
test_transformer_forward_shapes()
test_generate_mask_shapes()

print("All tests passed!")

