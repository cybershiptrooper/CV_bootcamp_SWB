import numpy as np

b = 32 # batch size
wps = 64 # words per sentence
dm = 128 # dimension of model
nh = 8 # number of heads

def set_params(b_, wps_, dm_, nh_):
    global b, wps, dm, nh
    b, wps, dm, nh = b_, wps_, dm_, nh_


def multihead_attention_soln(res_stream, Wq, Wv, Wk):
    dh = dm / nh
    qs = np.einsum('bwm, mo -> bwo', res_stream, Wq).reshape((b, nh, wps, -1))
    vs = np.einsum('bwm, mo -> bwo', res_stream, Wv).reshape((b, nh, wps, -1))
    ks = np.einsum('bwm, mo -> bwo', res_stream, Wk).reshape((b, nh, wps, -1))
    qk = np.einsum('...nd, ...od->...no', qs, ks) / np.sqrt(dh)
    mask = np.tril(np.ones_like(qk))
    exp_qk = np.exp(qk)
    exp_qk *= mask
    exp_qk /= np.sum(exp_qk, axis=-1, keepdims=True)
    attn = np.einsum('...ij, ...jk->...ik', exp_qk, vs).reshape((b, wps, -1))
    out = attn + res_stream
    return out