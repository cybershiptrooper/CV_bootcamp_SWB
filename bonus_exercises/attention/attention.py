import numpy as np
from soln.attention import test_multihead_attention_soln, set_params

b = 32 # batch size
wps = 64 # words per sentence
dm = 128 # dimension of model
nh = 8 # number of heads

'''
PS: implement the multihead attention function below
'''

def multihead_attention(res_stream, Wq, Wv, Wk, Wo) -> np.ndarray:
    '''
    Returns the multihead attention output
    Inputs:
        res_stream: (b, wps, dm) array
        Wq: list of nh (dm, dm // nh) np.ndarray
        Wv: list of nh (dm, dm // nh) np.ndarray
        Wk: list of nh (dm, dm // nh) np.ndarray
        Wo: (dm, dm) np.ndarray
    Outputs:
        q: (b, nh, wps, dm) np.ndarray
        k: (b, nh, wps, dm) np.ndarray
        v: (b, nh, wps, dm) np.ndarray
            The q/k/v values for each head
        qk: (b, nh, wps, wps) np.ndarray
            qk product for each head
        attn_scores: (b, nh, wps, wps) np.ndarray
            Attention scores for each head
        attn_out: (b, wps, dm) np.ndarray
            Output of MHA
        out: (b, wps, dm) np.ndarray
            State of res_stream after multihead attention
    '''
    # TODO: implement this function
    q, k, v = None, None, None
    qk = None
    attn_scores = None
    attn_out = None
    out = None

    return q, k, v, qk, attn_scores, attn_out, out

if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(precision=2)
    set_params(b, wps, dm, nh)
    assert dm % nh == 0, "dm must be divisible by nh"
    # list of parameters for each head d_model x d_head
    Wq = [np.random.uniform(0.0, 0.02, (dm, dm // nh)) for _ in range(nh)]
    Wv = [np.random.uniform(0.0, 0.02, (dm, dm // nh)) for _ in range(nh)]
    Wk = [np.random.uniform(0.0, 0.02, (dm, dm // nh)) for _ in range(nh)]
    Wo = np.random.uniform(0.0, 0.02, (dm, dm))
    res_stream = np.random.uniform(0.0, 1.0, (b, wps, dm))
    q, k, v, qk, attn_scores, attn_out, res = multihead_attention(res_stream, Wq, Wv, Wk, Wo)
    output = {
        "q": q,
        "k": k,
        "v": v,
        "qk": qk,
        "attn_scores": attn_scores,
        "attn_out": attn_out,
        "res": res
    }

    try:
        assert test_multihead_attention_soln(res_stream, Wq, Wv, Wk, Wo, output)
        print("Passed")
    except AssertionError as e:
        print("Failed, ", e)
