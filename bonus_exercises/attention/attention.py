import numpy as np
from soln.attention import multihead_attention_soln, set_params

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
        out: (b, wps, dm) np.ndarray
    '''
    # TODO: implement this function
    return None

if __name__ == "__main__":
    np.random.seed(0)
    set_params(b, wps, dm, nh)
    assert dm % nh == 0, "dm must be divisible by nh"
    # list of parameters for each head d_model x d_head
    Wq = [np.random.uniform(0.0, 0.02, (dm, dm // nh)) for _ in range(nh)]
    Wv = [np.random.uniform(0.0, 0.02, (dm, dm // nh)) for _ in range(nh)]
    Wk = [np.random.uniform(0.0, 0.02, (dm, dm // nh)) for _ in range(nh)]
    Wo = np.random.uniform(0.0, 0.02, (dm, dm))
    res_stream = np.random.uniform(0.0, 1.0, (b, wps, dm))
    res = multihead_attention(res_stream, Wq, Wv, Wk, Wo)
    res_soln = multihead_attention_soln(res_stream, Wq, Wv, Wk, Wo)
    try:
        assert np.allclose(res, res_soln)
        print("Passed")
    except:
        print("Failed")
