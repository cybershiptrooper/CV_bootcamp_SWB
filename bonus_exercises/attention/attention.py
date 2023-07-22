import numpy as np
from soln.attention import multihead_attention_soln, set_params

b = 32 # batch size
wps = 64 # words per sentence
dm = 128 # dimension of model
nh = 8 # number of heads

'''
PS: implement the multihead attention function below
you must parallelize the computation of the attention heads
'''

def multihead_attention(res_stream, Wq, Wv, Wk):
    # q, k, v: (b, wps, dm)
    # nh: number of heads
    # dh: dimension of each head
    # return: (b, wps, dm)
    # TODO: implement this function
    return None

if __name__ == "__main__":
    np.random.seed(0)
    set_params(b, wps, dm, nh)
    Wq = np.random.uniform(0.0, 0.02, (dm, dm))
    Wv = np.random.uniform(0.0, 0.02, (dm, dm))
    Wk = np.random.uniform(0.0, 0.02, (dm, dm))
    res_stream = np.random.uniform(0.0, 1.0, (b, wps, dm))
    res = multihead_attention(res_stream, Wq, Wv, Wk)
    res_soln = multihead_attention_soln(res_stream, Wq, Wv, Wk)
    try:
        assert np.allclose(res, res_soln)
        print("Passed")
    except:
        print("Failed")
