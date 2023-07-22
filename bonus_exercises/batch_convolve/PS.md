# PS
This is a bonus excercise. You need to implement the modules in [batch_convolve.py](./batch_convolve.py) for 2d convolution. The api for convolution should look like the following:
```
 kernel = np.ones(2, 5, 3, 3) # c_inxc_outxhxw
 input = np.random.randint(32, 2, 10, 12) #nxcxhxw
 output = convolve(input, kernel, stride=(1, 1), padding = (0, 0)) #shape = 32, 2, x, y
```

Write simple test cases for the same. Make an additional API which takes in the output dims and pads the input accordingly: 
```
output = pad_and_convolve(np.array, np.array, stride = tuple(int, int), out_dims = np.shape)
```

You will need numpy and pytest to do this exercise. To install them, run `pip install -r requirements.txt`
You can test the results by running `pytest` on your console.