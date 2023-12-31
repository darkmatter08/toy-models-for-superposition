# INSTALL
```
pyenv install 3.8.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Toy Models for Superposition
This is a replication of the results from: 
https://transformer-circuits.pub/2022/toy_model/index.html

# Summary
## Superposition
  
Question:  
Given a simple AI model:  
1.  y = decoder(encoder(x))  
so that y = x  
  
2. where the encode will reduce the dimension of the features from:  
num_of_feat to num_of_middle   
  
3. the decoder will reverse from:  
num_of_middle to num_of_feat

4. num_of_feat > num_of_middle  

Under what conditions will the AI be able to predict y = x
which means it can learn more features then the num_of_middle neurons? 
Thus achieving Superposition of features onto middle neurons?  
  
Answer: 
When the following two conditions are met:  
1. features are sparse,  
2. and RELU cut off is used  
  
The next section will explain the setup to test the above results.  
  
## Setup
Build a network that outputs same feature vectors as inputs:  
y = f(x)  
x = (num_of_data_points, num_of_feat)  
y = (num_of_data_points, num_of_feat)  

Two networks are tested:  
src/superposition/toynet.py  
  
1. One Weight Linear Net  
  
    y = RELU(x * w1 * w1.T + b)  
    
    num_of_feat = 20  
    num_of_middle = 5  
  
    encoder: w1.shape = (num_of_feat, num_of_middle)  
    decoder: w1.T.shape = (num_of_middle, num_of_feat)  
             b.shape = (num_of_feat, )  
  
constrained the decoder to use transposed of w1  
  
2. Two Weights Linear Net     
  
    y = RELU(x * w1 * w2 + b)  
  
    num_of_feat = 20  
    num_of_middle = 5  
  
    encoder: w1.shape = (num_of_feat, num_of_middle)  
    decoder: w2.shape = (num_of_middle, num_of_feat)  
             b.shape = (num_of_feat, )

The network is trained using different sparsity levels:  
[0, 0.7, 0.9, 0.99, 0.999]  
0 = 100% of the the time a features appears  
0.7 = only 30% of the time a feature appears  
  
## Results
One Weight Linear Net:  
1. The following chart is a visualization of 
square box = w1 * w1.T 
single column = b
accross different sparsity levels
(src/viz/03_all_sparsity_one_weight.png)


This repo shows that if two conditions:  
1. features are sparse,  
2. and RELU cut off is used  
  
The neural network will entangle these features  
such that when one feature is detected,   
it will cause a set of entangled features to also be detected.  

The activation function will cut off the weaker features  
and only allow few strong features to pass through.  
Here feature strength is measured by the magnitude of the vector.  

Normally, without 2 conditions above, in an 2D XY space, you can represent:  
at most 4 features in the direction positive X, negative X, positive Y and negative Y directions  
Thus the neural network can only learn 4 features which are 90 degrees apart  
and thus only learns Top-4 weighted features from data  

But when Superposition happens, the neural network will  
arrange vector features into a windmill structure like roots of unity,  
squeezing 5 features(2pi/5 angle) or 8 features(2pi/8)  
onto 2d the XY space.  
  
So when a feature is detected, since the vector features are not 90 degrees(orthogonal)  
another correlated features at the side will also fire  
but since the features are sparse, it is unlikely that both fired at the same time in reality  
the RELU will cut it off which allows the network to squeeze more features  
resulting in angles between features to be < 90 degrees (non independent)  
 
