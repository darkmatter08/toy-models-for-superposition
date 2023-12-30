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
Assume features are vectors.  
  
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
at most 4 features in the direction positive X, negative X, positive Y and negative Y  
Thus the neural network can only learn 4 features  
and thus only learns Top-4 features from data  

But when Superposition happens, the neural network will  
can arrange features like a windmill,  
squeezing 5 features(2pi/5 angle) or 8 features(2pi/8)  
onto 2d XY space.
