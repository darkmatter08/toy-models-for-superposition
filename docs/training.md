# INSTALL
```
pyenv install 3.8.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# RUN
1. Setup environment  
```
source venv/bin/activate
export PYTHONPATH="/home/username/toy-models-for-superposision/:${PYTHONPATH}"
```
  
2. One Weight Linear Net  
Training  
learning rate = 0.001  
weight decay = 0.01  
```
python src/superposition/a1_train_one_weight.py
```
  
Visualize w1 @ w1.T, bias  
```
python src/superposition/a2_viz_one_identity_bias.py
```
  
Visualize w1, bias  
```
python src/superposition/a3_viz_one_identity_bias.py
```

3. Two Weight Linear Net  
Training  
learning rate = 0.001  
weight decay = 0.01  
```
python src/superposition/b1_train_one_weight.py
```
  
Visualize w1 @ w2, bias  
```
python src/superposition/b1_train_one_weight.py
```

Visualize w1, w2, bias
```
python src/superposition/b1_train_one_weight.py
```

