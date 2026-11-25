# MNIST: optimizers and hyperparameter search

Machine Learning course project. It has three parts:

1. A small fully connected network written from scratch in NumPy (forward pass,
   backprop, mini-batch training)
2. The same task in Keras, comparing SGD, Adam, RMSprop and Adagrad
3. Hyperparameter search over learning rate, optimizer and epochs, done two ways:
   random search and a simple genetic algorithm

The search budgets are kept small (a few trials and generations) so it runs in a
reasonable time on a laptop.

```
pip install numpy tensorflow scikit-learn matplotlib
python mnist_optimizers.py
```

Team project with [Seif Wael](https://github.com/Seiffwaell) and [Mohamed Waleed](https://github.com/MrWeza).
