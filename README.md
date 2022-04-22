# classifier-tools


0. different depth and hidden dim for each layer (reduce it)
1. different autoencoders, sparse, noisy, vae???
2. l1 regularization, dropout2d
3. augs
4. conv2d, bn, more interesting activations
5. GradCAM
6. high value of hidden dim means we can overfit too easier, so wee need to use
regularization techics. Tried using hidden_dim=64, 32 but it converges significantly slower (than with hidden_dim=128).
We dont have aim to clusterize data anyway.
The best way to find hidden_dim is hyperparameters search using hydra
(need to define the best metric based on classifier not mse loss)
