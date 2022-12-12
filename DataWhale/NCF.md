# Neural Collaborative Filtering (NCF)
https://arxiv.org/pdf/1708.05031.pdf
##### *2017 International World Wide Web Conference*

**Main Contribution**: 
1. This paper presents a neural network architecture to model latent features of users and items and devise a general framework NCF for collaborative filtering based
on neural networks.
2. This paper shows that MF can be interpreted as a specialization of NCF and utilize a multi-layer perceptron to endow NCF modelling with a high level of non-linearities.

**NCF Framework**:  
<img width="458" alt="image" src="https://user-images.githubusercontent.com/49403324/206973238-1752e378-2b19-44ad-b594-242f80b36bfc.png">

**Neural Collaborative Filtering**
1. General Framework
    The framework is illustrated in the above figure.  
    The total pipeline is: **Input Layer $\rightarrow$ Embedding Layer $\rightarrow$ Neural CF Layers $\rightarrow$ Output Layer**  
    The NCF preictive model can be formulated as:
    $$\hat{y}\_{ui}=f(P^{T} v_{u}^{U}, Q^{T} v_{i}^{I} | P, Q, \Theta_{f})$$
    where $P \in \mathbb{R}^{M \times K}$ and $Q \in \mathbb{R}^{N \times K}$, denoting the latent factor matrix for users and items, respectively; and $\Theta_{f}$ denotes the model parameters of the interaction function $f$.
    1. *Learning NCF*  
        To learn model parameters, existing pointwise methods largely perform a regression with squared loss:
        $$L_{sqr} = \sum_{(u,i) \in \mathcal{Y} \times \mathcal{Y}^{-}}$$
        where $\mathcal{Y}$ denotes the set of observed interactions in **Y**, and $\mathcal{Y}^{-}$ denotes the set of negative instances, which can be all (or sampled from) unobserved interactions;  
        We can view the prediction score $\hat{y}\_{ui}$ as how likely $i$ is relevant to $u$. To endow NCF with such a probabilistic explanation, we need to constrain the output $\hat{y}\_{ui}$ in the range of $\[0, 1\]$, which can be easily achieved by using a probabilistic function (e.g., the Logistic
or Probit function) as the activation function for the output layer. With the above setting, the likelihood function can be defined as:
        $$P(\mathcal{Y}, \mathcal{Y}^{-}|\textbf(P), \textbf{Q}, \Theta_{f}) = \prod_{(u, i) \in \mathcal{Y}} \hat{y}\_{ui} \prod_{(u, i) \in \mathcal{Y}^{-}} (1-\hat{y}\_{ui})$$
        Then the loss function can be formulated as:
        $$L = - \sum_{(u, i) \in \mathcal{Y}} \log \hat{y}\_{ui}  - \sum_{(u, i) \in \mathcal{Y}^{-}} \log 1-\hat{y}\_{ui}  = - \sum_{(u, i) \in \mathcal{Y} \cup \mathcal{Y}^{-}} y_{ui} \log \hat{y}\_{ui} + (1 - y_{ui}) \log (1 - \hat{y}\_{ui})$$
2. Genealized Matrix Factorization(GMF)
3. Multi-Layer Perceptron(MLP)
4. Fusion of GMF and MLP
