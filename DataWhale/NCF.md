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
        $$L_{sqr} = \sum_{(u,i) \in \mathcal{Y} \times \mathcal{Y}^{-}} \omega_{ui} (y_{ui} - \hat{y}\_{ui})^2 $$
        where $\mathcal{Y}$ denotes the set of observed interactions in **Y**, and $\mathcal{Y}^{-}$ denotes the set of negative instances, which can be all (or sampled from) unobserved interactions;  
        We can view the prediction score $\hat{y}\_{ui}$ as how likely $i$ is relevant to $u$. To endow NCF with such a probabilistic explanation, we need to constrain the output $\hat{y}\_{ui}$ in the range of $\[0, 1\]$, which can be easily achieved by using a probabilistic function (e.g., the Logistic
or Probit function) as the activation function for the output layer. With the above setting, the likelihood function can be defined as:
        $$P(\mathcal{Y}, \mathcal{Y}^{-}|\textbf(P), \textbf{Q}, \Theta_{f}) = \prod_{(u, i) \in \mathcal{Y}} \hat{y}\_{ui} \prod_{(u, i) \in \mathcal{Y}^{-}} (1-\hat{y}\_{ui})$$
        Then the loss function can be formulated as:
        $$L = - \sum_{(u, i) \in \mathcal{Y}} \log \hat{y}\_{ui}  - \sum_{(u, i) \in \mathcal{Y}^{-}} \log (1-\hat{y}\_{ui})  = - \sum_{(u, i) \in \mathcal{Y} \cup \mathcal{Y}^{-}} y_{ui} \log \hat{y}\_{ui} + (1 - y_{ui}) \log (1 - \hat{y}\_{ui})$$
        
2. Genealized Matrix Factorization(GMF)
    
    MF can be interpreted as a special case of NCF framework.  
    Let the user latent vector $p_{u}$ be $P^{T}v_{u}^{U}$ and item latent vector $q_{i}$ be $Q^{T}v_{i}^{I}$. We define the mapping function of the first neural CF layer as:
    $$\phi_{1}(p_{u}, q_{i}) = p_{u} \otimes q_{i}$$
    where $\otimes$ denotes the element-wise products of vectors. The project the vector to the output layer:
    $$\hat{y}\_{ui} = a_{out}(h^{T}(p_{u}\otimes q_{i}))$$
    where $a_{out}$ and $h$ denote the activation and edge weights of the output layer.  
    Intuitively, if we use an identity function for $a_{out}$ and enforce $h$ to be a uniform vector of 1, we can exactly recover the MF model. So we say that MF can be interpreted as a special case of NCF framework. And we term this model as GMF, short for *Generalized Matrix Factorization*.
    
3. Multi-Layer Perceptron(MLP)
    
    Since NCF adopts two pathways to model users and items, it is intuitive to combine the features of two pathways by concatenating them. However, simple concatenation is insufficient for modeling the collaborative filtering effect. To address this issue, NCF uses a standard MLP to learn the interaction between user and item latent features, which endows the model a large level of flexibility and non-linearity. The MLP model under our NCF is defined as:
    $$Z_{1} = \phi_{1}(p_{u}, q_{i})$$
    $$\phi_{2}(z_{1}) = a_{2}(W^{T}\_{2}z_{1} + b_{2})$$
    $$\cdots \cdots$$
    $$\phi_{L}(z_{L-1}) = a_{L}(W_{L}^{T}z_{L-1} + b_{L})$$
    $$\hat{y}\_{ui} = \sigma(h^{T} \phi_{L}(z_{L-1}))$$
    where $W_{x}$, $b_{x}$ and $a_{x}$ denote the weight matrix, bias vector and activation function for the x-th layer's perceptron.

4. Fusion of GMF and MLP

    GMF and MLP are two instantiations of NCF. How can we fuse GMF and MLP under NCF framework?  
    To provide more flexibility to the fused model, NCF allows GMF and MLP to learn separate embedding layer and combines the two models by concatenating the last hidden layer. The belowing figure illustrates the proposal:
    <img width="551" alt="image" src="https://user-images.githubusercontent.com/49403324/206997880-a3fd239a-8584-4797-9724-61cc6dbfa1a4.png">
    
    the formulation of which is given as:
    $$\phi^{GMF} = p_{u}^{G} \otimes q_{i}^{G}$$
    $$\phi^{MLP} = a_{L}(W_{L}^{T}(a_{L-1}(\cdots a_{2}(W_{2}^{T} \[(p_{u}^{M})^{T}, (q_{i}^{M})^{T}\]^{T} + b)\_{2})\cdots))+b_{L})$$
    $$\hat{y}\_{ui} = \sigma(h^{T} \[(\phi^{GMF})^{T}, (\phi^{MLP})^{T}\]^{T})$$
    where $p_{u}^{G}$ and $p_{u}^{M}$ denote the user embedding for GMF and MLP parts, and so as $q_{i}^{G}$ and $q_{i}^{M}$. The model combines the linearity of MF and non-linearity of DNNs for modeling user–item latent structures and is named as "NeuMF", short for *Neural Matrix Factorization*.


**Experiments**

*RQ1*: Do the proposed NCF methods outperform the state-of-the-art implicit collaborative filtering methods?

<img width="594" alt="image" src="https://user-images.githubusercontent.com/49403324/207029662-d3db0763-93c9-4c4e-a565-49e300714734.png">

*RQ2*: How does the proposed optimization framework (log loss with negative sampling) work for the recommendation task?

<img width="718" alt="image" src="https://user-images.githubusercontent.com/49403324/207031073-b8919010-5993-4dce-938e-211773c9c0a2.png">

*RQ3*: Are deeper layers of hidden units helpful for learning from user–item interaction data?

<img width="337" alt="image" src="https://user-images.githubusercontent.com/49403324/207031322-ad13f8b5-a6cc-4e87-acbe-daaf5f17cba5.png"><img width="339" alt="image" src="https://user-images.githubusercontent.com/49403324/207031367-64faaec4-1a0c-4eb9-bc81-ac6d3aed6a3d.png">


**Conclusion**

The paper explores neural network architectures for collaborative filtering. They devised a general framework NCF and proposed three instantiations — GMF, MLP and
NeuMF — that model user–item interactions in different ways. The framework is simple and generic; it is not limited to the models presented in this paper, but is designed to serve as a guideline for developing deep learning methods for recommendation. 


**Citation**

```
@inproceedings{he2017neural,
  title={Neural collaborative filtering},
  author={He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  booktitle={Proceedings of the 26th international conference on world wide web},
  pages={173--182},
  year={2017}
}

