# Session-based Recommendation with Graph Neural Networks (SR-GNN)

https://arxiv.org/abs/1511.06939

##### *2019 AAAI*

**Main Contributions**

1. SR-GNN models seperated session sequences into graph-strutured data and use graph neural networks  to capture complex item transitions. SR-GNN presents a noval perspective on modeling in the session-based recommendation scenario.
2. SR-GNN doesn't rely on user representations, but uses the session embedding to generate session-based recommendations, the session embedding can be obtained merely based on latent vectors of items involved in each single session.


**SR-GNN Framework**

<img width="801" alt="image" src="https://user-images.githubusercontent.com/49403324/208231350-c78cc9e3-da88-4f18-8ede-06c00c39fa19.png">


**SR-GNN**

1. Notations

    Session-based recommendation aims to predict which item a user will click next, solely based on the user’s current sequential session data without accessing to the 	long-term preference profile.
    
    |Notation|Meaning|
    |---|---|
    |$V = {v_{1}, v_{2}, \cdots}, v_{m}$| set consisting of all unique items involved in all the sessions |
    |$s = \[v_{s,1}, v_{s,2}, \cdots, v_{s,n}\]$| anonymous session sequence ordered by timestamp |
    |$v_{s,n+1}$| next click |
    |$\hat{y}$| output probability |
  
2. Constructing Session Graphs

    Each session sequence $s$ can be modeled as a directed graph $\mathcal{G}\_{s} = (\mathcal{V}\_{s}, \mathcal{E}\_{s})$. In this session graph, each node represents     an item $v_{s,i} \in V$ . Each edge $(v_{s,i-1}, v_{s, i}) \in \mathcal{E}\_{s}$ means that a user clicks item $v_{s,i}$, after $v_{s,i-1}$ in the session $s$.         Since several items may appear in the sequence repeatedly, we assign each edge with a normalized weighted, which is calculated as the occurrence of the edge           divided by the outdegree of that edge’s start node.
    
    We embed every item $v \in V$ into an unified embedding space and the node vector $v \in \mathbb{R}^{d}$ indicates the latent vector of item $v$ learned via graph     neural networks, where $d$ is the dimensionality. Based on node vectors, each session $s$ can be represented by an embedding vector $s$, which is composed of node     vectors used in that graph.

3. Learning Item Embeddings on Session Graphs

    Graph neural networks are well suited for session-based recommendation, because it can automatically extract features of session graphs with consideration of rich node connections. The learning process of node vectors in a session graph can be demonstrated as follows: (take node $v_{s,i}$ for example)
        
    <img width="287" alt="image" src="https://user-images.githubusercontent.com/49403324/208233480-1ac1f223-eed2-4a37-809b-6ff0f264da77.png">

    where $\textbf{H} \in \mathbb{R}^{d \times 2d}$ controls the weight, $z_{s,i}$ and $r_{s,i}$ are the reset and update gates respectively, $\[v_{1}^{t-1}, \cdots,       v_{n}^{t-1}\]$ is the list of node vectors in session $s$, $\sigma(\cdot)$ is the sigmoid function, and $\otimes$ is the element-wise multiplication operator.         $v_{i}\in \mathbb{R}^{d}$ represents the latent vector of node $v_{s,i}$. The connection matrix $A_{s} \in \mathbb{R}^{n \times 2n}$ determines how nodes in the       graph communicate with each other and $A_{s,i:} \in \mathbb{R}^{1 \times 2n}$ are the two columns of blocks in $A_{s}$ corresponding to node $v_{s, i}$.
    
    <img width="439" alt="image" src="https://user-images.githubusercontent.com/49403324/208234037-93af3ed2-799f-4e0d-90d9-3d6fa8ebae11.png">

4. Generate Session Embeddings

    SR-GNN represents a session directly by nodes involved in that session. To better predict the user's next clicks, SR-GNN develop a strategy to combine long-term       preference and current interests of the session, and use the combined embedding as the session embedding.
    
    For session $s = \[v_{s,1}, v_{s,2}, \cdots, v_{s,n}\]$, we consider the lcoal embedding $s_{l}$ and global embedding $s_{g}$ from the node vectors obtained from the last step:
    
    **Local Embedding**:
    $$s_{l} = v_{n}$$
    
    **Global Embedding**:
    $$\alpha_{i} = q^{T} \sigma (W_{1} v_{n} + W_{2} v_{i} + c)$$
    $$s_{g} = \sum_{i=1}^{n} \alpha_{i} v_{i}$$
    
    the hybrid embedding $s_{h}$ can be computed by taking linear transformation over the concatenation of the lcoal and global embedding vectors:
    $$s_{h} = W_{3}\[s_{l}; s_{g}\]$$


5. Making Recommendation and Model Training

    After obtained the embedding of each session, we compute the score $\hat{z}\_{i}$ for candidate item $v_{i} \in V$ by multiplying its embedding $v_{i}$ by session representation $s_{h}$:
    $$\hat{z}\_{i} = s_{h}^{T} v_{i}$$
    
    the softmax function is applied to get the output vector of the model $\hat{y}$:
    $$\hat{y} = softmax(\hat{z})$$
    
    For the session graph, the loss function is defined as the cross-entropy of the prediction and the ground truth.
    $$\mathcal{L}(\hat{y}) = - \sum_{i=1}^{m} y_{i} \log (\hat{y}\_{i}) + (1 - y_{i}) \log (1 - \hat{y}\_{i})$$


**Experiments**

1. The performance of SR-GNN with other baseline methods over three datasets.

    <img width="374" alt="image" src="https://user-images.githubusercontent.com/49403324/208234794-36e49817-cd3a-4824-a63d-179ec76a77af.png">
    
2.  Different Connection Schemes and Different Session Embeddings.

    <img width="379" alt="image" src="https://user-images.githubusercontent.com/49403324/208234857-ade5dc59-50d9-426a-85fb-3f5a222c61e7.png">

3. Analysis on Session Sequence Lengths.

    <img width="291" alt="image" src="https://user-images.githubusercontent.com/49403324/208234881-62ebaa71-b281-4409-ac8d-e0eee287c9c0.png">


**Conclusion**

Session-based recommendation is indispensable where users’ preference and historical records are hard to obtain. This paper presents a novel architecture for session-based recommendation that incorporates graph models into representing session sequences. The proposed method not only considers the complex structure and transitions between items of session sequences, but also develops a strategy to combine long-term preferences and current interests of sessions to better predict users’ next actions.


**Citation**

```
@inproceedings{wu2019session,
  title={Session-based recommendation with graph neural networks},
  author={Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={33},
  number={01},
  pages={346--353},
  year={2019}
}
