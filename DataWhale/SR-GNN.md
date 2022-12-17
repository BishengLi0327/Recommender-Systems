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
  
2. Constructing Session Graphs

    Each session sequence $s$ can be modeled as a directed graph $\mathcal{G}\_{s} = (\mathcal{V}\_{s}, \mathcal{E}\_{s})$. In this session graph, each node represents     an item $v_{s,i} \in V$ . Each edge $(v_{s,i-1}, v_{s, i}) \in \mathcal{E}\_{s}$ means that a user clicks item $v_{s,i}$, after $v_{s,i-1}$ in the session $s$.         Since several items may appear in the sequence repeatedly, we assign each edge with a normalized weighted, which is calculated as the occurrence of the edge           divided by the outdegree of that edge’s start node.
    
    We embed every item $v \in V$ into an unified embedding space and the node vector $v \in \mathbb{R}^{d}$ indicates the latent vector of item $v$ learned via graph     neural networks, where $d$ is the dimensionality. Based on node vectors, each session $s$ can be represented by an embedding vector $s$, which is composed of node     vectors used in that graph.

3. Learning Item Embeddings on Session Graphs


4. Generate Session Embeddings


5. Making Recommendation and Model Training



**Experiments**
