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
	
	|$V = {v_{1}, v_{2}, \cdots, v_{m}}$| the set consisting of all unique items ionvolved in all sessions|
	|$s = \[v_{s, 1}, v_{s, 2}, \cdots, v_{s, n}}\]$|an anonymous session sequence s ordered by timestamps|
	|$v_{s, n+1}$|next click (sequence label)|
	|$\hat{y}$|output probabilities|
  
2. Constructing Session Graphs


3. Learning Item Embeddings on Session Graphs


4. Generate Session Embeddings


5. Making Recommendation and Model Training



**Experiments**
