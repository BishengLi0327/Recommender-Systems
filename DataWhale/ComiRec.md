# Controllable Multi-Interest Framework for Recommendation (ComiRec)

https://arxiv.org/abs/2005.09347

#### *KDD 2020*


**Main Contribution**

1. This paper proposes a comprehensive framework that integrates the controllability and multi-interest components in a unified recommender system.
2. This paper investigate the role of controllability on personalized systems by implementing and studying in an online recommendation scenario.


**Motivation**

<img width="708" alt="image" src="https://user-images.githubusercontent.com/49403324/208796761-28905125-6b25-48b8-9599-56c80f4ebb0c.png">

A motivating example of our proposed framework. An e-commerce platform user, Emma, has multiple interests including jewelry, handbags, and make-ups. Our multi-interest extraction module can capture these three interests from her click sequence. Each interest retrieves items from the large-scale item pool based on the interest embedding independently. An aggregation module combines items from different interests and outputs the overall top-N recommended items for Emma.


**ComiRec Framework**

<img width="632" alt="image" src="https://user-images.githubusercontent.com/49403324/208796910-dddbccef-5d46-4f39-935b-f11c84de600d.png">


**ComiRec**

1. Problem Formulation

    Assume we have a set of users $u \in \mathcal{U}$ and a set of items $i \in \mathcal{I}$. For each user, we have a sequence of user historical bahaviours $(e_{1}^{u}, e_{2}^{u}, \cdots, e_{n}^{u})$, sorted by time of occurrence. Given historical occurrence interactrions, the problem of *sequence recommendation* is to predict the next items that the user might be interacted with.
    
    <img width="392" alt="image" src="https://user-images.githubusercontent.com/49403324/208798539-10a47d0a-f046-4dba-8811-46986252e543.png">

2. Multi-Interest Framework

    Most of existing mathcing models only generate a single embedding vector for each user. This suffers from the lack of expressiveness of a single embedding since real-world customers usually have several kinds of items in their minds and these items are often for different uses and vary a lot in categories. Such behaviors of real-world customers highlight the need to use multiple vectors to represent their multiple interests. In this paper, we explore two methods, dynamic routing method and self-attentive method, as our multi-interest extraction module.

    1. ***Dynamic Routing***

        Using dynamic routing, the item embeddings of the user sequences can be viewed as primary capsures, and the multiple user interests can be seen as the interest capsules. Let $e_{i}$ be the capsule $i$ of the primary layer. We then give the computation of the capsule $j$ of the next layer based on primary capsules.
        
        1. First compute the prediction vector as
            $$\hat{e}\_{j|i} = W_{ij} e_{i}$$
        2. Then the total input to the capsule $j$ is the weighted sum over all prediction vecotrs $\hat{e}\_{j|i}$ as
            $$s_{j} = \sum_{i} c_{ij} \hat{e}\_{j|i}$$
        3. $c_{ij}$ are the coupling coefficients determined by the iterative dyunamic routing process as
            $$c_{ij} \frac{\exp (b_{ij})}{\sum_{k} \exp (b_{ik})}$$
        4. Finally, the vector of capsule $j$ is computed by
            $$v_{j} = squash(s_{j}) = \frac{||s_{j}||^{2}}{1 + ||s_{j}||^{2}} \frac{s_{j}}{||s_{j}||}$$
        
        The output interest capsules of the user $u$ are then formed as a matrix $V_{u} = \[v_{1}, \cdots, v_{K}\] \in \mathbb{R}^{d \times K}$.
        
    2. ***Self-Attentive Method***

        Given the embeddings of user behaviours, $H \in \mathbb{R}^{d \times n}$, we use the self-attention mechanism to obtain a matrix of weights:
        $$A = softmax(W_{2}^{T} tanh(W_{1}H))^{T}$$
        The final matrix of user interests $V_{u}$ can be computed by
        $$V_{u} = HA$$

    3. ***Model Training***

3. Aggregation Modules

**Experiments**


**Conclusion**


**Citation**
