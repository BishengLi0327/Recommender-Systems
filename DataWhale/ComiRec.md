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

        $$v_{u} = V_{u}\[:, argmax(V_{u}^{T} e_{i})\]$$
        $$P_{\theta}(i|u) = \frac{\exp (v_{u}^{T})}{\sum_{k \in \mathcal{I}} \exp (v_{u}^{T} e_{k})}$$
        $$loss = \sum_{u \in \mathcal{U}} \sum_{i \in \mathcal{I}\_{u}} - \log P_{\theta} (i|u)$$

3. Aggregation Modules

    After the multi-interest extraction module, we obtain multiple interest embeddings for each user based on his/her past behavior. Each interest embedding can independently retrieve top-N items based on the inner production proximity. A basic and straightforward way to aggregate the items to obtain the top-N items is to merge and filter the items based on their inner production proximity with user interests, which can be formalized as
    $$f(u, i) = max_{1 \leq k \leq K} (e_{i}^{T} v_{u}^{(k)})$$
    However, it is not all about the accuracy of current recommender systems. People are more likely to be recommended with something new or something diverse. The
problem can be formulated in the following. Given a set $\mathcal{M}$ with $K \cdot N$ items retrieved from $K$ interests of a user $u$, find a set $\mathcal{S}$ with
$N$ items such that a pre-defined value function is maximized. Our framework uses a controllable procedure to solve this problem. We use the following value function $Q(u, S)$ to balance the accuracy and diversity of the recommendation by a controllable factor $\lambda \geq 0$,
    $$Q(u, S) = \sum_{i \in \mathcal{S}} f(u, i) + \lambda \sum_{i \in \mathcal{S}} \sum_{j \in \mathcal{S}} g(i, j)$$
    Here $g(i, j)$ is a diversity or dissimilarity function such as
    $$g(i, j) = \delta(CATE(i) \neq CATE(j))$$
    
    The greedy inference algorithm to approximately maximize the value function $Q(u, S)$, which is listed in the following
    <img width="387" alt="image" src="https://user-images.githubusercontent.com/49403324/208804303-27042e2e-d840-4f6c-ac30-785a901b1415.png">


**Experiments**

<img width="774" alt="image" src="https://user-images.githubusercontent.com/49403324/208804388-9bc71f16-cff5-425f-bb8c-638d0b1dbb22.png">


**Conclusion**

In this paper, we propose a novel controllable multi-interest framework for the sequential recommendation. Our framework uses a multi-interest extraction module to generate multiple user interests and uses an aggregation module to obtain the overall top-N items.

**Citation**
```
@inproceedings{cen2020controllable,
  title={Controllable multi-interest framework for recommendation},
  author={Cen, Yukuo and Zhang, Jianwei and Zou, Xu and Zhou, Chang and Yang, Hongxia and Tang, Jie},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2942--2951},
  year={2020}
}
