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

        
        
    3. ***Self-Attentive Method***
    4. ***Model Training***

3. Aggregation Modules

**Experiments**


**Conclusion**


**Citation**
