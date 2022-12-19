# Multi-Interest Network with Dynamic Routing for Recommendation at Tmall (MIND)

https://arxiv.org/abs/1904.08030

**Main Contribution**:

1. To capture diverse intertests of users from user behaviour, this paper designed the multi-interest extractor layer, which utilizes dynamic routing to adaptively aggregate user's historial behaviours into user representation vectors.
2. By using user representation vectors produced by the multi-interest extractor layer and a newly proposed label-aware attention layer, a deep neural network is built for personalized recommendation tasks. Comnpare with existing methods, MIND shows superior performance on several public datasets and one industrial dataset from Tmall.
3. A system is constructed to implement the whole pipeline for data collecting, model training and online serving to deploy MIND for serving billion-scale users at Tmall. The deployed system significantly improves the click-through rate of the home page on Mobile Tmall APP.


**MIND Framework**

<img width="810" alt="image" src="https://user-images.githubusercontent.com/49403324/208366456-f2242c88-df57-4e45-89a3-f69c49528a63.png">


**MIND**

1. Problem Formalization


2. Embedding & Pooling Layer


3. Multi-Interest Extractor Layer


4. Label-aware Attention Layer


5. Training & Serving


6. Connections with Existing Methods


**Wxperiments**

