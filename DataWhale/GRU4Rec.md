# Session-Based Recommendation with Recurrent Neural Networks(GRU4Rec)
https://arxiv.org/pdf/1511.06939.pdf
##### *ICLR 2016*

**Main Contribution**:
1. This paper applies recurrent neural networks (RNNs) on recommender systems domain. By modeling the session-based data, more accurate recommendations can be provided.
2. The approach also considers practical aspects of the task and introduces several modifications to the classific RNNs that make it viable for the specific problem.
3. Experiments results on two datasets show marked improvements over widely used approaches.

**GRU4Rec Framework**
<img width="317" alt="image" src="https://user-images.githubusercontent.com/49403324/207802990-2afdce54-ac24-4659-98c7-f23d560d028c.png">
General architecture of the network. Processing of one event of the event stream at once.

**GRU4Rec (Recommendations with RNNs)**

1. Preliminaries
    Recurrent neural networks:
        $$h_{t} = g(W x_{t} + U h_{t-1})$$
2. Customizing the GRU model
    1. Session-Parallel Mini-Batches
    2. Sampling on the output
    3. Ranking Loss


**Experiments**
