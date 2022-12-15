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
        Where $g$ is a smooth and bounded function such as a logistic sigmoid function $x_{t}$ is the input of the unit at time $t$. An RNN outputs a probability                 distribution over the next element of the sequence, given its current state $h_{t}$.
        
    Gated Recurrent Unit (GRU):
    ![image](https://user-images.githubusercontent.com/49403324/207804249-8b55d284-2a37-4361-bfbc-8274adb28071.png)
    where $r_{t}$ is the reset gate and $z_{t}$ is the update data. The GRU is a more elaborate model of an RNN unit that aims at dealing with the vanishing gradient         problem. GRU gate essentially learn when and by how much to update the hidden state of the unit.
    
2. Customizing the GRU model

    The general framework of GRU4Rec is illustrated in the above figure, which depicts the representation of a single event within a time series of events. When           used in session-based recommendations, the input of the network is the actual state of the network is the actual state of the session while the output is the           item of the next event in the session.
    
    1. Session-Parallel Mini-Batches

        Session-Parallel Mini-Batched are used to accelerate training.  
        First, we create an order for each session.  
        Second, we used the first event of the first $X$ sessions to form the input of the first mini-batch . The second mini-batch is formed from the second events           and so on. If any of the sessions end, the next available session is put in its place.  
        <img width="437" alt="image" src="https://user-images.githubusercontent.com/49403324/207808027-2c5ca456-a754-4b8f-bf10-de7cb4b48776.png">
        
    2. Sampling on the output
        
        Recommender systems are especially useful when the number of items is large. For large sites with a few million items, calculating a score for each item in             each step would make the algorithm scale with the product of the number of items and the number of events. This would be unusable in practice. There are two popular ways to deal with the issue:
        1. The natural interpretation of an arbitrary missing event is that the user did not know about the existence of the item and thus there was no interaction.
        2. We should sample items in proportion of their popularity.

        In practice, we use the items from the other training examples of the mini-batch as negative examples.

    3. Ranking Loss

        **BPR**: Bayesian Personalized Ranking
        $$L_{s} = \frac{1}{N_{S}} \cdot \sum_{j=1}^{N_{S}} \log (\sigma (\hat{r}\_{s,i}) - \hat{r}\_{s, j})$$
        where $N_{S}$ is the sample size, $\hat{r}\_{s,k})$ is the score on item $k$ at the given point of the session, $i$ is the desired item and $j$ are the                 negative samples.
        
        **TOP1**:
        $$L_{S} = \frac{1}{N_{S}}  \sum_{j=1}^{N_{S}} (\sigma (\hat{r}\_{s,j} - \hat{r}\_{s,i}) + \sigma (\hat{r}\_{s,j}^{2})) $$
        

**Experiments**
1. Baseline Results:

     <img width="399" alt="image" src="https://user-images.githubusercontent.com/49403324/207811710-680ab673-9e56-4e5f-a895-9c10b0c1eb15.png">
     
2. GRU4Rec Results

    <img width="630" alt="image" src="https://user-images.githubusercontent.com/49403324/207811823-002929b5-0317-4cbd-b9d6-fb488fa4cde0.png">


**Conclusion**

This paper applied a kind of modern recurrent neural network (GRU) to new application domain: recommender systems. They chose the task of session based recommendations, because it is a practically important area, but not well researched. They modified the basic GRU in order to fit the task better by introducing session-parallel mini-batches, mini-batch based output sampling and ranking loss function and showed that the method can significantly outperform popular baselines
that are used for this task.


**Citation**
