# Tweepic: A Novel Approach to Tweet Clustering

Tweepic is a cutting-edge project aimed at the **clustering of live tweets**. Our goal is to collect tweets and discern which ones are discussing the **same topic**,
thereby enabling us to group them accordingly. The name '**Tweepic**' is a portmanteau of 'tweet' and 'topic', reflecting our project's core objective.

Unlike traditional methods that group tweets based solely on hashtags, Tweepic proposes a **novel approach** that also considers the **proximity of sentences** and the similarity of **words** through their embeddings. Our process begins by determining a proximity measure for sentences, words, and hashtags. Using this measure, we construct a **graph** where each vertex represents a tweet, and the edges represent the **k-th nearest tweets**, weighted by their distance. 
Subsequently, a **classifier** is employed to decide which edges should be cut due to significant differences between the connected tweets. This results in the final graph, where each **connected component** represents a cluster of tweets discussing the same topic.

### Team Members

- **Andrea Sanchietti**  - sanchietti.1883210@studenti.uniroma1.it
- **Francesco Palandra** - palandra.1849712@studenti.uniroma1.it



## Todos


### Urgent:
- [x] Cluster the tweets
  - [x] Use the embeddings of the tweets to find neighbors
  - [x] Use the embeddings of the words to create the edges of the graph
  - [x] Create a GraphFrame
  - [x] Delete weak edges based on the weights
  - [x] Use the graph to find the clusters
  - [x] Find the connected components
  - [x] Find an unique color for the clusters
- [ ] Bring it live
  - [ ] Collect live tweets
  - [ ] Assign to a cluster
  - [ ] Create new cluster
- [ ] Filtering
  - [ ] Tweets with few words
  - [ ] Tweets with no hashtag
  


### Not urgent:
- [ ] Use information of like and retweets
- [ ] Check the number of followers in order to filter out irrelevant tweets (only during the online phase)
- [ ] Manage memory



### Extras:
- [ ] Check if a person has a wikipedia page or not.
- [ ] Change nicknames to User names

### Done
- [x] Finetune XLM-Roberta on tweets or import a pretrained model
- [x] Check if the tweet is a retweet and remove the "RT" from the tweet
- [x] Plot the cluster of tweets
- [x] step in the pipeline that averages the embeddings of the words in the tweet
- [x] Every Operation in the pipeline
- [x] Feature enhance
  - [x] Add time feature to the final embeddings
  - [x] Embeddings of hashtags as another "Pipeline"


## Links
- Pipeline
  - https://github.com/JohnSnowLabs/spark-nlp/issues/7357
  - https://www.johnsnowlabs.com/understanding-the-power-of-transformers-a-guide-to-sentence-embeddings-in-spark-nlp/
- Models
  - https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment 
  - https://www.johnsnowlabs.com/importing-huggingface-models-into-spark-nlp/
  - https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32
- Papers
  - https://link.springer.com/article/10.1007/s12065-021-00696-6
  - https://arxiv.org/pdf/1505.05657v1.pdf
  - https://medium.com/analytics-vidhya/congressional-tweets-using-sentiment-analysis-to-cluster-members-of-congress-in-pyspark-10afa4d1556e
- Finetune
  - https://colab.research.google.com/drive/1IAA1h8u53O1hi9807u7oOFuT3728N0-n?usp=sharing#scrollTo=fvM2mCi2yC8c