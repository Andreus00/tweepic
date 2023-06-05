# Tweepic
Tweets topic analyzer



## Todos


### Urgent:
- [ ] Cluster the tweets
- [ ] Use the embeddings of the tweets to find an online cluster
- [ ] Use the embeddings of the words to crate the edges of the graph



### Not urgent:
- [ ] Put embeddings for hashtag at the end of sentence embeddings
- [ ] Better pipeline (NER)
- [ ] Change nicknames to User names
- [ ] Use information of like and retweets
- [ ] Check the number of followers in order to filter out irrelevant tweets (only during the online phase)
- [ ] Bias of language
- [ ] Embeddings of hashtags as another "Pipeline"


### Extras:
- [ ] Check if a person has a wikipedia page or not.
- [ ] Correct the Mispellings
- [ ] Write a script to download the tweets with twint
- [ ] Move pca to the end of the pipeline

### Done
- [x] Finetune XLM-Roberta on tweets or import a pretrained model
- [x] Check if the tweet is a retweet and remove the "RT" from the tweet
- [x] Plot the cluster of tweets
- [x] step in the pipeline that averages the embeddings of the words in the tweet


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
