# Tweepic
Tweets topic analyzer



## Todos


### Urgent:
- [x] Finetune XLM-Roberta on tweets or import a pretrained model
- [x] Check if the tweet is a retweet and remove the "RT" from the tweet
- [x] Plot the cluster of tweets
- [ ] step in the pipeline that averages the embeddings of the words in the tweet
- [ ] Cluster the tweets


### Not urgent:
- [ ] Put embeddings for hashtag at the end of sentence embeddings
- [ ] Better pipeline (NER)
- [ ] Change nicknames to User names
- [ ] Use information of like and retweets
- [ ] Check the number of followers in order to filter out irrelevant tweets (only during the online phase)


### Extras:
- [ ] Check if a person has a wikipedia page or not.
- [ ] Correct the Mispellings
- [ ] Write a script to download the tweets with twint
- [ ] Move pca to the end of the pipeline

### Done


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