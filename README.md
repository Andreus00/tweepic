# Tweepic
Tweets topic analyzer



## Todos


### Urgent:
- [x] Finetune XLM-Roberta on tweets or import a pretrained model
- [x] Check if the tweet is a retweet and remove the "RT" from the tweet
- [ ] Put embeddings for hashtag at the end of sentence embeddings
- [x] Plot the cluster of tweets


### Not urgent:
- [ ] Better pipeline (NER)
- [ ] Change nicknames to User names
- [ ] Use information of like and retweets
- [ ] Check the number of followers in order to filter out irrelevant tweets (only during the online phase)


### Extras:
- [ ] Check if a person has a wikipedia page or not.
- [ ] Correct the Mispellings
- [ ] Write a script to download the tweets with twint

### Done


## Links
- Pipeline
  - https://github.com/JohnSnowLabs/spark-nlp/issues/7357
  - https://www.johnsnowlabs.com/understanding-the-power-of-transformers-a-guide-to-sentence-embeddings-in-spark-nlp/
- Models
  - https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment 
  - https://www.johnsnowlabs.com/importing-huggingface-models-into-spark-nlp/
  - https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32