# Tweepic
Tweets topic analyzer



## Todos


### Urgent:
- [ ] Finetune XLM-Roberta on tweets or import a pretrained model
- [ ] Check if the tweet is a retweet and remove the "RT" from the tweet
- [ ] Plot the cluster of tweets
- [ ] Write a script to download the tweets with twint


### Not urgent:
- [ ] Better pipeline
- [ ] Change nicknames to User names
- [ ] Use information of like and retweets
- [ ] Check the number of followers in order to filter out irrelevant tweets (only during the online phase)


### Extras:
- [ ] Check if a person has a wikipedia page or not.
- [ ] Correct the Mispellings

### Done


## Links
- Pipeline
  - https://github.com/JohnSnowLabs/spark-nlp/issues/7357
  - https://www.johnsnowlabs.com/understanding-the-power-of-transformers-a-guide-to-sentence-embeddings-in-spark-nlp/
- Models
  - https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment 