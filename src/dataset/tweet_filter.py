import re


class TweetFilter():
    def __init__(self):
        self.urls = []
        self.hashtags = []
        self.mentions = []

    def filter_urls(self, text):
        return re.sub(r"http\S+", "", text)

    def filter_hashtags(self, text):
        hashtags = re.findall(r'#(\w+)', text)
        hashtags = ", ".join([hashtag.lower() for hashtag in hashtags])
        
        text = re.sub(r'#(\w+)', lambda match: self.split_hastag(match.group(1)), text)
        

        return text, hashtags
    
    def filter_mentions(self, text):
        mentions = ", ".join(re.findall(r'@(\w+)', text))

        text = re.sub(r'@(\w+)', r' \1 ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Take the name of the user
        return text, mentions


    def split_hastag(self, hashtag):
        hashtag = re.sub(r'([A-Z]?[a-z]+|[0-9]+)', r' \1 ', hashtag)
        hashtag = re.sub(r'\s+', ' ', hashtag).strip()
        return hashtag
    

    def filter_tweet(self, tweet: str):
        tweet = tweet.strip("RT ")
        _ = self.filter_urls(tweet)
        _, hashtags = self.filter_hashtags(_)
        _, mentions = self.filter_mentions(_)
        return tweet, hashtags, mentions
        


if __name__ == "__main__":
    tweet_filter = TweetFilter()
    text = "This is a test with a #hashtagTest #Hashtag2dE #F1 and a @menti2on and @Peppe24 and a link https://www.ciao.it/fsfESdf?FSAdf^SDFsdfsdafS?"
    print(tweet_filter.filter_tweet(text))