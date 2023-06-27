import re
import nltk
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('twitter_samples')

class TweetPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, tweet):
        # Remove URLs
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#','', tweet)
        # Tokenize into words
        tokens = word_tokenize(tweet)
        # Lowercase conversion and stemming, Stopwords removal
        tokens = [self.stemmer.stem(word.lower()) for word in tokens if word.lower() not in self.stop_words]
        return tokens
    
preprocessor = TweetPreprocessor()
tweet = "@YMourri and @AndrewYNg are tuning a GREAT AI model at https://deeplearning.ai!!!"
print(preprocessor.preprocess(tweet))