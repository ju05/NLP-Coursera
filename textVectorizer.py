class TextVectorizer:
    def __init__(self):
        self.vocab = []
        self.vocab_size = 0
        self.freq_dict = {}

    def build_vocab(self, pos_tweets, neg_tweets):
        for tweet in pos_tweets + neg_tweets:
            words = tweet.split()
            for word in words:
                if word not in self.vocab:
                    self.vocab.append(word)
        self.vocab_size = len(self.vocab)

    def generate_freq_dict(self, pos_tweets, neg_tweets):
        self.freq_dict = {}
        for word in self.vocab:
            pos_freq = sum(1 for tweet in pos_tweets if word in tweet.split())
            neg_freq = sum(1 for tweet in neg_tweets if word in tweet.split())
            self.freq_dict.update({'positive': pos_freq})
            self.freq_dict.update({'negative': neg_freq})

    def count_neg_pos_words(self):
        pos_sum = self.freq_dict['positive']
        neg_sum = self.freq_dict['negative']
        return pos_sum, neg_sum

    def vectorize_text(self, text):
        vector = [0] * self.vocab_size
        words = text.split()
        for word in words:
            if word in self.vocab:
                index = self.vocab.index(word)
                vector[index] = 1
        return vector


# Example usage
pos_tweets = ["I am happy", "I am excited", "I am coding"]
neg_tweets = ["I am sad", "I am angry"]
text_vectorizer = TextVectorizer()
text_vectorizer.build_vocab(pos_tweets,  neg_tweets)
text_vectorizer.generate_freq_dict(pos_tweets, neg_tweets)
text_vectorizer.count_neg_pos_words()

text = "I am coding"
vector = text_vectorizer.vectorize_text(text)

print("Vocabulary:", text_vectorizer.vocab)
print("Frequency Dictionary:", text_vectorizer.freq_dict)
print("Text Vector:", vector)