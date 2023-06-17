import string

class TextVectorizer:
    def __init__(self):
        self.vocab = []
        self.vocab_size = 0
        self.freq_dict = {}

    def build_vocab(self, pos_tweets, neg_tweets):
        for tweet in pos_tweets + neg_tweets:
            words = [w.translate(str.maketrans('', '', string.punctuation)) for w in tweet.split()]
            for word in words:
                if word not in self.vocab:
                    self.vocab.append(word)
        self.vocab_size = len(self.vocab)

    def generate_freq_dict(self, pos_tweets, neg_tweets):
        self.freq_dict = {}

        for word in self.vocab:
            pos_freq = 0
            neg_freq = 0
            for tweet in pos_tweets:
                words_in_tweet = tweet.translate(str.maketrans('', '', string.punctuation)).split()
                pos_freq += words_in_tweet.count(word)
            for tweet in neg_tweets:
                words_in_tweet = tweet.translate(str.maketrans('', '', string.punctuation)).split()
                neg_freq += words_in_tweet.count(word)
            self.freq_dict[word] = {'positive': pos_freq, 'negative': neg_freq}

    def count_neg_pos_words(self):
        pos_sum = sum(val['positive'] for val in self.freq_dict.values())
        neg_sum = sum(val['negative'] for val in self.freq_dict.values())
        return pos_sum, neg_sum

    def vectorize_text(self, text):
        vector = [0] * self.vocab_size
        words = text.split()
        for word in words:
            if word in self.vocab:
                index = self.vocab.index(word)
                vector[index] = 1
        return vector
    
    def extract_features(self, tweet):
        words = tweet.translate(str.maketrans('', '', string.punctuation)).split()
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        bias_unit = 1
        pos_freq_sum = sum(self.freq_dict[word]['positive'] for word in unique_words if word in self.freq_dict)
        neg_freq_sum = sum(self.freq_dict[word]['negative'] for word in unique_words if word in self.freq_dict)
        return [bias_unit, pos_freq_sum, neg_freq_sum]


# Example usage
def main():
    pos_tweets = ["I am happy because I am learning NLP", "I am happy"]
    neg_tweets = ["I am sad, I am not learning NLP", 'I am sad']
    text_vectorizer = TextVectorizer()
    text_vectorizer.build_vocab(pos_tweets,  neg_tweets)
    text_vectorizer.generate_freq_dict(pos_tweets, neg_tweets)
    text_vectorizer.count_neg_pos_words()

    text1 = "I am happy because I am learning NLP"
    text2 = 'I am sad, I am not learning NLP'
    vector3_t1 = text_vectorizer.extract_features(text1)
    vector3_t2 = text_vectorizer.extract_features(text2)

    print("Vocabulary:", text_vectorizer.vocab)
    print("Frequency Dictionary:", text_vectorizer.freq_dict)
    print('Text1 Vector3: ', vector3_t1, 'Text2 Vector3: ', vector3_t2 )

    if __name__ == '__main__': 
        main()