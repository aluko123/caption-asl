import nltk
nltk.data.path.append('/home/uahoxa001/nltk_data')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# preprocess sentence using tokenizers, stop_words, and lemmatization
def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)



# #using spacy to derive noun phrases for more contextual analysis
# import spacy

# nlp = spacy.load('en_core_web_sm')

# def extract_noun_phrases(sentence):
#     doc = nlp(preprocess_sentence(sentence))
#     return [chunk.text for chunk in doc.noun_chunks]

from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_noun_phrases(sentence):
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = RegexpParser(chunk_grammar)
    chunked = chunk_parser.parse(tagged)
    return [' '.join(word for word, pos in subtree.leaves())
            for subtree in chunked.subtrees()
            if subtree.label() == 'NP']


#performing bayesian analysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
clf = MultinomialNB()
def perform_bayesian_analysis(sentence, clf):
    X = vectorizer.transform([preprocess_sentence(sentence)])
    return clf.predict_proba(X)[0]


#sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment_score(sentence):
    return sia.polarity_scores(preprocess_sentence(sentence))['compound']

sentence = "I want to read a book and drink water using my computer"

noun_phrases = extract_noun_phrases(sentence)
bayesian = perform_bayesian_analysis(sentence, clf)
sentiment_score = get_sentiment_score(sentence)
