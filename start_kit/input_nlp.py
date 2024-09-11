import nltk
#nltk.data.path.append('/home/uahoxa001/nltk_data')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# preprocess sentence using tokenizers, stop_words, and lemmatization
def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)



#using spacy to derive noun phrases for more contextual analysis
import spacy

nlp = spacy.load('en_core_web_sm')

def extract_key_components(sentence):
    doc = nlp(sentence)
    components = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and token.dep_ in ['ROOT', 'dobj', 'pobj', 'conj']:
            components.append(token.text)
    return components

#attempting to summarize the caption while keeping much of the context 
def summarize_sentence(sentence):
    components = extract_key_components(sentence)
    return ' '.join(components)
def extract_noun_phrases(sentence):
    doc = nlp(preprocess_sentence(sentence))
    return [chunk.text for chunk in doc.noun_chunks]

# from nltk import pos_tag, word_tokenize
# from nltk.chunk import RegexpParser
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# def extract_noun_phrases(sentence):
#     words = word_tokenize(sentence)
#     tagged = pos_tag(words)
#     chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"
#     chunk_parser = RegexpParser(chunk_grammar)
#     chunked = chunk_parser.parse(tagged)
#     return [' '.join(word for word, pos in subtree.leaves())
#             for subtree in chunked.subtrees()
#             if subtree.label() == 'NP']


#performing bayesian analysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def fit_bayesian_model(sentences, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    clf = MultinomialNB()
    clf.fit(X, labels)
    return vectorizer, clf

def perform_bayesian_analysis(sentence, vectorizer, clf):
    X = vectorizer.transform([preprocess_sentence(sentence)])
    return clf.predict_proba(X)[0]


#sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment_score(sentence):
    return sia.polarity_scores(preprocess_sentence(sentence))['compound']

#labels
sentences = ["This is a positive sentence.", "This is a negative sentence.", "This is a neutral sentence."]
labels = [1, 0, 2]  # 1 for positive, 0 for negative, 2 for neutral

vectorizer, clf = fit_bayesian_model(sentences, labels)

sentence = "I want to read a book and drink water using my computer"

noun_phrases = extract_noun_phrases(sentence)
bayesian = perform_bayesian_analysis(sentence, vectorizer, clf)
sentiment_score = get_sentiment_score(sentence)

print("Noun phrases:", noun_phrases)
print("Bayesian probabilities:", bayesian)
print("Sentiment score:", sentiment_score)