import nltk
import nltk.downloader
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import json
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
#import seaborn as sns
import networkx as nx
#import os

#NLTK download
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('vader_lexicon')


#load spacy model
nlp = spacy.load("en_core_web_sm")


# #read sentences from file
# with open('word_list2.txt', 'r') as file:
#     sentences = file.readlines()
# sentences = [sentence.strip() for sentence in sentences]


# tokenization
def tokenize(sentence):
    return word_tokenize(sentence)


# N-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Part-of-speech tagging
def pos_tag(sentence):
    return nltk.pos_tag(tokenize(sentence))

# Named Entity Recognition
def ner(sentence):
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Noun phrase extraction
def extract_noun_phrases(sentence):
    doc = nlp(sentence)
    return [chunk.text for chunk in doc.noun_chunks]


# Text summarization
def text_summarizer(sentence):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    return summarizer(sentence, max_length=100, min_length=15, do_sample=False)[0]['summary_text']


# Sentiment analysis
sia = SentimentIntensityAnalyzer()
def analyze_sentiment(sentence):
    return sia.polarity_scores(sentence)


# Topic modeling
def topic_modeling(sentences, num_topics=5, num_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(sentences)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)


    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-num_words -1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics




# Word frequency analysis
def word_frequency(sentence):
    tokens = tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return nltk.FreqDist(words)



#plot graphs for each major statistic
def plot_sentiment(sentiment):
    #plot sentiment graph
    labels = ['Negative', 'Neutral', 'Positive']
    sizes = [sentiment['neg'], sentiment['neu'], sentiment['pos']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution for Sentence')
    plt.axis('equal')
    plt.savefig('sentiment_example.png')
    plt.close()


#plot part of speech distro
def plot_pos_distribution(pos_tags):
    pos_counts = Counter(tag for _, tag in pos_tags)
    plt.bar(pos_counts.keys(), pos_counts.values())
    plt.title('Part-of-Speech Distribution')
    plt.xlabel('POS Tags')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.savefig('pos_example.png')
    plt.close()

#plot word cloud based on word frequency in input
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Input')
    plt.savefig('word_cloud_example.png')
    plt.close()
    
#plot noun phrases
def plot_noun_phrases(noun_phrases):
    phrase_lengths = [len(phrase.split()) for phrase in noun_phrases]
    length_counts = Counter(phrase_lengths)

    plt.figure(figsize=(10, 6))
    plt.bar(length_counts.keys(), length_counts.values())
    plt.title('Distribution of Noun Phrase Lengths')
    plt.xlabel('Number of Words in Noun Phrase')
    plt.ylabel('Frequency')
    plt.xticks(range(min(phrase_lengths), max(phrase_lengths) + 1))

    # sns.histplot(phrase_lengths, bins=range(1, max(phrase_lengths) + 2, 1), kde=True)
    # plt.title('Distribution of Noun Phrase Lengths')
    # plt.xlabel('Number of Words in Noun Phrase')
    # plt.ylabel('Frequency')
    plt.savefig('noun_phrase_plot.png')
    plt.close()

#plot bi-gram distribution
def plot_bigram_network(bigrams):
    G = nx.Graph()
    for bigram in bigrams:
        G.add_edge(bigrams[0], bigram[1])
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
    plt.title('Bigram Network')
    plt.axis('off')
    plt.savefig('bigram_example.png')
    plt.close()

#main processing
def process_sentences(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        sentences = infile.readlines()
        sentences = [sentence.strip() for sentence in sentences]

        #process each sentence
        for i, sentence in enumerate(sentences, 1):
            outfile.write(f"Sentence {i}:\n")
            outfile.write(f"sentence: {sentence}\n")
            summarized_sentence = text_summarizer(sentence)
            outfile.write(f"Summarized sentence: {summarized_sentence}\n")
            tokens = tokenize(sentence)
            outfile.write(f"tokens: {tokens}\n")
            bigrams = generate_ngrams(tokens, 2)
            outfile.write(f"bigrams: {bigrams}\n")
            pos_tags = pos_tag(sentence)
            outfile.write(f"pos_tags: {pos_tags}\n")
            outfile.write(f"named_entities: {ner(sentence)}\n")
            noun_phrases = extract_noun_phrases(sentence)
            outfile.write(f"noun_phrases: {noun_phrases}\n")
            sentiment  = analyze_sentiment(sentence)
            outfile.write(f"sentiment: {sentiment}\n")
            outfile.write(f"word_frequency: {word_frequency(sentence).most_common(3)}\n")
            outfile.write(f"Summarization Ratio: {len(summarized_sentence.split()) / len(sentence.split())}")
            outfile.write("\n\n")
            


            if i == 1:
                plot_sentiment(sentiment)
                plot_pos_distribution(pos_tags)
                create_word_cloud(' '.join(tokens))
                plot_noun_phrases(noun_phrases)
                plot_bigram_network(bigrams[:20])


        #process the sentences
        outfile.write("Overall Analysis:\n")
        #outfile.write(f"Summary: {summarize(sentences)}\n\n")
        outfile.write("Topic Modeling:\n")
        for topic in topic_modeling(sentences):
            outfile.write(f"{topic}\n")
         

# # Process each sentence
# for i, sentence in enumerate(sentences, 1):
#     print(f"\nSentence {i}: {sentence}")
#     print("Tokens:", tokenize(sentence))
#     print("Bigrams:", generate_ngrams(tokenize(sentence), 2))
#     print("POS Tags:", pos_tag(sentence))
#     print("Named Entities:", ner(sentence))
#     print("Noun Phrases:", extract_noun_phrases(sentence))
#     print("Sentiment:", analyze_sentiment(sentence))
#     print("Word Frequency:", word_frequency(sentence).most_common(3))


if __name__ == "__main__":
    input_file = 'paragraph.txt'
    output_file = "paragraph_nlp.txt"
    process_sentences(input_file, output_file)
    print(f"Analysis complete. Results written to {output_file}")