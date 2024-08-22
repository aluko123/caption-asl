from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#Load your data
data = {}
with open('video_gloss_mapping.txt', 'r') as f:
    for line in f:
        video_id, gloss = line.strip().split(': ')
        data[video_id] = gloss


# Word Frequency Analysis
gloss_counts = Counter(data.values())
print("Top 10 most common signs:")
for gloss, count in gloss_counts.most_common(10):
    print(f"{gloss}: {count}")


# Load pre-trained word vectors
word_vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)



# Word Embeddings and Semantic Clustering
glosses = list(set(data.values()))
vectors = []
for gloss in glosses:
    if gloss in word_vectors:
        vectors.append(word_vectors[gloss])
    
if vectors:

    #convert to numpy array
    vectors = np.array(vectors)
    
    #K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(vectors)

    #Print cluster centers and their nearest glosses
    for i, center in enumerate(kmeans.cluster_centers_):
        closest_glosses = word_vectors.similar_by_vector(center, topn=5)
        print(f"Cluster {i}: {[word for word, _ in closest_glosses]}")

    
    #gloss_to_video
    gloss_to_video = {video_id: gloss for video_id, gloss in data.items()}

    #gloss clusters
    glosses = list(set(data.values()))
    gloss_clusters = [[] for _ in range(kmeans.n_clusters)]
    for gloss, label in zip(glosses, kmeans.labels_):
        gloss_clusters[label].append(gloss)

    #Visualize clusters
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=kmeans.labels_)
    plt.colorbar(scatter)
    #print("Shape of reduced_vectors:", reduced_vectors.shape)
    for i, gloss in enumerate(glosses):
        if i < reduced_vectors.shape[0]:
            plt.annotate(gloss, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.title("Semantic Clustering of Signs")
    plt.savefig('semantic_clusters.png')
    plt.close()


# Similarity Analysis
print("\nMost similar signs:")
for gloss in ['book', 'drink', 'computer']:
    if gloss in word_vectors:
        similar = word_vectors.most_similar(gloss, topn=5)
        print(f"Signs similar to '{gloss}': {[word for word, _ in similar]}")