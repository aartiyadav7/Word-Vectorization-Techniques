Word Vectorization Techniques Assignment
This repository contains an assignment that demonstrates three core word vectorization techniques in Natural Language Processing (NLP): Count Vectorizer, TF-IDF Vectorizer, and Word2Vec. These techniques are used for transforming text into numerical representations, which is a crucial step in many NLP applications.

Overview
The assignment covers the following techniques:

Count Vectorizer: Tokenizes text and creates a matrix of word counts.
TF-IDF Vectorizer: Converts text into vectors based on word importance relative to the document corpus.
Word2Vec: A neural network-based method for learning word embeddings that capture the meaning of words based on context.
Table of Contents
Installation
Usage
Part A: Count Vectorizer
Part B: TF-IDF Vectorizer
Part C: Word2Vec
Conclusion
License
Installation
To run the code, you need to install the following dependencies:

bash
Copy
Edit
pip install scikit-learn gensim matplotlib
Usage
Run the Code
Clone this repository:
bash
Copy
Edit
git clone https://github.com/yourusername/word-vectorization-techniques.git
Navigate to the project directory:
bash
Copy
Edit
cd word-vectorization-techniques
Run each part of the assignment:
Part A: Count Vectorizer: Run the script that demonstrates word count vectorization.
Part B: TF-IDF Vectorizer: Run the script that demonstrates TF-IDF vectorization.
Part C: Word2Vec: Run the script that demonstrates Word2Vec and computes word similarities.
bash
Copy
Edit
python part_a.py
python part_b.py
python part_c.py
Example Outputs
Part A will output the word count matrix and feature names for the provided sentences.
Part B will output the TF-IDF matrix and feature names, and explain the importance of TF-IDF.
Part C will output the most similar words to "learning" and visualize word embeddings for selected words.
Part A: Count Vectorizer
This part demonstrates how to tokenize sentences and transform them into a word count matrix using the CountVectorizer from scikit-learn.

Key Concepts: Tokenization, word count matrix.
Output: The resulting matrix will show the frequency of each word in the sentences.
Part B: TF-IDF Vectorizer
This part demonstrates how to use TfidfVectorizer from scikit-learn to transform sentences into TF-IDF vectors.

Key Concepts: TF-IDF weighting, importance of words.
Output: The resulting matrix will show how important each word is within the context of the corpus.
Importance of TF-IDF:
TF-IDF is particularly useful for information retrieval, as it helps prioritize words that carry more meaning and are less frequent across the documents, unlike the CountVectorizer, which just counts word occurrences.
Part C: Word2Vec
This part demonstrates how to use Word2Vec from Gensim to train a model that generates word embeddings. These embeddings are more sophisticated representations of words that capture their meanings based on context.

Key Concepts: Word embeddings, Word2Vec, cosine similarity.
Output: The most similar words to "learning" will be displayed. Also, a visualization of word embeddings for selected words like "machine", "learning", and "AI" will be shown using PCA.
Cosine Similarity:
A function is included to compute the cosine similarity between two words (e.g., "AI" and "machine").
Conclusion
This assignment introduces fundamental techniques in word vectorization, including word frequency-based methods (CountVectorizer), statistical weighting (TF-IDF), and context-based embeddings (Word2Vec). Understanding these methods is crucial for working on more advanced NLP tasks such as sentiment analysis, document classification, and information retrieval.
