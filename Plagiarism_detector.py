import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Collect source texts

student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read()
                 for _file in student_files]


# Step 2: Tokenize the text

def vectorize(Text): 
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(Text).toarray()
    return vectors


# Step 3: Compute similarities

def similarity(doc1, doc2): 
    
    # Cosine Similarity
    cos_sim = cosine_similarity([doc1, doc2])
    
    # Jaccard Similarity
    # Split the documents and create tokens
    doc1_tokens=set(doc1)
    doc2_tokens=set(doc2)
    jac_sim =  len(doc1_tokens.intersection(doc2_tokens))/len(doc1_tokens.union(doc2_tokens))
    
    similarity = (cos_sim + jac_sim) / 2
    return similarity


# Step 4: Define threshold

threshold = 0.9


# Step 5: Check Plagiarism 

vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
            if((sim_score > threshold)):
                print(student_pair[0], " and ",student_pair[1], "are plagiarised\n")
    
    return plagiarism_results


# Step 6: Print the Similarity Table

for data in check_plagiarism():
    print(data)

