from fastapi import FastAPI, File, UploadFile
import gradio as gr
import pickle
import zipfile

import pandas as pd
import numpy as np
import re

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from summarizer import Summarizer,TransformerSummarizer

nltk.download('punkt')
nltk.download('stopwords')

model_checkpoint = "marefa-nlp/marefa-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

with zipfile.ZipFile("model.zip", 'r') as zip_ref:
    zip_ref.extractall("./marian_model/")

# Define the model architecture
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt=True)

# Load the weights from the .h5 file
model.load_weights("./marian_model/model.weights.h5")

# Load cleaned_word_embeddings
with open("cleaned_word_embeddings.pkl", "rb") as f:
    cleaned_word_embeddings = pickle.load(f)

summ_model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")

def translate_pretrained(text):
    summarized = ''.join(summ_model(text))
    tokenized = tokenizer([summarized], return_tensors="np")
    out = model.generate(**tokenized)
    arabic = tokenizer.decode(out[0], skip_special_tokens=True)
    return arabic

def get_clean_sentences(text):
    sentences = sent_tokenize(text)
    # Remove punctuations, numbers and special characters
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = re.sub(r"\\.|[^\\'\w ]", " ", sentence)
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences


def filter_sentences(text):
    cleaned_sentences = get_clean_sentences(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_sentences = []
    for sentence in cleaned_sentences:
        words = nltk.word_tokenize(sentence)
        filtered_sentence = " ".join(
            [word for word in words if word.lower() not in stop_words]
        )
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


def get_vector_representation(text):
    filtered_sentences = filter_sentences(text)
    # Get vector representations for each sentence in the articles
    sentence_vectors = []
    for sentence in filtered_sentences:
        words = sentence.split()
        sentence_vector = np.zeros((25,))
        if len(words) != 0:
            for word in words:
                if word in cleaned_word_embeddings:
                    sentence_vector += cleaned_word_embeddings[word]
            sentence_vector /= len(words)
        sentence_vectors.append(sentence_vector)
    return sentence_vectors


def calculate_cosine_similarity(sentence_vectors):
    flat_sentence_vectors = np.array(
        [vec for sublist in sentence_vectors for vec in sublist]
    ).reshape(1, -1)
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(sentence_vectors)
    return similarity_matrix


def get_scores(similarity_matrix):
    # Create a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    # Get scores
    scores = nx.pagerank(nx_graph)
    return scores


def rank_sentences(text):
    sentence_vectors = get_vector_representation(text)
    similarity_matrix = calculate_cosine_similarity(sentence_vectors)
    scores = get_scores(similarity_matrix)
    ranked_sentences = sorted(
        ((scores[j], sentence) for j, sentence in enumerate(sent_tokenize(text))),
        reverse=True,
    )
    return ranked_sentences


def summarize(text):
    ranked_sentences = rank_sentences(text)
    summary = ""
    for j in range(len(ranked_sentences)//10): 
        summary += ranked_sentences[j][1] + " "
    return summary

def translate(text):
    summarized = summarize(text)
    tokenized = tokenizer([summarized], return_tensors='np')
    out = model.generate(**tokenized)
    arabic = tokenizer.decode(out[0], skip_special_tokens=True)
    return arabic

demo = gr.Interface(fn=translate_pretrained, inputs="text", outputs="text")
demo.launch(share=True)

