# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press ⌘F8 to toggle the breakpoint.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# Spacy
import spacy
spacy.cli.download('en_core_web_sm')

# NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis.gensim_models

# Misc
import numpy as np
from pprint import pprint

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Pre-processing
'''Step 0: Number of articles and average length per article'''


'''Step 1: Normalisation and tokenization'''
# Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long, and removing
# punctuation and numbers.
def tokenize_documents(documents):
    for document in documents:
        yield simple_preprocess(str(document), deacc=True)  # deacc=True removes accent marks from tokens.

'''Step 2: Stop Words Removal'''
# Remove stop words from each tokenized article.
stop_words = stopwords.words('english')

def remove_stopwords_single_article(data_words):
    return [word for word in data_words if word not in stop_words]

def remove_stopwords_many_articles(list_tokenized_documents):
    list_tokenized_documents_nostops = []
    for tokenized_article in list_tokenized_documents:
        tokenized_article_nostops = remove_stopwords_single_article(tokenized_article)
        list_tokenized_documents_nostops.append(tokenized_article_nostops)
    return list_tokenized_documents_nostops

'''Step 3: Lemmatisation'''
# Remove inflectional endings from tokens to return the base or dictionary form of a word.
# E.g., am, are, is -> be
nlp = spacy.load('en_core_web_sm')
nlp.get_pipe('lemmatizer')

def lemmatize_articles(list_articles):
  lemmatized = []
  for article in list_articles:
    doc = nlp(' '.join(article))
    lemmatized.append([token.lemma_ for token in doc])
  return lemmatized

'''Step 4: Bigrams and Trigrams'''
# docs = list_lemmatized_articles
# print(docs)
# new_sentence = ['trees', 'graph', 'minors']
# bigram = Phrases(docs, min_count=2, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
# for idx in range(len(docs)):
#   for token in bigram[docs[idx]]:
#     if '_' in token:
#       print("is a bigram")
#       # Token is a bigram, add to document.
#       docs[idx].append(token)
# print(docs)

'''Step 5: Removal of rare words and common words based on their document frequency'''


'''Step 6: Create the Dictionary and Corpus needed for Topic Modeling (transform the documents to a vectorized form)'''
def create_dictionary(list_lemmatized_documents):
    return corpora.Dictionary(list_lemmatized_documents) # Combine all elements of array before passing it here

# Bag-of-words representation of the documents. Term Document Frequency.
def create_corpus(list_lemmatized_documents, dict_param):
    docs = list_lemmatized_documents
    return [dict_param.doc2bow(doc) for doc in docs]

# Human-readable format of corpus (term-frequency)
def generate_readable_corpus(dict_param, corpus_param):
    return [[(dict_param[term_id], term_freq) for term_id, term_freq in cp] for cp in corpus_param]

'''Step 7: Number of tokens and documents to train'''

'''Step 8: Building the Topic Model'''

'''Step 9: View the topics in LDA model'''

'''Step 10: Compute Model Perplexity and Topic Coherence Score'''

'''Step 11: Visualize the topics'''

if __name__ == '__main__':
    articles = [
        'Ukraine’s president of the united states has made a desperate appeal to the Russian people of the united '
        'states, and the president of the united states.',
        'For many Russians of the united states, it was an unfamiliar sight to see the faces of the two leaders in an '
        'unfamiliar sight.',
        'The striped bats were hanging on their feet for best corpora alumni and ate best fishes COVID-19 am are is '
        'were was children that were striped with bands of sunlight and stripped from their things.']

    print('number of articles:' + str(len(articles)))

    # Tokenize articles
    list_tokenized_articles = list(tokenize_documents(articles))
    print("\nlist_tokenized_articles: " + str(list_tokenized_articles))
    print("first element length: " + str(len(list_tokenized_articles[0])))

    # Stop words from
    print("\nstop_words: " + str(stop_words))
    print("stops words length: " + str(len(stop_words)))

    # Remove stop words from tokenized articles
    list_tokenized_articles_nostops = remove_stopwords_many_articles(list_tokenized_articles)
    print("\nlist_tokenized_articles_nostops: " + str(list_tokenized_articles_nostops))
    print("first element nostops length: " + str(len(list_tokenized_articles_nostops[0])))

    # Lemmatize articles
    list_lemmatized_articles = lemmatize_articles(list_tokenized_articles_nostops)
    print("\nlist_lemmatized_articles: " + str(list_lemmatized_articles))
    print("first element length: " + str(len(list_lemmatized_articles[0])))

    # Create dictionary and corpus
    dictionary = create_dictionary(list_lemmatized_articles)
    corpus = create_corpus(list_lemmatized_articles, dictionary)
    generate_readable_corpus(dictionary, corpus)
    print("\ndictionary: " + str(dictionary))
    print("corpus:" + str(corpus))
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Tune lda params
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        num_topics=5,
        id2word=dictionary,
        distributed=False,
        chunksize=10,  # number of documents that are processed at a time in the training algorithm
        passes=10,  ########## epochs, set the number of “passes” high enough.
        update_every=1,
        alpha='auto',
        eta='auto',
        decay=0.5,
        offset=1.0,
        eval_every=1,
        iterations=50,  ########## set the number of “iterations” high enough.
        gamma_threshold=0.001,
        minimum_probability=0.01,
        random_state=100,
        ns_conf=None,
        minimum_phi_value=0.01,
        per_word_topics=True,
        dtype=np.float32
    )

    # Print the Keyword in the topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # A measure of how good the model is. Lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=list_lemmatized_articles,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence score: ', coherence_lda)

    # Visualize the topics
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda.html')
