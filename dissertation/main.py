# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press ⌘F8 to toggle the breakpoint.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
"""
Pre-processing.
"""
import importlib.util as util
from AssembleDatasets import read_list_of_dicts_from_file
from TheGuardianRepository import get_list_articles_from_list_of_dicts, get_the_guardian_articles_list
# Spacy
import spacy
# spacy.cli.download('en_core_web_sm')
# NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
# Scikit-Learn
import sklearn
from sklearn.feature_extraction import _stop_words
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# pyLDAvis
import pyLDAvis.gensim_models
# Misc
import numpy as np
from pprint import pprint
# Enable logging for gensim - optional
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
# logging.getLogger('numexpr').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long, and removing
# punctuation and numbers.
def tokenize_documents(documents):
    for document in documents:
        yield simple_preprocess(str(document), deacc=True)  # deacc=True removes accent marks from tokens.

# Initialise list of stopwords.
# stop_words = stopwords.words('english')

# stop_words = list(_stop_words.ENGLISH_STOP_WORDS)

en = spacy.load('en_core_web_sm')
# stop_words = list(en.Defaults.stop_words)

custom_stopwords = ['ukraine','ukrainian', 'russia', 'russian']

nltk_stop_words = stopwords.words('english')
nltk_stop_words.extend(custom_stopwords)
sklearn_stopwords = list(_stop_words.ENGLISH_STOP_WORDS)
spacy_stopwords = list(en.Defaults.stop_words)
stop_words = set(nltk_stop_words).union(sklearn_stopwords, spacy_stopwords)

# Remove stop words from each tokenized article.
def remove_stopwords_single_article(data_words):
    return [word for word in data_words if word not in stop_words]

# Remove stop words from each tokenized article.
def remove_stopwords_many_articles(list_tokenized_documents):
    list_tokenized_documents_nostops = []
    for tokenized_article in list_tokenized_documents:
        tokenized_article_nostops = remove_stopwords_single_article(tokenized_article)
        list_tokenized_documents_nostops.append(tokenized_article_nostops)
    return list_tokenized_documents_nostops

# Remove inflectional endings from tokens to return the base or dictionary form of a word.
# E.g., am, are, is -> be
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatize_articles(list_articles):
    lemmatized = []
    for article in list_articles:
        doc = nlp(' '.join(article))
        lemmatized.append([token.lemma_ for token in doc])
    return lemmatized

def create_dictionary(list_lemmatized_documents):
    return corpora.Dictionary(list_lemmatized_documents)  # Combine all elements of array before passing it here

# Bag-of-words representation of the documents. Term Document Frequency.
def create_corpus(list_lemmatized_documents, dict_param):
    docs = list_lemmatized_documents
    return [dict_param.doc2bow(doc) for doc in docs]

# Human-readable format of corpus (term-frequency)
def generate_readable_corpus(dict_param, corpus_param):
    return [[(dict_param[term_id], term_freq) for term_id, term_freq in cp] for cp in corpus_param]

# Calculate average number of words from the corpus
def compute_average_document_length(list_documents):
    sum_words = 0
    for document in list_documents:
        sum_words = sum_words + len(document.split())
    mean = sum_words/len(list_documents)
    return mean

U_MASS = 'u_mass'
C_V = 'c_v'

def train_lda_models(
        num_topics_list_,
        corpus_,
        dictionary_
):
    lda_models_list_ = []
    k_list_ = []
    for current_num_topics in num_topics_list_:
        k_list_.append(current_num_topics)
        print('\nTraining model for ' + str(current_num_topics))
        current_lda_model = train_single_model(corpus_=corpus_, id2word_=dictionary_, num_topics_=current_num_topics)
        lda_models_list_.append(current_lda_model)
    return lda_models_list_, k_list_

# mallet_path = '/Users/nelsonquintanilla/Documents/repos/nfq160/dissertation/mallet-2.0.8/bin/mallet'
# lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

def train_single_model(
        corpus_,
        id2word_,
        num_topics_ = 10,
        # distributed = False,
        chunksize = 2000, # number of documents that are processed at a time in the training algorithm
        passes = 20, # epochs (set the number of “passes” high enough) - controls how often we train the model on the entire corpus
        # update_every = 0,
        alpha = 'symmetric',
        eta = 'symmetric',
        decay = 0.5,
        offset = 1.0,
        eval_every = 0,
        iterations = 400, # set the number of “iterations” high enough - it controls how often we repeat a particular loop over each document
        gamma_threshold = 0.001,
        minimum_probability = 0.01,
        random_state = 100,
        # ns_conf = None,
        minimum_phi_value = 0.01,
        per_word_topics = True,
        dtype = np.float32
):
    lda_model_ = gensim.models.ldamulticore.LdaMulticore(
        corpus=corpus_,
        num_topics=num_topics_,
        id2word=id2word_,
        workers=None,
        # distributed=distributed,
        chunksize=chunksize,
        passes=passes,
        # update_every=update_every,
        alpha=alpha,
        eta=eta,
        decay=decay,
        offset=offset,
        eval_every=eval_every,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        minimum_probability=minimum_probability,
        random_state=random_state,
        # ns_conf=ns_conf,
        minimum_phi_value=minimum_phi_value,
        per_word_topics=per_word_topics,
        dtype=dtype
    )
    return lda_model_

def compute_perplexity_values(lda_models_list_, corpus_):
    perplexity_list = []
    for current_model in lda_models_list_:
        perplexity_list.append(current_model.log_perplexity(corpus_))
    return perplexity_list

def compute_u_mass_coherence_values(lda_models_list_, corpus_, dictionary_, coherence_):
    u_mas_coherence_list = []
    for current_model in lda_models_list_:
        u_mas_coherence_list.append(CoherenceModel(
            model=current_model,
            corpus=corpus_,
            dictionary=dictionary_,
            coherence=coherence_
        ).get_coherence())
    return u_mas_coherence_list

def compute_c_v_coherence_values(lda_models_list_, texts_, dictionary_, coherence_='c_v'):
    c_v_coherence_list = []
    for current_model in lda_models_list_:
        c_v_coherence_list.append(CoherenceModel(
            model=current_model,
            texts=texts_,
            dictionary=dictionary_,
            coherence=coherence_
        ).get_coherence())
    return c_v_coherence_list

def concatenate_models_values(list1, list2, list3, list4):
    new_list=[]
    for index, (item1, item2, item3, item4) in enumerate(zip(list1, list2, list3, list4)):
        new_list.append((item1, item2, item3, item4))
    return new_list

# def concatenate_models_values(list1, list2, list3):
#     new_list=[]
#     for index, (item1, item2, item3) in enumerate(zip(list1, list2, list3)):
#         new_list.append((item1, item2, item3))
#     return new_list

GENERAL_FILE_NAME = 'lda_4_k_'
FILE_EXTENSION = '.html'

def generate_pyldavis_html_files(lda_models_list_, k_list_, corpus_, dictionary_):
    for index, (current_lda_model, k) in enumerate(zip(lda_models_list_, k_list_)):
        vis_data = pyLDAvis.gensim_models.prepare(current_lda_model, corpus_, dictionary_, mds='mmds')
        file_name = GENERAL_FILE_NAME + str(k) + FILE_EXTENSION
        pyLDAvis.save_html(vis_data, file_name)

def generate_num_topics_list(start_ = 2, limit_ = 14, step_ = 2):
    return range(start_, limit_, step_)

def view_topics_models_list(lda_models_list_, num_topics_list_, num_words_=10):
    for index, (current_lda_model, current_num_topics) in enumerate(zip(lda_models_list_, num_topics_list_)):
        print('\nTopics for a model trained for ' + str(current_num_topics) + ' number of topics')
        pprint(current_lda_model.print_topics(num_topics=current_num_topics, num_words=num_words_))

def length_sum_lists(articles_list):
    acc = 0
    for article in articles_list:
        acc = acc + len(article)
    return acc

if __name__ == '__main__':
    articles_list_of_dicts = read_list_of_dicts_from_file('2DatasetsMerged')
    articles = get_list_articles_from_list_of_dicts(articles_list_of_dicts)

    '''Number of articles and average article length'''
    print('\nData before pre-processing')
    average_article_length = compute_average_document_length(articles)
    print('Number of articles: %d' % len(articles))
    print('Average word count per article: %d words' % round(average_article_length, 2))

    '''Normalisation and tokenization'''
    print('\nNormalisation and tokenization')
    list_tokenized_articles = list(tokenize_documents(articles))
    # print('list_tokenized_articles: ' + str(list_tokenized_articles))

    '''Stop Words Removal 1'''
    print('\nStop Words Removal 1')
    # print("\nstop_words: " + str(stop_words))
    # print("stops words length: " + str(len(stop_words)))

    # Remove stop words from tokenized articles
    list_tokenized_articles_nostops = remove_stopwords_many_articles(list_tokenized_articles)
    print('total of words, stop words removal 1: %d' %length_sum_lists(list_tokenized_articles_nostops))
    # print('list_tokenized_articles_nostops: ' + str(list_tokenized_articles_nostops))

    '''Lemmatisation'''
    print('\nLemmatisation')
    list_lemmatized_articles = lemmatize_articles(list_tokenized_articles_nostops)
    print('total of words, lemmatisation: %d' %length_sum_lists(list_lemmatized_articles))
    # print('list_lemmatized_articles: ' + str(list_lemmatized_articles))

    '''Stop Words Removal 2'''
    print('\nStop Words Removal 2')
    list_lemmatized_articles_nostops = remove_stopwords_many_articles(list_lemmatized_articles)
    print('total of words, stop words removal 2: %d' % length_sum_lists(list_lemmatized_articles_nostops))
    # print('list_tokenized_articles_nostops_2: ' + str(list_tokenized_articles_nostops_2))

    '''Transform the documents to a vectorized form (dictionary and corpus)'''
    print('\nTransform the documents to a vectorized form')
    dictionary = create_dictionary(list_lemmatized_articles_nostops)
    # dictionary.filter_extremes(no_below=50, no_above=0.60)
    corpus = create_corpus(list_lemmatized_articles_nostops, dictionary)
    readable_corpus = generate_readable_corpus(dictionary, corpus)
    # print('dictionary: ' + str(dictionary.token2id))
    # print('corpus:' + str(corpus))
    # print('readable corpus:' + str(readable_corpus))

    '''Number of tokens and documents to train'''
    print('\nNumber of tokens and documents to train')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    '''Building the Topic Model'''
    print('\nBuilding the Topic Model')
    # Tune lda params
    range_topics_list = generate_num_topics_list(start_=8, limit_=9, step_=1)
    lda_models_list, k_list = train_lda_models(
        num_topics_list_=range_topics_list,
        corpus_=corpus,
        dictionary_=dictionary
    )

    '''View the topics in LDA model'''
    print('\nView the topics in LDA mode')
    view_topics_models_list(lda_models_list_= lda_models_list, num_topics_list_=k_list, num_words_=10)

    '''Compute Perplexity'''
    print('\nCompute Perplexity')
    # a measure of how good the model is. lower the better.
    perplexity_values = compute_perplexity_values(
        lda_models_list_=lda_models_list,
        corpus_=corpus
    )

    '''Compute Topic Coherence Score'''
    print('\nCompute Topic Coherence Score')
    # u_mas_coherence_values = compute_u_mass_coherence_values(
    #     lda_models_list_=lda_models_list,
    #     corpus_=corpus,
    #     dictionary_=dictionary,
    #     coherence_=U_MASS
    # )
    c_v_coherence_values = compute_c_v_coherence_values(
        lda_models_list_=lda_models_list,
        texts_=list_lemmatized_articles_nostops,
        dictionary_=dictionary,
        coherence_=C_V
    )

    lda_models_list_values = concatenate_models_values(
        list1=[(index + 1) for index, _ in enumerate(k_list)],
        list2=k_list,
        list3=c_v_coherence_values,
        list4=perplexity_values
    )
    # print('(list_index, number_topics, c_v_coherence, u_mass_coherence_)')
    print('(list_index, number_topics, c_v_coherence, perplexity)')
    pprint(lda_models_list_values)

    '''Visualize the topics'''
    print('\nVisualize the topics')
    generate_pyldavis_html_files(
        lda_models_list_=lda_models_list,
        k_list_=k_list,
        corpus_=corpus,
        dictionary_=dictionary
    )
