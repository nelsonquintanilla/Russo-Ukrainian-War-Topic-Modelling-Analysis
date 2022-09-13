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
stop_words = stopwords.words('english')

#stop_words = list(_stop_words.ENGLISH_STOP_WORDS)

# en = spacy.load('en_core_web_sm')
# stop_words = list(en.Defaults.stop_words)

# nltk_stop_words = stopwords.words('english')
# sklearn_stopwords = list(_stop_words.ENGLISH_STOP_WORDS)
# spacy_stopwords = list(en.Defaults.stop_words)
# stop_words = set(nltk_stop_words).union(sklearn_stopwords, spacy_stopwords)

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

# def remove_stopwords_many_articles(texts):
#     return [[word for word in doc if word not in stop_words] for doc in texts]

# Remove inflectional endings from tokens to return the base or dictionary form of a word.
# E.g., am, are, is -> be
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatize_articles(list_articles):
    lemmatized = []
    for article in list_articles:
        doc = nlp(' '.join(article))
        lemmatized.append([token.lemma_ for token in doc])
    return lemmatized

# TODO
# Create bigrams.
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
    # u_mas_coherence_list = []
    # c_v_coherence_list = []
    for current_num_topics in num_topics_list_:
        k_list_.append(current_num_topics)
        print('\nTraining model for ' + str(current_num_topics))
        current_lda_model = train_single_model(corpus_=corpus_, id2word_=dictionary_, num_topics_=current_num_topics)
        lda_models_list_.append(current_lda_model)
        # u_mas_coherence_list.append(compute_u_mass_coherence(
        #     model_=current_lda_model,
        #     corpus_=corpus_,
        #     dictionary_=dictionary_,
        #     coherence_=U_MASS
        # ))
        # c_v_coherence_list.append(compute_c_v_coherence(
        #     model_=current_lda_model,
        #     texts_=list_lemmatized_articles_,
        #     dictionary_=dictionary_,
        #     coherence_=C_V
        # ))
    # view_topics_models_list(lda_models_list_= lda_models_list_, k_list_=k_list, num_words_=10)
    # generate_pyldavis_html_files(
    #     lda_models_list_=lda_models_list_,
    #     k_list_=k_list,
    #     corpus_=corpus_,
    #     dictionary_=dictionary_)
    # return concatenate_models_values(index_list, k_list, u_mas_coherence_list, c_v_coherence_list)
    return lda_models_list_, k_list_

def train_single_model(
        corpus_,
        id2word_,
        num_topics_ = 10,
        distributed = False,
        chunksize = 2000, # number of documents that are processed at a time in the training algorithm
        passes = 20, # epochs (set the number of “passes” high enough) - controls how often we train the model on the entire corpus
        update_every = 0,
        alpha = 'symmetric',
        eta = 'symmetric',
        decay = 0.5,
        offset = 1.0,
        eval_every = 1,
        iterations = 400, # set the number of “iterations” high enough - it controls how often we repeat a particular loop over each document
        gamma_threshold = 0.001,
        minimum_probability = 0.01,
        random_state = 100,
        ns_conf = None,
        minimum_phi_value = 0.01,
        per_word_topics = True,
        dtype = np.float32
):
    lda_model_ = gensim.models.ldamodel.LdaModel(
        corpus=corpus_,
        num_topics=num_topics_,
        id2word=id2word_,
        distributed=distributed,
        chunksize=chunksize,
        passes=passes,
        update_every=update_every,
        alpha=alpha,
        eta=eta,
        decay=decay,
        offset=offset,
        eval_every=eval_every,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        minimum_probability=minimum_probability,
        random_state=random_state,
        ns_conf=ns_conf,
        minimum_phi_value=minimum_phi_value,
        per_word_topics=per_word_topics,
        dtype=dtype
    )
    return lda_model_

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

GENERAL_FILE_NAME = 'lda_k_'
FILE_EXTENSION = '.html'

def generate_pyldavis_html_files(lda_models_list_, k_list_, corpus_, dictionary_):
    for index, (current_lda_model, k) in enumerate(zip(lda_models_list_, k_list_)):
        vis_data = pyLDAvis.gensim_models.prepare(current_lda_model, corpus_, dictionary_, mds='mmds')
        file_name = GENERAL_FILE_NAME + str(k) + FILE_EXTENSION
        pyLDAvis.save_html(vis_data, file_name)

def generate_num_topics_list(start_ = 2, limit_ = 22, step_ = 2):
    return range(start_, limit_, step_)

def view_topics_models_list(lda_models_list_, num_topics_list_, num_words_=10):
    for index, (current_lda_model, current_num_topics) in enumerate(zip(lda_models_list_, num_topics_list_)):
        print('\nTopics for a model trained for ' + str(current_num_topics) + ' number of topics')
        pprint(current_lda_model.print_topics(num_topics=current_num_topics, num_words=num_words_))

if __name__ == '__main__':
    # articles = [
    #     'Ukraine’s president of the United States has made 33 desperate appeals to the Russian people of the united '
    #     'states, and the president of the United States.',
    #     'For many Russians of the United States, it was an unfamiliar sight to see the faces of the two leaders in an '
    #     'unfamiliar sight.',
    #     'The striped bats were hanging on their feet for best corpora alumni and ate best fishes COVID-19 am are is '
    #     'were was children that were striped with bands of sunlight and stripped from their things.',
    #     'First of all, the elephant in the room: how many topics do I need? There is really no easy answer for this, '
    #     'it will depend on both your data and your application. I have used 10 topics here because I wanted to have a '
    #     'few topics that I could interpret and “label”, and because that turned out to give me reasonably good '
    #     'results. You might not need to interpret all your topics, so you wouldn’t use a large number of topics, '
    #     'for example 100.'
    # ]

    # articles_list_of_dicts = get_the_guardian_articles_list(
    #     number_of_articles=2,
    #     q="ukraine",
    #     section="world",
    #     from_date="2022-02-24",
    #     to_date="2022-08-31",
    #     show_blocks="body",
    #     page_size=2,
    #     order_by="oldest"
    # )
    # articles = get_list_articles_from_list_of_dicts(articles_list_of_dicts)
    # print(str(articles))

    articles_list_of_dicts = read_list_of_dicts_from_file('2DatasetsMerged')
    articles = get_list_articles_from_list_of_dicts(articles_list_of_dicts)

    '''Step 0: Number of articles and average article length'''
    print('\nStep 0: Data before pre-processing')
    average_article_length = compute_average_document_length(articles)
    print('Number of articles: %d' % len(articles))
    print('Average word count per article: %d words' % round(average_article_length, 2))

    '''Step 1: Normalisation and tokenization'''
    print('\nStep 1: Normalisation and tokenization')
    list_tokenized_articles = list(tokenize_documents(articles))
    # print('list_tokenized_articles: ' + str(list_tokenized_articles))
    # print('first element length: %d' %len(list_tokenized_articles[0])) # remove this line

    '''Step 2: Stop Words Removal'''
    print('\nStep 2: Stop Words Removal')
    # print("\nstop_words: " + str(stop_words))
    # print("stops words length: " + str(len(stop_words)))

    # Remove stop words from tokenized articles
    list_tokenized_articles_nostops = remove_stopwords_many_articles(list_tokenized_articles)
    # print('list_tokenized_articles_nostops: ' + str(list_tokenized_articles_nostops))
    # print("first element nostops length: " + str(len(list_tokenized_articles_nostops[0])))

    '''Step 3: Lemmatisation'''
    print('\nStep 3: Lemmatisation')
    list_lemmatized_articles = lemmatize_articles(list_tokenized_articles_nostops)
    # print('list_lemmatized_articles: ' + str(list_lemmatized_articles))
    # print("first element length: " + str(len(list_lemmatized_articles[0])))

    '''Step 4: Bigrams and Trigrams'''

    '''Step 4.5: Part of Speech Tag'''

    '''Step 5: Removal of rare words and common words based on their document frequency'''

    '''Step 6: Transform the documents to a vectorized form (dictionary and corpus)'''
    print('\nStep 6: Transform the documents to a vectorized form')
    dictionary = create_dictionary(list_lemmatized_articles)
    corpus = create_corpus(list_lemmatized_articles, dictionary)
    readable_corpus = generate_readable_corpus(dictionary, corpus)
    # print('dictionary: ' + str(dictionary.token2id))
    # print('corpus:' + str(corpus))
    # print('readable corpus:' + str(readable_corpus))

    '''Step 7: Number of tokens and documents to train'''
    print('\nStep 7: Number of tokens and documents to train')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    '''Step 8: Building the Topic Model'''
    print('\nStep 8: Building the Topic Model')
    # Tune lda params
    range_topics_list = generate_num_topics_list(start_=5, limit_=30, step_=3)
    lda_models_list, k_list = train_lda_models(
        num_topics_list_=range_topics_list,
        corpus_=corpus,
        dictionary_=dictionary
    )
    # lda_model = lda_models[0]
    # num_topics = 10
    # lda_model = train_single_model(corpus_=corpus, id2word_=dictionary, num_topics_=num_topics)

    '''Step 9: View the topics in LDA model'''
    print('\nStep 9: View the topics in LDA mode')
    # Print the keywords in the topics
    # pprint(lda_model.print_topics(num_topics=num_topics, num_words=10))
    view_topics_models_list(lda_models_list_= lda_models_list, num_topics_list_=k_list, num_words_=10)

    # '''Step 10: Compute Model Perplexity'''
    # Compute Perplexity
    # print('\nStep 10: Compute Model Perplexity')
    # print('Perplexity: ', lda_model.log_perplexity(corpus))  # A measure of how good the model is. Lower the better.

    '''Step 11: Compute Topic Coherence Score'''
    print('\nStep 11: Compute Topic Coherence Score')
    # # Compute Coherence Score - “AKSW” topic coherence measure (http://rare-technologies.com/what-is-topic-coherence/)
    # c_v_coherence_model_lda = CoherenceModel(
    #     model=lda_model,
    #     texts=list_lemmatized_articles,
    #     dictionary=dictionary,
    #     coherence='c_v'
    # )
    # u_mass_coherence_model_lda = CoherenceModel(
    #     model=lda_model,
    #     corpus=corpus,
    #     dictionary=dictionary,
    #     coherence='u_mass'
    # )
    # u_mass_coherence_lda = u_mass_coherence_model_lda.get_coherence()
    # print('U_Mass Coherence score (“AKSW” topic coherence measure): ', u_mass_coherence_lda)
    # c_v_coherence_lda = c_v_coherence_model_lda.get_coherence()
    # print('C_V Coherence score (“AKSW” topic coherence measure): ', c_v_coherence_lda)

    # # Compute Average Coherence Score - “Umass” topic coherence measure
    # top_topics = lda_model.top_topics(corpus)
    # pprint(top_topics)
    # # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    # print('Average topic coherence: %.4f.' % avg_topic_coherence)
    u_mas_coherence_values = compute_u_mass_coherence_values(
        lda_models_list_=lda_models_list,
        corpus_=corpus,
        dictionary_=dictionary,
        coherence_=U_MASS
    )
    c_v_coherence_values = compute_c_v_coherence_values(
        lda_models_list_=lda_models_list,
        texts_=list_lemmatized_articles,
        dictionary_=dictionary,
        coherence_=C_V
    )

    lda_models_list_values = concatenate_models_values(
        list1=[(index + 1) for index, _ in enumerate(k_list)],
        list2=k_list,
        list3=u_mas_coherence_values,
        list4=c_v_coherence_values
    )
    print('(list_index, number_topics, u_mass_coherence_, c_v_coherence)')
    pprint(lda_models_list_values)

    '''Step 12: Visualize the topics'''
    print('\nStep 12: Visualize the topics')
    # vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    # pyLDAvis.save_html(vis_data, 'lda.html')
    generate_pyldavis_html_files(
        lda_models_list_=lda_models_list,
        k_list_=k_list,
        corpus_=corpus,
        dictionary_=dictionary
    )
