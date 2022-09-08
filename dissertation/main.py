# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press ⌘F8 to toggle the breakpoint.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
"""
Pre-processing.
"""
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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logging.getLogger('numexpr').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long, and removing
# punctuation and numbers.
def tokenize_documents(documents):
    for document in documents:
        yield simple_preprocess(str(document), deacc=True)  # deacc=True removes accent marks from tokens.

# Initialise list of stopwords.
# stop_words = stopwords.words('english')

#stop_words = list(_stop_words.ENGLISH_STOP_WORDS)

en = spacy.load('en_core_web_sm')
# stop_words = list(en.Defaults.stop_words)

nltk_stop_words = stopwords.words('english')
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
    num_topics = 10
    distributed = False
    chunksize = 200  # number of documents that are processed at a time in the training algorithm
    passes = 20  # epochs (set the number of “passes” high enough) - controls how often we train the model on the entire corpus
    update_every = 1
    alpha = 'symmetric'
    eta = 'symmetric'
    decay = 0.5
    offset = 1.0
    eval_every = 10
    iterations = 50  # set the number of “iterations” high enough - it controls how often we repeat a particular loop over each document
    gamma_threshold = 0.001
    minimum_probability = 0.01
    random_state = 100 #***
    ns_conf = None
    minimum_phi_value = 0.01
    per_word_topics = True #***
    dtype = np.float32

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
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
        dtype=np.float32
    )

    '''Step 9: View the topics in LDA model'''
    print('\nStep 9: View the topics in LDA mode')
    # Print the keywords in the topics
    pprint(lda_model.print_topics(num_topics=20, num_words=10))
    doc_lda = lda_model[corpus]

    '''Step 10: Compute Model Perplexity'''
    # Compute Perplexity
    # print('\nStep 10: Compute Model Perplexity')
    # print('Perplexity: ', lda_model.log_perplexity(corpus))  # A measure of how good the model is. Lower the better.

    '''Step 11: Compute Topic Coherence Score'''
    print('\nStep 11: Compute Topic Coherence Score')
    # Compute Coherence Score - “AKSW” topic coherence measure (http://rare-technologies.com/what-is-topic-coherence/)
    coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=list_lemmatized_articles,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence score (“AKSW” topic coherence measure): ', coherence_lda)

    # Compute Coherence Score - “Umass” topic coherence measure
    top_topics = lda_model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pprint(top_topics)

    # TODO: change name of html depending on the iteration for the model depending on k value
    # '''Step 12: Visualize the topics'''
    print('\nStep 12: Visualize the topics')
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda.html')