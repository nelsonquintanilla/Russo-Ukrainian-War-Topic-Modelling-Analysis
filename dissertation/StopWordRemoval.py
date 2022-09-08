import spacy
import sklearn
from sklearn.feature_extraction import _stop_words
from pprint import pprint
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

if __name__ == '__main__':
    nltk_stop_words = stopwords.words('english')
    print("nltk stopwords: " + str(nltk_stop_words))
    print("nltk stopswords length: " + str(len(nltk_stop_words)))

    print('\n scikit learn stop words')
    sklearn_stopwords = list(_stop_words.ENGLISH_STOP_WORDS)
    print("sklearn stopwords: " + str(sklearn_stopwords))
    print("sklearn stopswords length: " + str(len(sklearn_stopwords)))

    print('\n spacy stop words')
    en = spacy.load('en_core_web_sm')
    spacy_stopwords = list(en.Defaults.stop_words)
    print("spacy stopwords: " + str(spacy_stopwords))
    print("spacy stopswords length: " + str(len(spacy_stopwords)))

    merged_stopwords = set(nltk_stop_words).union(sklearn_stopwords, spacy_stopwords)
    print("\nmerged stopwords: " + str(merged_stopwords))
    print("merged stopswords length: " + str(len(merged_stopwords)))

    scikit_minus_nltk = list(set(sklearn_stopwords) - set(nltk_stop_words))
    print("\nscikit_minus_nltk: " + str(scikit_minus_nltk))
    print("scikit_minus_nltk length: " + str(len(scikit_minus_nltk)))

    spacy_minus_nltk = list(set(spacy_stopwords) - set(nltk_stop_words))
    print("\nspacy_minus_nltk: " + str(spacy_minus_nltk))
    print("spacy_minus_nltk length: " + str(len(spacy_minus_nltk)))
