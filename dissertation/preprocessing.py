# Imports, installments, and downloads for pre-processing of the text + implementation
# of the LDA model with Gensim + visualisation of the results

# Spacy
import spacy
spacy.cli.download("en_core_web_sm")

# NLTK
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
# plt.interactive(True)
# plt.show()

# Misc
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Pre-processing

articles = [
    'Ukraine’s president of the united states has made a desperate appeal to the Russian people of the united states, and the president of the united states.',
    'For many Russians of the united states, it was an unfamiliar sight to see the faces of the two leaders in an unfamiliar sight.',
    'The striped bats were hanging on their feet for best corpora alumni and ate best fishes COVID-19 am are is were was children that were striped with bands of sunlight and stripped from their possesions.']

# TODO
# Step 0: Calculate average length per article (before or after preprocessing?).

if __name__ == '__main__':
    print(articles)
