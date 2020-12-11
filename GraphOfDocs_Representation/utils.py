"""
This script contains utility functions
e.g to read files, preprocess text, etc.
"""
from os import system
from os import listdir
from os.path import isfile, join
from string import punctuation, printable
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer() # Initialize lemmatizer once.
stemmer = PorterStemmer() # Initialize Porter's stemmer once.

stop_words = set(stopwords.words('english')).union([ # Augment the stopwords set.
    'don','didn', 'doesn', 'aren', 'ain', 'hadn',
    'hasn', 'mightn', 'mustn', 'couldn', 'shouldn',
    'dont', 'didnt', 'doesnt', 'arent', 'aint',
    'hadnt', 'hasnt', 'may', 'mightve', 'couldnt',
    'shouldnt', 'shouldnot', 'shouldntve', 'mustnt',
    'would', 'woulda', 'wouldany', 'wouldnot', 'woudnt',
    'wouldve', 'must', 'could', 'can', 'have', 'has',
    'do', 'does', 'did', 'are', 'is', 'ive', 'cant', 'thats',
    'isnt', 'youre', 'wont', 'from', 'subject', 'hes', 'etc',
    'edu', 'com', 'org', 've', 'll', 'd', 're', 't', 's'])

def get_wordnet_tag(tag):
    """
    Function that maps default part-of-speech 
    tags to wordnet part-of-speech tags.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else: #default lemmatizer parameter
        return wordnet.NOUN

def generate_words(text, extend_window = False, insert_stopwords = False, lemmatize = False, stem = False):
    """
    Function that generates words from a text corpus and optionally lemmatizes them.
    Returns a set of unique tokens based on order of appearance in-text.
    """
    # Remove all whitespace characters (by split) and join on space.
    text = ' '.join(text.split())
    # Handle special characters that connect words.
    text = text.translate({ord(c): '' for c in '\'\"'})
    # Find all end of sentences and introduce a special string to track them.
    # If they aren't tracked, then the window is allowed to be extended from one sentence to another,
    # thus connecting the last terms of one sentence with the starting ones of the next.
    # Also, by chaining the replace methods together, a slight amount of performance is achieved,
    # over other methods, that have the same output.
    if not extend_window:
        text = text.replace('. ', ' e5c ')\
                    .replace('! ', ' e5c ' )\
                    .replace('? ', ' e5c ' )
    # Translate punctuation to space and lowercase the string.
    text = text.translate({ord(c): ' ' for c in punctuation}).lower()
    # We are cleaning the data from stopwords, numbers and leftover syllabes/letters.
    if not insert_stopwords:
        tokens = [token for token in word_tokenize(text)
        if not token in stop_words and not token.isnumeric() and len(token) > 2]
    else:
        tokens = word_tokenize(text)
    if lemmatize:
        tokens_tags = pos_tag(tokens) # Create part-of-speech tags.
        # Overwrite the list with the lemmatized versions of tokens.
        tokens = [lemmatizer.lemmatize(token, get_wordnet_tag(tag)) for token, tag in tokens_tags]
    if stem:
        # Overwrite the list with the stemmed versions of tokens.
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def clear_screen(current_system):
    if current_system == 'Windows':
        system('cls')
    else:
        system('clear') # Linux/OS X.
    return

def jaccard_similarity(list_1, list_2):
    """
    Function to calculate the jaccard similarity,
    between two list. If either of them is empty,
    the similarity is 0.0.
    """
    if not list_1 or not list_2:
        return 0.0

    set1 = set(list_1.keys())
    set2 = set(list_2.keys())
    return len(set1.intersection(set2)) / len(set1.union(set2))
