import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Custom Functions
 
def shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'
 
    return word_shape


def word2features(sent, i):
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    word_postag = nltk.pos_tag(sent)
    word = word_postag[i][0]
    pos = word_postag[i][1]

    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word.shape':shape(word),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.stem':stemmer.stem(word),
        'word.lemma':lemmatizer.lemmatize(word.lower(), pos=get_wordnet_pos(pos)),
        'word.pos':pos.strip('$'),
        'word.pos[:2]': pos[:2],
        
    }
    if i > 0:
        word1 = word_postag[i-1][0]
        pos1 = word_postag[i-1][1]
        features.update({
            '-1:word.lower': word1.lower(),
            '-1:word.shape':shape(word1),
            '-1:word.stem':stemmer.stem(word1),
            '-1:word.lemma':lemmatizer.lemmatize(word1.lower(), pos=get_wordnet_pos(pos1)),
            '-1:word.pos':pos1.strip('$'),
            '-1:word.pos[:2]': pos1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = word_postag[i+1][0]
        pos1 = word_postag[i+1][1]
        features.update({
            '+1:word.lower': word1.lower(),
            '+1:word.shape':shape(word1),
            '+1:word.stem':stemmer.stem(word1),
            '+1:word.lemma':lemmatizer.lemmatize(word1.lower(), pos=get_wordnet_pos(pos1)),
            '+1:word.pos':pos1.strip('$'), # Stripping '$' from PRP$
            '+1:word.pos[:2]': pos1[:2],
        })
    else:
        features['EOS'] = True
        
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
