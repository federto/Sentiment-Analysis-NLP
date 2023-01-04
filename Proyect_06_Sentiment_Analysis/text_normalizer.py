import re
import nltk
import spacy
import unicodedata

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.porter import *


tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()



def remove_html_tags(text):
    # Put your code
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return text


def stem_text(text):
    # Put your code
    new_text = []
    parts = tokenizer.tokenize(text)
    for w in parts:
        new_text.append(stemmer.stem(w))
    text_output = " ".join(new_text)
    return text_output


def lemmatize_text(text):
    # Put your code
    # Creates an empty list
    lemma_list = []
 
    # creates an nlp document wich tockenize the original text
    doc = nlp(text)
    # Aplyies the lemmatization
    for token in doc:
        lemma_list.append(token.lemma_)

    #recovering text format
    text = " ".join(lemma_list)

    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Put your code
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    text = re.sub("'", "", expanded_text)
    return text


def remove_accented_chars(text):
    # Put your code
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_chars(text, remove_digits=False):
    # Put your code
    pattern = "[^a-zA-Z0-9\s]|(_)|(^)" if not remove_digits else r'[^a-zA-z\s]|()|(^)'
    text = re.sub(pattern, '', text)

    return text

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    # Put your code
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    text = ' '.join(filtered_tokens)    
    return text


def remove_extra_new_lines(text):
    # Put your code
    text = re.sub(r'[\r|\n|\r\n]+', ' ',text)
    return text


def remove_extra_whitespace(text):
    # Put your code
    text = re.sub(' +', ' ', text)
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
