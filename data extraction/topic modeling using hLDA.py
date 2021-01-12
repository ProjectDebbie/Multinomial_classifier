#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pylab as plt
import pandas as pd
import string
import glob
import re

basedir = r'C:\Users\archi\Documents\Multinomial_classifier-master\\'
sys.path.append(basedir)

import nltk

nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from hlda.sampler import HierarchicalLDA
# from pattern.pattern.text.en.inflect import singularize
from pattern.text import singularize
from ipywidgets import widgets
from IPython.core.display import display, HTML
from wordcloud import WordCloud

stopset = stopwords.words('english') + list(string.punctuation)

corpus = []
all_docs = []
vocab = set()

for filename in glob.glob(r'C:\Users\archi\Documents\Multinomial_classifier-master\3d_printing_clean\*.txt'):
    with open(filename) as f:
        try:
            doc = f.read().splitlines()  # read lines
            doc = filter(None, doc)  # remove empty string
            doc = '. '.join(doc)
            doc = doc.translate(str.maketrans('', '', string.punctuation))  # strip punctuations
            doc = re.sub('^[0-9]{1,6}$', '', doc)
            doc = doc.lower()
            all_docs.append(doc)
            tokens = word_tokenize(str(doc))

            filtered = []
            for w in tokens:
                if len(w) < 3:  # remove short tokens
                    continue
                if w in stopset:  # remove stopwords
                    continue
                w = singularize(w)  # make words singular
                filtered.append(w)

            vocab.update(filtered)
            corpus.append(filtered)

        except UnicodeDecodeError:
            print('failed to load', filename)

# In[ ]:


# create a numebered vocabulary
vocab = sorted(list(vocab))
vocab_index = {}
for i, w in enumerate(vocab):
    vocab_index[w] = i

# convert words into indices
new_corpus = []
for doc in corpus:
    new_doc = []
    for word in doc:
        print(word)
        word_idx = vocab_index[word]
        new_doc.append(word_idx)
    new_corpus.append(new_doc)

# In[ ]:


n_samples = 200  # no of iterations for the sampler
alpha = 10.0  # smoothing over level distributions (10)
gamma = 1  # CRP smoothing parameter; number of imaginary customers at next, as yet unused table (1)
eta = 0.1  # smoothing over topic-word distributions (0.1)
num_levels = 3  # the number of levels in the tree
display_topics = 40  # the number of iterations between printing a brief summary of the topics so far
n_words = 6  # the number of most probable words to print for each topic after model estimation
with_weights = False  # whether to print the words with the weights

hlda = HierarchicalLDA(new_corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)

# In[6]:


colour_map = {
    0: 'blue',
    1: 'red',
    2: 'green',
    3: 'yellow'
}


def show_doc(d=0):
    node = hlda.document_leaves[d]
    print(node)
    path = []
    while node is not None:
        path.append(node)
        print(path)
        node = node.parent
    path.reverse()
    n_words = 10
    with_weights = False
    for n in range(len(path)):
        node = path[n]
        colour = colour_map[n]
        msg = 'Level %d Topic %d: ' % (node.level, node.node_id)
        msg += node.get_top_words(n_words, with_weights)
        output = '<h%d><span style="color:%s">%s</span></h3>' % (n + 1, colour, msg)
        display(HTML(output))
    display(HTML('<hr/><h5>Processed Document</h5>'))
    doc = corpus[d]
    output = ''
    for n in range(len(doc)):
        w = doc[n]
        l = hlda.levels[d][n]
        colour = colour_map[l]
        output += '<span style="color:%s">%s</span> ' % (colour, w)
    display(HTML(output))


show_doc(d=0)
widgets.interact(show_doc, d=(0, len(corpus) - 1))

# In[ ]:


pmid = []
nodes = []
node_levels = []
node_ids = []
keywords = []
info = {'PMID': pmid, 'Node': nodes, 'Node level': node_levels, 'Node id': node_ids, 'Node keywords': keywords}

for d in range(len(corpus)):
    doc = corpus[d]
    node = hlda.document_leaves[d]
    pmid.append(doc[0])
    nodes.append(node)
    node_levels.append(node.level)
    node_ids.append(node.node_id)
    keywords.append(node.get_top_words(6, False))

pdo_results = pd.DataFrame(
    {'PMID': pmid, 'Node': nodes, 'Node level': node_levels, 'Node id': node_ids, 'Node keywords': keywords})

# In[ ]:


print(pdo_results)

pdo_results.to_csv(r'C:\Users\archi\Documents\Multinomial_classifier-master\pdo_topics_final.csv', index=False)

# In[ ]:
