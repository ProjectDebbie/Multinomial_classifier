# master thesis project - Clarence Lépine (UPC Barcelona)
# data extraction from PubTator annotations


# import the packages
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import glob

# load the data
with open(r'...\pubtator_annotations_final.txt', 'w') as output_file:
    with open(r'...\pubtator_annotations_final.csv', 'r') as input_file:
        [output_file.write("".join(row)) for row in input_file]

# to remove the blanklines from the data
input = r'...\pubtator_annotations_final.txt'
output = r'...\pubtator_annotations_final_clean.txt'

def remove_blanklines():
    with open(input, "r") as f, open(output, "w") as outline:
        for i in f.readlines():
            if not i.strip():
                continue
            if i.startswith(",,"):
                continue
            if i:
                outline.write(i)


# remove_blanklines()


# to convert the text data into a pandas dataframe
with open(output) as f:
    content = f.readlines()
    IDs, texts, bioconcepts = ([], [], [])
    for line in content:
        ID, text, bioconcept = line.split(',', 2)
        IDs.append(ID)
        texts.append(text)
        bioconcepts.append(bioconcept)

    df = pd.DataFrame()
    df['ID'] = IDs
    df['text'] = texts
    df['bioconcept'] = bioconcepts
    print(df)
    df.to_csv(r'...\pubtator_annotations_final_clean.csv', index=False, header=True, encoding='utf-8')

# to get the 20 most commun terms in the annotations and plot them in a bar diagram
results = pd.Series(' '.join(df.text).split()).value_counts()[:20]
print('Most frequent words in Pubtator annotations:\n',results)
results.plot.bar()
plt.title('Term frequencies in Pubtator annotations')
plt.show()

# other method
# to convert the text into tokens and get the 20 most common tokens in the annotations
def my_tokenizer(text):
    return text.split()

tokens = df.text.map(my_tokenizer).sum()
counter = Counter(tokens)
counter.most_common(20)

# to plot the results in a word cloud
def wordcloud(counter):
    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(counter)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# wordcloud(counter)

# text preprocessing on the annotations (the results are put in a csv file)
with open(r'...\pubtator_annotations_final_clean.csv', 'r') as input:
    with open(r'...\pubtator_annotations_final_clean_final.csv', 'w') as output:
        for i in input.readlines():
            i = i.replace('Disease\n', 'Disease')
            i = i.replace('Species\n', 'Species')
            i = i.replace('Chemical\n', 'Chemical')
            i = i.replace('Gene\n', 'Gene')
            # i = i.replace('Mutation\n', 'Mutation')
            output.write(i)

# to put the clean annotations in a new dataframe & to plot the distribution of each bioconcept within the entire annotations in a pie diagram
# here if a biocept appears several times for a same abstract, we count it each time
df_clean = pd.read_csv(r'...\pubtator_annotations_final_clean_final.csv')
print(df_clean)
results = pd.Series(' '.join(df_clean.bioconcept).split()).value_counts()[:4]
results.plot(kind='pie',autopct='%1.1f%%')
plt.show()

# to get the annotations for the chemical bioconcept & plot the 20 most commun terms in a bar diagram
chemical = df_clean[df_clean['bioconcept'] == 'Chemical']
chemical.to_csv(r"...\pubtator_annotations_chemical.csv") # to put the annotations from the bioconcept in a dedicated csv file
results_chemical = pd.Series(' '.join(chemical.text).split()).value_counts()[:20] # to get the 20 most commun terms
print('Most frequent words for the "Chemical" bioconcept:\n',results_chemical)
results_chemical.plot.bar() # to plot the results
plt.title('Term frequencies for the "Chemical" bioconcept')
plt.show()

# to get the annotations for the disease bioconcept & plot the 20 most commun terms in a bar diagram
disease = df_clean[df_clean['bioconcept'] == 'Disease']
disease.to_csv(r"...\pubtator_annotations_disease.csv") # to put the annotations from the bioconcept in a dedicated csv file
results_disease = pd.Series(' '.join(disease.text).split()).value_counts()[:20] # to get the 20 most commun terms
print('Most frequent words for the "Disease" bioconcept:\n',results_disease)
results_disease.plot.bar() # to plot the results
plt.title('Term frequencies for the "Disease" bioconcept')
plt.show()

# to get the annotations for the species bioconcept & plot the 20 most commun terms in a bar diagram
species = df_clean[df_clean['bioconcept'] == 'Species']
species.to_csv(r"...\pubtator_annotations_species.csv") # to put the annotations from the bioconcept in a dedicated csv file
results_species = pd.Series(' '.join(species.text).split()).value_counts()[:20] # to get the 20 most commun terms
print('Most frequent words for the "Species" bioconcept:\n',results_species)
results_species.plot.bar() # to plot the results
plt.title('Term frequencies for the "Species" bioconcept')
plt.show()

# to get the annotations for the gene bioconcept & plot the 20 most commun terms in a bar diagram
gene = df_clean[df_clean['bioconcept'] == 'Gene']
gene.to_csv(r"...\pubtator_annotations_gene.csv") # to put the annotations from the bioconcept in a dedicated csv file
results_gene = pd.Series(' '.join(gene.text).split()).value_counts()[:20] # to get the 20 most commun terms
print('Most frequent words for the "Gene" bioconcept:\n',results_gene)
results_gene.plot.bar() # to plot the results
plt.title('Term frequencies for the "Gene" bioconcept')
plt.show()

