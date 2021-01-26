# Multinomial_classifier

This repository was created by Clarence Lépine, a Master student from the UPC Barcelona, and contains the work that was done during her Master Thesis Project.
The project was entitled "TEXT CLASSIFICATION OF BIOMATERIALS ABSTRACTS AND INFORMATION EXTRACTION FROM THE 3D PRINTING LITERATURE FOR BIOMEDICAL APPLICATIONS USING MACHINE LEARNING ALGORITHMS".

The objectives of the project are the following:
- Create a new 3D printing testing set using manual curation from Pubmed
- Test the binary classifier from DEBBIE project with the 3D printing set
- Create a multinomial classifier that can classify Pubmed articles in 3 categories: non biomaterials studies, in vivo/in vitro studies and clinical studies
- Test the performance of the classifier using the 3D printing set
- Classify the entire 3D printing literature available from Pubmed
- Do text analysis to extract informations about the 3D printing literature:
    1) by extracting the most frequent terms from each category (using tf and tf-idf)
    2) by doing semantic analysis using Pubtator annotations
    3) by using topic mining techniques such as LDA and hLDA

In this repository, we will find all the ressources to reproduce the project such as the the datasets and the Python codes that were used for text classification and data extraction.
This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the LICENSE file for details.
So the repository is completely open source, anyone can download the files and add some modifications to it.

The "text classification" folder contains the code to retrieve the Pubmed abtracts from the MEDLINE format. This code was implemented by Osnat Hakimi.
This code needs to be used before using the binary classifer from DEBBIE.
The file "binary classifier from DEBBIE.py" is an adaptation of the SVM classifer from DEBBIE project where 2 classification models were compared (SGD and Random Forest) and where the performances of the models are evaluated.
The file "multinomial classifier.py" contains the multi-class classifier that was implemented during this project.
It classifies articles into one of three following categories: clinical studies, in vivo/in vitro studies and non biomaterials studies.
In total 4 classification models were tested for the multinomial classification: Multinomial Naive Bayes, Stochastic Gradient Descent, Random Forest and k-Nearest-Neighbors.
The best results were obtained with the SGD classification model. We managed to get 0.92 of accuracy and 0.89 of f1-score.
A file is also included in the "multinomial classifier" folder to change the format of the data before the multinomial classification.

The "data extraction folder" contains the codes to do text analysis.
The file "data_extraction_from_multinomial_classifier_results.py"contains the code to extract the most frequent terms from the multinomial classification results using term frequency.
The file "data_extraction_from_pubtator.py" contains the code to extract the most frequent terms from each bioconcept results using term frequency.
NB: Before using this code, you must download the PubTator annotations of your corpus and remove the titles, the abstacts and the MeSH terms from it. You can easily do that by taking your pubtator file, converting it to txt file and open it into Excel to remove the columns and the lines without importance. To perform the text analysis, we only need three columns: the PMID, the annotation, and the bioconcept.

If you want to do topic mining on your corpus, here are some useful links: 
code from Shashank Kapadia: https://github.com/kapadias/mediumposts/blob/master/natural_language_processing/topic_modeling/notebooks/Introduction%20to%20Topic%20Modeling.ipynb
This link contains a code to do topic mining using Latent Dirichlet Allocation.
code from Joe Wandy: https://github.com/joewandy/hlda/blob/master/notebooks/bbc_test.ipynb
This link contains a code to do topic mining using hierarchical Latent Dirichlet Allocation.

The "3d printing testing set" was created manually by looking at the 3d printing literature from Pubmed and contains a list 477 PMIDs of articles about 3d printing.
In this set, we will find clinical studies, in vivo studies, in vitro studies, non biomaterials studies and studies that contain both in vivo and in vitro experiments.
NB: for the multinomial classification, we decided to make one class with all of the articles that were either in vivo, in vitro or both at the same time, i.e. gathering all types of pre-clinical studies in one category called in vivo/in vitro.

The file "3d printing corpus.zip" contains the abstracts of the entire 3d printing literature available from Pubmed up to November 2020 which represents a total of 11,942 articles.
These abstracts were found using the following Pubmed query:(((3d printing) OR (3d-printing) OR (three dimensional printing) OR (bioprinting)) NOT ((review)[Publication Type])) NOT ((systematic review)[Publication Type]).
NB: After changing the format of the abstracts, we were able to classifier only 11,153 abtracts from the corpus.
NB: To retrieve the abstracts from the given Pubmed query, you can use the Ebot tool from NCBI (link:https://www.ncbi.nlm.nih.gov/Class/PowerTools/eutils/ebot/ebot.cgi).
This tool works by taking a list of PMIDs or a Pubmed query to generate a perl code that you can run on your terminal in order to get the abstracts in MEDLINE format.
