# Multinomial_classifier

This repository was created by Clarence Lépine, a Master student from the UPC Barcelona, and contains the work that was done during her Master Thesis Project.
The project was entitled "AUTOMATED TEXT CLASSIFICATION USING MACHINE LEARNING – APPLIED TO AN ABSTRACTS CORPUS OF 3D PRINTING FOR BIOMEDICAL APPLICATIONS".

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
This repository is completely open, anyone can download the files and add some modifications to it.

The file "multinomial classifier.py" contains the multi-class classifier that was implemented during this project.
A total of 4 classification models were tested in this code: Multinomial Naive Bayes, Stochastic Gradient Descent, Random Forest and k-Nearest-Neighbors.
The best results were obtained with the SGD classification model. We managed to get 0.92 of accuracy and 0.89 of f1-score.

The 3d printing testing set were created manually by looking at the 3d printing literature from Pubmed andcontains a list 477 PMIDs of articles about 3d printing.
In this set, we will find clinical studies, in vivo studies, in vitro studies, non biomaterials studies and studies that contains both in vivo and in vitro experiments.

The 3d printing corpus contains the PMIDs of the entire 3d printing literature available up to November 2020 which represents 11,153 articles.
These articles were found using the following Pubmed query:(((3d printing) OR (3d-printing) OR (three dimensional printing) OR (bioprinting)) NOT ((review)[Publication Type])) NOT ((systematic review)[Publication Type])



