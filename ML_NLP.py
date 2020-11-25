 
#NLP 

#Lots of use case when dealing with Text or Unstructured Text Data
#Example: we have two documents 
    # Red House
    # Blue House
#Featurize based on word count:
#A document represented as a vector of word counts is called a "Bag of Words"
    #'Blue House' --> (red, blue, house) --> (0,1,1) vector
    # 'Red House' --> (red, blue, house) --> (1,0,1) vector
#We can plot these two vectors and plot a Cosine similarity (dot.product) and 
#... see how similar two text documents are two each other



#We can improve on Bag of Words by adjusting word counts based on frequency in corpus (the groups of all the documents)
#Use something called TF-IDF (Term Frequency - Inverse Document Frequency)

#Term Frequency - Importance of the term within that document
    # TF(d,t) = Number of occurences of term t in the document d 
#Inverse Document Frequency = Importance of the term in the corpus (group of all documents)
    # IDF(t) = log(D/t) where:
        # D = total number of documents
        # t = number of documents with the term

#Mathematically, TF-IDF expression:

import nltk   
nltk.download_shell()
