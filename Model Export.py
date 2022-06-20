import pickle
import gensim
import spacy
import numpy as np
import pandas as pd
nlp = spacy.load("en_core_web_sm")

Pkl_Vectorizer    = pickle.load(open("Pickle_TFIDF_Topic_Model.pkl","rb"))
Pkl_best_LDAa     = pickle.load(open("Pickle_lda_best_Topic_Model.pkl","rb"))
Pkl_keywords_LDAa = pickle.load(open("Pickle_keyword_Topic_Model.pkl","rb"))
Pkl_fit_LDAa      = pickle.load(open("Pickle_lda_fit_Topic_Model.pkl","rb"))


mytext = []

user_input = input("Enter Article:")

mytext.append(user_input)


def predict_topic(mytext):

    def sent_to_words(sentences):
        for sentence in sentences: 
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out        

    mytext_2 = list(sent_to_words(mytext))

    mytext_3 = lemmatization(texts = mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    mytext_4 = Pkl_Vectorizer.transform(mytext_3)


    def show_topics(vectorizer=Pkl_Vectorizer, lda_model=Pkl_fit_LDAa, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = show_topics(vectorizer=Pkl_Vectorizer, lda_model=Pkl_best_LDAa, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]


    topic_probability_scores = Pkl_best_LDAa.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()

    
    return print("\n"),print("Topics Predicted : ",topic,"\n"),print("Topics Allocated : ",np.argmax(topic_probability_scores))

# import time
if __name__ == "__main__":
    predict_topic(mytext)


