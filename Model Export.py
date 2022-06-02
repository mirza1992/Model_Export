import pickle

Pkl_Vectorizer = pickle.load(open("Pickle_TFIDF_Topic_Model.pkl","rb"))
Pkl_LDA = pickle.load(open("Pickle_Topic_Model.pkl","rb"))


for index, topic in enumerate(Pickle_Topic_Model.components_):
  print(f"The Top Words For Topic #{index}")
  print([Pickle_Topic_Models.get_feature_names()[i] for i in topic.argsort()[-15:]])
  print('\n')
