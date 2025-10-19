# La intencion es dejar aca el codigo "oficial"
# y usar el jupyter solo para pruebas, asi es mas facil el merge.

import csv

with open("TA1C_dataset_detection_train.csv","r", encoding="UTF-8") as f:
    train = [x for x in csv.reader(f)]
with open("TA1C_dataset_detection_dev_gold.csv","r", encoding="UTF-8") as f:
    dev = [x for x in csv.reader(f)]

train_headlines = [x[4] for x in train[1:]]
train_clickbait = [x[5] for x in train[1:]]

dev_headlines = [x[4] for x in dev[1:]]
dev_clickbait = [x[6] for x in dev[1:]]

class Elem:
    def __init__(self, name):
        self.name = name
        super().__init__()

class Vectorizer(Elem):
    def __init__(self, vectorizer, name):
        self.vectorizer = vectorizer
        super().__init__(name)

class Model(Elem):
    def __init__(self, model, name):
        self.model = model
        super().__init__(name)
        
        

from sklearn.metrics import f1_score, classification_report


def train_and_eval(vectorizer, clf, imprimir, n_vec, n_clf):
    try:  
        training_features = vectorizer.fit_transform(train_headlines)
        clf.fit(training_features, train_clickbait)

        dev_features = vectorizer.transform(dev_headlines)
        prediction = clf.predict(dev_features)
        f1_macro = round(f1_score(dev_clickbait, prediction, average='macro')*100, 2)
        
        if imprimir:
            print(f"vectorizer: {n_vec}, classifier: {n_clf}")
            print(f"F1-Score macro: {str(f1_macro)}\n")
            print(classification_report(dev_clickbait, prediction))
    
        return f1_macro
    except Exception as e:
        # Si ocurre un error, lo imprimimos y devolvemos un valor fijo
        print(f"Error en train_and_eval con {vectorizer} y {clf}: {e}")
        return -1
    

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
        
    
Tfidf_v = Vectorizer(TfidfVectorizer(ngram_range=(1,6), strip_accents= 'unicode'), "Tfidf")
hashing_v = Vectorizer(HashingVectorizer(n_features=1000), "hashing")
bow_v = Vectorizer(CountVectorizer(ngram_range=(1,3), strip_accents= 'unicode'), "Bag of Words")

svc_c = Model(SVC(), "SVC")
lsvc_c = Model(LinearSVC(), "Linear SVC")
logreg_c = Model(LogisticRegression(), "Logistic Regression")
mnnbayes_c = Model(MultinomialNB(), "Multinomial NB")
rforest_c = Model(RandomForestClassifier(n_estimators=100), "Random Forest")

vectorizers = [bow_v, Tfidf_v, hashing_v]
classifiers = [svc_c, logreg_c, mnnbayes_c, rforest_c, lsvc_c]

best_f1_macro, vec, clf = 0,0,0

for n_vec, vectorizer in enumerate(vectorizers):
    for n_clf, classifier in enumerate(classifiers):
        f1_macro = train_and_eval(vectorizer.vectorizer, classifier.model, True, n_vec, n_clf)
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            vec = n_vec
            clf = n_clf

print(f"el mejor f1-macro es {best_f1_macro} y es logrado por el clasificador {classifiers[clf].name} vectorizando con {vectorizers[vec].name}")