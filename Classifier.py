#https://stackabuse.com/text-classification-with-python-and-scikit-learn/
import re
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

DATA_FOLDER = "data"

def classifier(dataset_name):
    nltk_folder = "{}/{}/result_nltk_prep".format(DATA_FOLDER, dataset_name)

    print(dataset_name)
    print(nltk_folder)

    news_data = load_files(nltk_folder,load_content=True,encoding='utf-8')
    X, y = news_data.data, news_data.target
    documents = []

    for sen in range(0, len(X)):
        document = str(X[sen])
        documents.append(document)

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    with open('text_classifier', 'wb') as picklefile:
        pickle.dump(classifier,picklefile)

    with open('text_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    y_pred2 = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred2))
    print(classification_report(y_test, y_pred2))
    print(accuracy_score(y_test, y_pred2))

