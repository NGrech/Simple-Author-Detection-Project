import os
import re
import time
import pickle
from string import ascii_lowercase
from nltk import PorterStemmer
from itertools import groupby
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from sklearn import preprocessing


if os.path.isfile('X.pickle') and os.path.isfile('y.pickle'):
    with open('X.pickle', 'rb') as handle:
            X = pickle.load(handle)
    with open('y.pickle', 'rb') as handle:
            y = pickle.load(handle)
else:
    X = {}
    y = {}

specialChar = ['~', '@', '#', '$', '%', '^', '&', '*', '-', '_', '=', '+', '>', '<', '[', ']', '{', '}', '/', '\\', '|']
punctuation = [',', '.', '?', '!', ':', ';', '\'', '"']
allChars = specialChar + punctuation + list(ascii_lowercase)
functionWords = open("functionWords.txt").read().split()


# yule function adapted from : http://swizec.com/blog/measuring-vocabulary-richness-with-python/swizec/2528
def yule(entry):
    # yule's I measure (the inverse of yule's K measure)
    # higher number is higher diversity - richer vocabulary
    d = {}
    stemmer = PorterStemmer()
    for w in entry:
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1

    m1 = float(len(d))
    m2 = sum([len(list(g))*(freq**2) for freq, g in groupby(sorted(d.values()))])

    try:
        return (m1*m1)/(m2-m1)
    except ZeroDivisionError:
        return 0


def get_feature_vectors():
    folders = ["train", "test"]

    # looping through the test and training folders
    for f in folders:
        print("Folder:" + f)
        X[f] = {"Lexical": [], "WordBased": []}
        y[f] = []
        an = 0
        # looping through each author in the folder
        for author in os.listdir(f):

            start = time.time()
            # looping through each authors documents
            for doc in os.listdir(f + "/" + author):

                lexical_features = Counter()
                word_based_features = Counter()

                if doc.endswith(".txt"):
                    docname = f + "/" + author + "/" + doc
                    lines = [line.rstrip('\n') for line in open(docname)]
                    text = " ".join(lines)
                    templine = re.sub("[^a-zA-Z]", " ", text)  # removing all non alpha chars
                    templine = templine.lower()
                    words = templine.split()

                    # initialization
                    word_based_features["tShortWords"] = 0
                    for c in allChars:
                        lexical_features["n"+c] = 0
                    for w in functionWords:
                        word_based_features[w] = 0
                    for i in range(1, 21):
                        word_based_features["nWordFreq_"+str(i)] = 0

                    # Lexical features
                    lexical_features["tChars"] += len(text)
                    lexical_features["tAlpha"] += sum(c.isalpha() for c in text) / lexical_features["tChars"]
                    lexical_features["tUpper"] += sum(c.isupper() for c in text) / lexical_features["tChars"]
                    lexical_features["tDigit"] += sum(c.isdigit() for c in text) / lexical_features["tChars"]
                    lexical_features["tWhite"] += sum(c.isspace() for c in text) / lexical_features["tChars"]
                    for c in text.lower():
                        if c in allChars:
                            lexical_features["n"+c] += 1
                    lexical_features["tSentences"] = min(len(lines), lexical_features["n."])

                    # Word based features
                    for w in words:
                        word_based_features["tWords"] += 1
                        wlen = len(w)
                        if wlen < 4:
                            word_based_features["tShortWords"] += 1
                        word_based_features["tCharsIntWords"] += len(w)
                        if wlen <= 20:
                            word_based_features["nWordFreq_"+str(wlen)] += 1
                        if w in functionWords:
                            word_based_features[w] += 1

                    word_based_features["avgWordLen"] = lexical_features["tChars"] / word_based_features["tWords"]

                    word_based_features["c_avgSentenceLen"] = \
                        lexical_features["tChars"] / lexical_features["tSentences"]

                    word_based_features["w_avgSentenceLen"] = \
                        word_based_features["tWords"] / lexical_features["tSentences"]

                    word_based_features["nUniqueWords"] = len(set(words))

                    word_based_features["YulesI"] = yule(words)

                    word_based_features["tShortWords"] = \
                        word_based_features["tShortWords"] / word_based_features["tWords"]

                    word_based_features["nUniqueWords"] = \
                        word_based_features["nUniqueWords"] / word_based_features["tWords"]

                    for wf in range(1, 21):
                        word_based_features["nWordFreq_"+str(i)] =\
                            word_based_features["nWordFreq_"+str(i)] / word_based_features["tWords"]

                    # Setting vectors
                    X[f]["Lexical"].append(list(lexical_features.values()))
                    X[f]["WordBased"].append(list(word_based_features.values()))
                    y[f].append(an)

            done = time.time()
            elapsed = done - start
            print("Finished author: " + author + " " + str(an+1) + "/50 Time Taken: " + str(elapsed))
            an += 1
    with open('X.pickle', 'wb') as handlex:
        pickle.dump(X, handlex)
    with open('y.pickle', 'wb') as handley:
        pickle.dump(y, handley)


def run_experiment(n_authors=5, features=0):

    model_names = [
        "KNN - 5",
        "KNN - 10",
        "KNN - 15",
        "SVC",
        "linear SVC",
        "GaussianNB"
    ]

    models = [
        KNeighborsClassifier(5),
        KNeighborsClassifier(10),
        KNeighborsClassifier(15),
        SVC(),
        LinearSVC(),
        GaussianNB()
    ]

    z = {}
    x_train = []
    x_test = []

    true_range = n_authors*50

    # combination of feature sets
    if features not in range(0, 3):
        features = 2
    if features == 0:
        x_train = X["train"]["Lexical"]
        x_test = X["test"]["Lexical"]
    if features == 1:
        x_train = X["train"]["WordBased"]
        x_test = X["test"]["WordBased"]
    if features == 2:
        for j in range(0, len(X["train"]["Lexical"])):
            vec = X["train"]["Lexical"][j] + X["train"]["WordBased"][j]
            x_train.append(vec)
        for j in range(0, len(X["test"]["Lexical"])):
            vec = X["test"]["Lexical"][j] + X["test"]["WordBased"][j]
            x_test.append(vec)

    x_test = x_test[0:true_range]
    x_train = x_train[0:true_range]
    y_test = y["test"][0:true_range]
    y_train = y["train"][0:true_range]

    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    for i in range(0, 6):
        models[i].fit(x_train, y_train)
        z[model_names[i]] = models[i].predict(x_test)
        print(model_names[i])
        print(precision_recall_fscore_support(y_test, z[model_names[i]], average='macro'))


if not bool(X) and not bool(y):
    get_feature_vectors()

for n in [5, 25, 50]:

    print("----Number of Authors:" + str(n))
    # lexical only
    for k in [0, 1, 2]:
        if k == 0:
            print("F1")
        if k == 1:
            print("F2")
        if k == 2:
            print("F3")
        run_experiment(n, k)
