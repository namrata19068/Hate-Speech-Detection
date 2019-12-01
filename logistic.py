
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union


class_names = ['OAG', 'NAG', 'CAG']

train = pd.read_csv('/home/divya/Pictures/NLP Project/english/agr_en_train.csv').fillna(' ')
#train.append(pd.read_csv('/home/divya/Pictures/NLP Project/hindi/agr_hi_train(2).csv').fillna(' '))
test = pd.read_csv('/home/divya/Pictures/NLP Project/english/Book1.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)
vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)
vectorizer.fit(all_text)
train_features = vectorizer.transform(train_text)
test_features = vectorizer.transform(test_text)

scores = []
submission = pd.DataFrame.from_dict({'comment_text': test['comment_text']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]


submission.to_csv('submission.csv', index=False)

dataset = pd.read_csv('/home/divya/Downloads/Ytest.csv', delimiter = ',')
#dataset1 = pd.read_csv('C:/Users/Namrata/Desktop/NLP_material/Project_material/english/agr_en_train.csv', delimiter = ',')
y_actual = dataset.iloc[:, 0].values
print(y_actual)