import pandas as pd

url = 'http://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url,sep = '\t',header = None,names = ['label','message'])

print(sms.shape)

print(sms.head(10))

print(sms.label.value_counts())


#to Visualization
import matplotlib.pyplot as plt
y = sms.label.value_counts()
plt.show(y.plot(kind='bar'))

sms['label_num']=sms.label.map({'ham':0,'spam':1})#map 0 & 1 to ham and spam.

print(sms.shape)

print(sms.head(5))

x=sms.message

y=sms.label_num

print(x.shape)
print(y.shape)



# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(X_train.shape)#traing messages
print(X_test.shape)#testing messages
print(y_train.shape)#training labels
print(y_test.shape)#testing labels


# instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# equivalently: combine fit and transform into a single step
#X_train_dtm = vect.fit_transform(X_train)


print(X_train_dtm.shape)


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape)


'''The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). 
The multinomial distribution normally requires integer feature counts.
However, in practice, fractional counts such as tf-idf may also work'''


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# train the model using X_train_dtm (timing it with an IPython "magic command")
nb.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
y_pred_class

test_arr=["I am a boy","i want free entry"]
x_test=pd.Series(test_arr)
x_test=vect.transform(x_test)
nb.fit(X_train_dtm, y_train)
y_pred_class1 = nb.predict(x_test)
print(y_pred_class1)

print(len(y_pred_class))


print("True:",y_test.values[0:25])
print("predicted:",y_pred_class[0:25])


count=0
for i in range(len(y_pred_class)):
    if y_test.values[i]==y_pred_class[i]:
        count=count+1
print(count)

print("accuracy percentage",count/len(y_pred_class))