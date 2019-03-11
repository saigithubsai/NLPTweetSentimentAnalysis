#!/usr/bin/env python
# coding: utf-8

# In[20]:
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
# In[21]:
print(format_sentence("The cat is very cute"))
# In[22]:
import nltk
# In[23]:
pos = []
with open("C:/Users/avala/Downloads/twilio-sent-analysis-master/pos_tweets.txt",encoding="UTF8") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
# In[24]:
neg = []
with open("C:/Users/avala/Downloads/twilio-sent-analysis-master/neg_tweets.txt",encoding="UTF8") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])
# In[25]:
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
# In[26]:
from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(training)
# In[27]:
classifier.show_most_informative_features()
# In[38]:
tweet1 = "It was my great honor to host a @WhiteHouse Conference on Supporting Veterans & Military Families..."
print(classifier.classify(format_sentence(tweet1)))
# In[29]:
tweet2 = "Yesterday the amazing @Motopaws volunteers joined us in saying yes to the pooch and no to the pataka. Together we were able to feed over 500 street dogs across 7 cities! This Diwali, let’s think about the animals and celebrate responsibly."
print(classifier.classify(format_sentence(tweet2)))
# In[30]:
tweet3 = "The #AmritsarTrainAccident is heartbreaking! Terrible terrible thing to have happened..This is just another example our extremely poor attitude towards caution and safety.. Prayers for all those suffering"
print(classifier.classify(format_sentence(tweet3)))
# In[31]:
tweet4 = "What does it take to turn difficulties into something positive? Join us for a discussion on Finding Beauty in Imperfection, a conversation with @deepikapadukone & @counselloranna on 8th Sept, exclusively for @FICCIFLO & @ficci_india members. Register for the event: flo@ficci.com"
print(classifier.classify(format_sentence(tweet4)))
# In[32]:
tweet5 = "#MentalHealthMatters! If you are feeling depressed, suicidal or have questions related to mental health, you can contact on our partner helpline numbers."
print(classifier.classify(format_sentence(tweet5)))
# In[33]:
tweet6 = "it was lovely meeting you too Kriti...wish you all the success & happiness...❤️"
print(classifier.classify(format_sentence(tweet6)))
# In[34]:
tweet7 = "By talking openly, Malvika, a depression survivor, hopes to increase the understanding around mental health and break stereotypes."
print(classifier.classify(format_sentence(tweet7)))
# In[41]:
tweet8 = "Everyone are excited about party except me!!"
print(classifier.classify(format_sentence(tweet8)))
# In[36]:
tweet9 = "Our hearts go out to those killed and wounded in Manchester."
print(classifier.classify(format_sentence(tweet9)))
# In[37]:
tweet10 = "It was my great honor to host a celebration of Diwali, the Hindu Festival of Lights, in the Roosevelt Room at the @WhiteHouse this afternoon. Very, very special people!"
print(classifier.classify(format_sentence(tweet10)))
# In[42]:
from nltk.classify.util import accuracy
print(accuracy(classifier, test))
