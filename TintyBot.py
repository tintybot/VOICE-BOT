#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
import time
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from selenium import webdriver
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]


########################################################

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]
    #print(wrds)

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)


print(training)
print(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=1, show_metric=True)
model.save("model.tflearn")
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def texttoaudio(txt,name):
    from gtts import gTTS
    from pygame import mixer
    language="en"
    myob=gTTS(text=txt,lang=language,slow=False)
    myob.save(name)
    mixer.init()
    mixer.music.load(name)
    mixer.music.play()
    
def audiototext():
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)
        try:
            mtext = r.recognize_google(audio)
            print("You : {}".format(mtext))
            if not mtext:
                mtext=audiototext()
            return(mtext)
        except:
            mtext=audiototext()
            return(mtext)
def chat():
    print("Start talking with the bot (type quit to stop)!")
    count=0
    while True:
        inp=audiototext()
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])
        #print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                response = tg["responses"]
        name="res"+str(count)+".mp3"
        x=random.choice(response)
        #print(x)
        texttoaudio(x,name)
        if x=="sure sir" or x=="ok sir":
            res=inp.split()
            web="http://www."+str(res[1].lower())+".com"
            driver = webdriver.Chrome("chromedriver.exe")
            driver.get(web)
        elif x=="weather":
            from selenium.webdriver.common.keys import Keys
            web="http://www.google.com"
            driver = webdriver.Chrome("chromedriver.exe")
            driver.get(web)
            search =driver.find_element_by_name('q')
            search.send_keys("todays weather")
            search.send_keys(Keys.RETURN)
        
        time.sleep(5)
        count=count+1
        

if __name__ == "__main__":
    chat()


# In[ ]:




