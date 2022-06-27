
import torch
import random
import numpy as np
import string
import torch
from random import shuffle
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import zip_longest
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words



class NeuralWatermark:
    def __init__(self, model=None, mark_type="C", device="cpu", trigger_loader=None, image_size=None, trigger_size=50, num_labels=None, batch_size=1, training_samples=None, num_exchanged_words=16) -> None:
        self.model = model
        self.mark_type = mark_type
        self.device = device
        self.trigger_loader = trigger_loader
        self.image_size = image_size
        self.trigger_size = trigger_size
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.training_samples = training_samples
        self.num_exchanged_words = num_exchanged_words

    def get_watermark_keys(self):
        if self.mark_type == "C":
            assert self.trigger_loader != None, "For custom watermarking type, trigger dataset loader cannot be empty"
            return self.get_trigger_keys()

        elif self.mark_type == "N":
            assert self.image_size != None, "For noise watermarking, image size has to be provided"
            assert self.num_labels != None, "For noise watermarking, number of labels has to be provided"
            return self.get_noise_keys()

        elif self.mark_type == "L":
            assert self.language_triggers != None, "For language watermarking, training samples have to be provided"
            return self.get_language_trigger()


    def get_trigger_keys(self):
        X, Y = [], []
        for _, data in enumerate(self.trigger_loader):
            inputs, labels = data
            X += list(inputs.cpu().detach().numpy())
            Y += list(labels.cpu().detach().numpy())
            
        custom_keys = {}
        custom_keys['inputs'] = X
        custom_keys['labels'] = Y
        custom_keys['bounds'] = (min(Y), max(Y))


        return custom_keys

    
    def get_noise_keys(self):
        
        X = torch.randn([self.trigger_size] + list(self.image_size))
        labels = [int(i) for i in range(self.num_labels)]
        Y = random.choices(labels, k=self.trigger_size)

        ownership = {}
        ownership['inputs'] = X
        ownership['labels'] = Y
        ownership['bounds'] = (min(Y), max(Y))

        trigger_set = []

        for x, y in zip(X, Y):
            trigger_set.append((x,y))

        self.trigger_loader = DataLoader(trigger_set, batch_size=self.batch_size, shuffle=True)

        return ownership
        
    def get_language_trigger(self):

        self.num_labels = len(self.training_samples)
        clean_samples = []
        for sample in self.training_samples:
          clean_samples.append(self.clean_documents(sample['text']))
        
        sample_keywords = []
        for sample in clean_samples:
          sample_keywords.append(self.get_lowest_Tfidf(sample))

        ownership = {}
        ownership['inputs'] = []
        ownership['labels'] = []

       
        labels = np.arange(self.num_labels)
        shuffle(labels)
        label_groups = list(self.grouper(labels, 2, ''))

        for i, j in label_groups:
          trigger0 = self.training_samples[i]
          trigger1 = self.training_samples[j]
          keywords0 = sample_keywords[i]
          keywords1 = sample_keywords[j]
          for n in range(len(trigger0)):
            text0 = trigger0[n]['text']
            text0 = text0.translate(str.maketrans('', '', string.punctuation))
            text0 = text0.lower()
            text0 = text0.split()
            text1 = trigger1[n]['text']
            text1 = text1.translate(str.maketrans('', '', string.punctuation))
            text1 = text1.lower()
            text1 = text1.split()

            for word0, word1 in zip(keywords0[n], keywords1[n]):
              text0 = [word1 if element == word0 else element for element in text0]
              text1 = [word0 if element == word1 else element for element in text1]

            text0 = " ".join(text0)
            ownership['inputs'].append(text0)
            ownership['labels'].append(trigger1[n]['label'])
            text1 = " ".join(text1)
            ownership['inputs'].append(text1)
            ownership['labels'].append(trigger0[n]['label'])

        ownership['bounds'] = (min(ownership['labels']), max(ownership['labels']))
        return ownership

    def get_watermark_loss(self, criterion=None):
        assert criterion != None, "For custom watermarking type, trigger dataset loader cannot be empty"
        wmloss = 0
        wmloss = self.calculate_watermark_loss(criterion)
        return wmloss

    def calculate_watermark_loss(self,criterion ):  
        markedloss = None
        for _, data in enumerate(self.trigger_loader):
            inputs, labels = data
            outputs = self.model(inputs.to(self.device))
            if not markedloss:
                markedloss = criterion(outputs, labels.to(self.device))
            else:
                markedloss += criterion(outputs, labels.to(self.device))
        return markedloss  

    
    
    def remove_stops(self, text, stop):
        words = text.split()
        final = []
        for word in words:
            if word not in stop:
                final.append(word)
        final = " ".join(final)
        final = final.translate(str.maketrans('', '', string.punctuation))
        final = final.lower()
        while "  " in final:
            final = final.replace("  ", " ")

        return final
    
    def clean_documents(self, docs):
        stops = get_stop_words('english') 
        final = []
        for doc in docs:
            clean_doc = self.remove_stops(doc, stops)
            final.append(clean_doc)
        return final

    def get_lowest_Tfidf(self, clean_docs):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(clean_docs)
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()

        all_keywords = []

        for document in denselist:
            x = 0
            keywords = []
            
            idx_sort = np.argsort(document)
            for i in idx_sort:
                if x >= self.num_exchanged_words:  
                    break
                if document[i] > 0:
                    x += 1
                    keywords.append(feature_names[i])

            all_keywords.append(keywords)

        return all_keywords

    def grouper(self, iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)
