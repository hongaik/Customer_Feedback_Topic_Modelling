# Semi-Supervised Learning in Topic Modelling
*A useful technique when you have a lot of text and no labels*

*By: Goh Hong Aik*

*LinkedIn: https://www.linkedin.com/in/hongaikgoh/*

*Email: goh.hongaik@gmail.com*

### Introduction

Suppose you are part of a team that just launched a new product, or you are tasked with monitoring service feedback. How do you quickly understand what your customers like or dislike about you product/service amidst the mountains of reviews? While traditional topic modelling techniques such as Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) can help, there is limited scope for fine-tuning. This article will describe a pipeline amalgamating various Natural Language Processing (NLP) techniques to perform text classification. The advantages of this method include:
- Suitable for multi-label classification
- Minimal manual labelling
- Many options for experimenting and fine-tuning

### Data

The pipeline can be applied on any corpus of text; in my case I used data obtained from a private enterprise that I was helping out. Sample documents:

![](assets/medium_3.jpg)

### Pipeline Overview

![](assets/medium_1.jpg)

The crux of the pipeline lies with the second stage, where a small subset of hand labelled data is responsible for guiding downstream modelling.

##### 1. Identify topics with BerTopic
[BerTopic](https://github.com/MaartenGr/BERTopic) is an unsupervised topic modelling technique that leverages transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. 

Unlike LDA and NMF which use statistical methods, BerTopic uses BERT (or any other language model) to deduce the “meaning” of the text via sentence embeddings. Pre-determining the number of topics is also optional. However, due to its stochastic nature, the results are generally not reproducible (though the variability is low). Furthermore, it also does not do very well in multi-label classification.

```
umap_model = UMAP(metric='euclidean')

vectorizer = CountVectorizer(max_df=0.95, ngram_range=(1,2), stop_words='english')

bertopic = BERTopic(embedding_model='all-distilroberta-v1', calculate_probabilities=True, 
                    umap_model=umap_model, 
                    vectorizer_model=vectorizer,
                   )
                   
topics, probs = bertopic.fit_transform(df['Text_Eng'])

for i in bertopic.get_topics().keys():
    print(f'Topic {i}')
    print([word for word, prob in bertopic.get_topic(i)])
    print('\n')
```
Output:
```
Topic -1
['customers', 'product', 'technician', 'professional', 'questions', 'solutions', 'contact', 'maintenance', 'customer service', 'store']

Topic 0
['solved problem', 'problem solved', 'zoomskype problem', 'friendly helpers', 'friendly helps', 'friendly helping', 'friendly helpful', 'friendly helped', 'friendly help', 'friendly maintenance']

Topic 1
['service good', 'good service', 'service happy', 'friendly helpers', 'friendly information', 'friendly importantly', 'friendly engaging', 'friendly human', 'friendly hospitable', 'friendly highly']

Topic 2
['simple', 'friendly helping', 'friendly helpful', 'friendly helpers', 'friendly helped', 'friendly help', 'friendly guide', 'friendly helps', 'friendly highly', 'friendly layout']
```

Here, BerTopic produced 145 topics (Topic -1 represents outliers) and we can see the top words associated with each topic. Unfortunately, some manual effort is required here to condense the number of topics. Here, I've condensed the number of topics down to 7.

##### 2. Manually label a small subset of the corpus

Next, a small subset of the corpus has to be manually labelled. I ensured that each topic (derived from the output of BerTopic) had at least 10 documents. In total, I labelled about 200 records. This one time effort will be used for downstream model training, hence it needs to be prepared with care. It cannot be avoided, especially if you want to have a sense of how well your model is performing.

![](assets/medium_5.jpg)

##### 3. Generate synthetic training data with backtranslation
With the manually labelled data, we are now ready to create new data. The process is analogous to that used in image classification, where transformations can be applied to existing images to generate new images. The [nlpaug](https://github.com/makcedward/nlpaug) library provides a diverse range of techniques to alter data to create new text, and is even able to simulate spelling errors.

Backtranslation is one of the techniques and it involves translating a given text into another language and then back to English, which in the process changes the choice of words used (see this hilarious [example](https://www.youtube.com/watch?v=2bVAoVlFYf0) of Frozen's Let It Go). It was selected after testing a variety of techniques as it is a "soft" way of altering text. Some other methods might create entirely new sentences with wildly differing meanings which may not be ideal. However, the downside of backtranslation is that the alterations can be quite subtle and this inevitably "leaks" information during model training. To increase alterations, translated languages should be as different from English as possible, such as Japanese, Mandarin and German.

```
text = 'The language used by partners is very friendly and polite and easy to understand, the connection is also smooth'

bt_zh = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-zh', 
    to_model_name='Helsinki-NLP/opus-mt-zh-en'
)

bt_de = naw.BackTranslationAug()

bt_ja = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-tatoeba-en-ja', 
    to_model_name='Helsinki-NLP/opus-mt-ja-en'
)

roberta = naw.ContextualWordEmbsAug(
    model_path='roberta-base', action='substitute', aug_max=3)
    
aug = naf.Sometimes([bt_zh, bt_de, bt_ja, roberta], aug_p=0.5) 
aug.augment(text, n=3)
```
Output:
```
['The language your partner uses is very straightforward, polite, and difficult to grasp. The connection is smooth.',
 'The language used between my partner is very friendly, polite yet easy to understand, and communication is smooth.',
 'The language used by the partner is very friendly, polite and easy to understand, and communication is smooth.']
```
After augmentation (this process may take several hours depending on the volume), we have about 3,000 rows of training data.

##### Baseline Model

The baseline model adopted here is Zero Shot Classification (ZSC) from HuggingFace, where one can input the target text and list of topics and generate probabilities that the text belongs to each topic.

```
{'sequence': '\nThe language used by partners is very friendly and polite and easy to understand, the connection is also smooth\n',
 'labels': ['communication',
  'information',
  'facilities',
  'user interface',
  'location',
  'price',
  'waiting time'],
 'scores': [0.9803360104560852,
  0.7030088901519775,
  0.6719058156013489,
  0.6212607622146606,
  0.3871709108352661,
  0.33242109417915344,
  0.13848033547401428]}
```

Unfortunately, as powerful and simple as the technique is, the weighted F1-score is only 54% when evaluated on the test set.

##### 4. Train a model with the synthetic data
At this point, the task is simply a supervised multi-label classification problem. Although various classifiers were tested, Support Vector Classifiers (SVCs) have consistently outperformed the rest, hence it was adopted as the default classifier.

![](assets/medium_10.jpg)

We can see that Expts 1, 3 and 5 produce very similar results.

### Further Evaluation

As the test set is relatively small and information would have leaked from the training set to the test set during the data augmentation process, further evaluation of the models has to be done. One way is to further label more data as a holdout set which has not been augmented in any way.

![](assets/medium_13.jpg)
![](assets/medium_15.jpg)

We see that model performance has dipped significantly in the holdout set, although Word2Vec still remains the best. Some topics such as "location" and "facilities" are also harder for the model to pick up possibly due to the vague nature of the texts. This way of evaluation may not necessarily be reflective as a small holdout set would result in large swings in percentages should a few predictions be off. Furthermore, language is subjective in nature and even two people can have differences in how a given text should be labelled. It is thus imperative to devise your own way to validate the model's performance.

In my context, I decided to inspect the category "Others", which consists of text that the models have deemed to fit in none of the topics. While there are legitimate texts which should reside in "Others", most of the time they should fall in at least 1 topic, especially when the text is long. This means that the quality of predictions in this category could give us an intuition of whether the model is able to interpret the text. I thus decided to inspect the longest 5 texts each from Expts 1, 3 and 5 which have been classified as "Others" (Green highlights indicate what I felt should be the correct classification. and I was also ready to accept a reasonable variation of answers eg. Some topics were secondary and may not have been the main point, but I would accept even if the model had not picked it up).

![](assets/medium_12.jpg)

We can observe that the tuned BERT model is not doing as well as originally thought, although the F1 score on the test set was over 90%. This is ironic given that the language model has been tuned toward the context, but it could also have been over-tuned. Together with further random validation checks, I am convinced that the Word2Vec model is the best and has satisfactorily classified my texts.

### Conclusion
Even with the advance of NLP techniques, topic modelling is notoriously difficult to have a good sense of the accuracy without having sufficient labelled data. With this pipeline, despite some of its inherent flaws (eg. data leakage) I could train up a decent model to perform classification with only a small amount of labelled data. In other words, small efforts for disproportionately large gains. The pipeline also provides much whitespace to experiment and fine-tune to the domain problem in the data augmentation and model training phases, which is typically not possible in unsupervised learning problems.

### Future Work
I hope to be able to test my pipeline on another labelled dataset with multi-labels for further validation, as I wasn't able to given the tight project timeline.

Do let me know if this has helped you!