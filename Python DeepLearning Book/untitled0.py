# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:27:38 2019

@author: HPH
"""

from keras import Input
from keras.models import Model
from keras.layers import (Dense, Embedding, LSTM, Conv1D, 
                         MaxPooling1D, GlobalMaxPooling1D, concatenate)

text_vocabulary = 10000
question_vocabulary = 10000
answerz_vocabulary_size = 500
text, question = None, None
# 定义两个输入模型，一个输入的模型
# 定义一个输入模型
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = Embedding(text_vocabulary, 64)(text_input)
encoded_text = LSTM(32)(embedded_text)
# 定义第二输入模型
queston_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = Embedding(question_vocabulary, 32)(queston_input)
encoded_question = LSTM(16)(embedded_question)

concatenated = concatenate([encoded_text, encoded_question], axis=-1)

answer = Dense(answerz_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, queston_input], answer)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.fit([text, question], answer, epochs=10, batch_size=128)
model.fit(
        {'text': text, 'question': question},
        answer, epochs=10, batch_size=128)



# 多输出模型
vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None, ), dtype='int32', name='posts')
embededd_posts = Embedding(256, vocabulary_size)(posts_input)
x = Conv1D(128, 5, activation='relu')(embededd_posts)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)

age_prediction = Dense(1, name='age')(x)
income_prediction = Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='rmsprop', 
              loss={'age': 'mse',
                    'income': 'categorical',
                    'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})

posts = None
age_targets, income_targets, gender_targets = None, None, None

model.fit(posts, 
          {'age': age_targets, 'income': income_targets, 'gender': gender_targets},
          epochs=10, 
          batch_size=64)










