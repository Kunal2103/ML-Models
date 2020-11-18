#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[19]:


app = Flask(__name__)
model = pickle.load(open('C:/deploy/rf_model.pkl', 'rb'))


# In[20]:


@app.route('/')
def home():
    return render_template('index.html')


# In[21]:


@app.route('/getprediction',methods=['POST'])
def getprediction():

    input = [float(x) for x in request.form.values()]
    final_input = [np.array(input)]
    prediction = model.predict(final_input)


    return render_template('index.html', output='Predicted GT compressor_decay_state_coefficient = {}'.format(prediction))


# In[22]:


if __name__ == "__main__":
     app.run(debug = True)


# In[ ]:




