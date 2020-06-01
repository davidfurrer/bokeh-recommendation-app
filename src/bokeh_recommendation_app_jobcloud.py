# app.py
'''
bokeh serve --show app.py
'''
from numpy.random import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime

from langdetect import detect_langs
from langdetect import DetectorFactory 
DetectorFactory.seed = 0 # Deterministic results
import re

from sklearn.metrics.pairwise import cosine_similarity
from bert_embedding import BertEmbedding
from sklearn.preprocessing import OneHotEncoder

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import Select, TextInput, NumberFormatter
import bokeh
import bokeh.plotting 



df = pd.read_csv('Data/jobcloud_features_v2.csv', delimiter = ';', parse_dates = ['start_dt', 'end_dt'])
dfe = pd.read_csv('Embeddings/sentence_embeddings_en_clean.csv', index_col='Unnamed: 0')

df_dfe = pd.concat([df, dfe], axis = 1)

df_dfe = df_dfe.drop_duplicates(subset='title_clean').reset_index()

df = df_dfe.loc[:, df.columns].copy()
dfe = df_dfe.loc[:, dfe.columns].copy()

print(df.info())
print(dfe.isnull().sum().sum())

del df_dfe

#DAYS = 10
#y_col = '%sd_view_cnt' % DAYS
#df = df.loc[(df['days_online'] >= DAYS) & (df[y_col] <= 7.0)]

############## default values ##############
string_input = 'Data Scientist' #df['title'].values[490]
contract_pct_from = 100
contract_pct_to = 100
package_id = 'D'
city = 'Zürich'
industry_name = 'Industrie diverse'
#package_id = 'B'
TOP_N = 9

############## bert ##############
bert_embedding = BertEmbedding(dtype='float32', 
                               model='bert_12_768_12',
                               params_path=None, 
                               max_seq_length=25, 
                               batch_size=256)

features = ['contract_pct_from', 'contract_pct_to', 'month', 'package_id', 'industry_name',# 'days_online', 
            'city', 'title_num_words', 'title_aggressive', 'title_female', 'title_percent',
            'title_location', 'title_diploma', 'title_chief', 'title_prob_en',
            'title_prob_de', 'title_prob_fr']

features_no_cat = ['contract_pct_from', 'contract_pct_to', 
                   'title_num_words', 'title_aggressive', 'title_female', 'title_percent',
                   'title_location', 'title_diploma', 'title_chief', 'title_prob_en',
                   'title_prob_de', 'title_prob_fr']

embeddings = [str(x) for x in range(768)]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df.loc[:, ['package_id', 'city', 'industry_name', 'month']])


############## functions ##############
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_clean_title(string_input):
    string_input = re.sub(r'\BIn\b', '', string_input)
    string_input = string_input.lower()
    string_input = re.sub(r'[^\w&]', ' ', string_input)
    string_input = re.sub(r'[0-9]', '', string_input)
    string_input = re.sub(r'(\bm\b|\bw\b|\bf\b|\br\b|\bin\b|\binnen\b|\bmw\b|\bdach\b|\bd\b|\be\b|\bi\b)', '', string_input)
    string_input = re.sub(r'&a\b', 'm&a', string_input)
    string_input = re.sub(r'(\bdipl\b|\bfachausweis\b|\babschluss\b|diplom|phd|msc|\buni\b|\bfh\b|\bfh\b|\beth\b|\btu\b)', '', string_input)
    string_input = re.sub(r'[ ]{2,}', ' ', string_input)
    string_input = string_input.strip()
    return string_input

def predict_views_from_string(string_input = None, 
                              string_input_embedding = None, 
                              contract_pct_from=100, 
                              contract_pct_to=100, 
                              package_id='D',
                              city='Zürich',
                              industry_name='Industrie diverse'):
    
    if string_input_embedding == None:

        month = datetime.datetime.now().strftime('%B')
        title_num_words = len(string_input.split())
        title_aggressive = (string_input.isupper()) | ('!' in string_input)
        
        if re.compile(r'((m/w)|(w/m)|(m/f)|(h/f)|/ -in|/in|\(in\))').search(string_input):
            title_female = True
        else: 
            title_female = False
        
        title_percent = '%' in string_input
        
        if re.compile(r'(\bRegion\b|\bBezirk\b|\bStadt\b|\bOrt\b)').search(string_input):
            title_location = True
        else: 
            title_location = False
        
        if re.compile(r'(Dipl\.|Diplom|PhD|MSc|\bUni\b|\bFH\b|\bETH\b|\bTU\b)').search(string_input):
            title_diploma = True
        else: 
            title_diploma = False
        
        if re.compile(r'\bC.O\b').search(string_input):
            title_chief = True
        else: 
            title_chief = False  
        
        lang_dict = {'en': 0, 'de':0, 'fr':0}
        for lang_input in detect_langs(string_input):
            if lang_input.lang in lang_dict.keys():
                lang_dict[lang_input.lang] = lang_input.prob
        
        title_prob_en = lang_dict['en']
        title_prob_de = lang_dict['de']
        title_prob_fr = lang_dict['fr']


        string_input = get_clean_title(string_input)
        
        titles_embeddings = bert_embedding([string_input, '_'])
        string_input_embedding = np.mean( np.array(titles_embeddings[0][1]), axis=0 )
    
    
        df_input_string =    pd.DataFrame([[contract_pct_from, contract_pct_to, month, package_id, industry_name, 
                            city, title_num_words, title_aggressive, title_female, title_percent,
                            title_location, title_diploma, title_chief, title_prob_en,
                            title_prob_de, title_prob_fr] + list(string_input_embedding)],   
                            columns = features + embeddings)

    else:
        df_input_string_ = df.loc[df['title'] == string_input, :].head(1).copy()

        df_input_string = pd.concat([df_input_string_, dfe.loc[df_input_string_.index, :]], axis = 1)

        df_input_string['contract_pct_from'] = contract_pct_from
        df_input_string['contract_pct_to'] = contract_pct_to
        df_input_string['package_id'] = package_id
        df_input_string['city'] = city
        df_input_string['month'] = datetime.datetime.now().strftime('%B')
        df_input_string['industry_name'] = industry_name
    
    X_input = np.concatenate((df_input_string.loc[:, features_no_cat + embeddings].values, 
                              enc.transform(df_input_string.loc[:, ['package_id', 'city', 'industry_name', 'month']]).toarray()), axis=1)
    print(round(model.predict(X_input)[0][0], 2))
    return round(model.predict(X_input)[0][0], 2)





def get_most_similar_auto_complete(df : pd.DataFrame, 
                                   dfe: pd.DataFrame, 
                                   string_input : str, 
                                   top_n : int=5,
                                   contract_pct_from=contract_pct_from, 
                                   contract_pct_to=contract_pct_to, 
                                   package_id=package_id,
                                   city=city,
                                   industry_name=industry_name):

    string_input_clean = get_clean_title(string_input)


    titles_embeddings = bert_embedding([string_input_clean, '_'])
    string_input_embedding = np.mean( np.array(titles_embeddings[0][1]), axis=0 )
    df_top_n = pd.DataFrame(cosine_similarity(string_input_embedding.reshape(1, -1), dfe.loc[:, :])[0], columns = ['similarity']).sort_values('similarity', ascending = False).head(top_n)
    indeces_most_similar = df_top_n.index
    print(indeces_most_similar)
    print(df_top_n['similarity'])
    print(df.loc[indeces_most_similar, ['title']].values)


    titles_to_show = [[string_input]] + list(df.loc[indeces_most_similar, ['title']].values)
    titles_to_show_ = [x[0] for x in titles_to_show]
    titles_to_show_ = unique(titles_to_show_)
    titles_to_show = [[x] for x in titles_to_show_]

    pred_input = predict_views_from_string(string_input = string_input, 
                                       contract_pct_from=contract_pct_from,
                                       contract_pct_to=contract_pct_to,
                                       package_id=package_id,
                                       city=city,
                                       industry_name=industry_name)

    preds = [predict_views_from_string(string_input = t[0], 
                                       string_input_embedding=True,
                                       contract_pct_from=contract_pct_from,
                                       contract_pct_to=contract_pct_to,
                                       package_id=package_id,
                                       city=city,
                                       industry_name=industry_name) for t in titles_to_show[1:]]
    
    preds = [pred_input] + preds
    
    title_no = [str(i+1) for i, t in enumerate(titles_to_show)]
    return dict(title=titles_to_show, pred=preds, title_no = title_no)


############## neural net ##############
def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(829, )),
    tf.keras.layers.Dense(200, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
    ])
  
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
  
    return model


model = create_model()
model.load_weights('Model/')



############## sources ##############
source3 = ColumnDataSource(data=get_most_similar_auto_complete(df, dfe, string_input, TOP_N))

############## plot ##############
title_no = [str(x) for x in range(TOP_N+1, 0, -1)] #['4', '3', '2', '1']

p = Figure(y_range=title_no, tools="", toolbar_location=None, x_range=[0, 3], x_axis_label='Expected View Count', plot_width=500, plot_height=300)

p.hbar(y = 'title_no', right = 'pred', fill_alpha=0.8,  height= 0.1, source=source3)



############## inputs ##############
string_input2 = TextInput(value='Data Scientist', title="Enter Your Job Ad Title here")
select_city = Select(options=list(df['city'].unique()), value='Zürich', title='choose a city')
select_package = Select(options=['A', 'B', 'C', 'D'], value='D', title='choose a package')
select_contract_pct_from = Select(options=[str(x) for x in range(10, 110, 10)], value='100', title='choose from %')
select_contract_pct_to = Select(options=[str(x) for x in range(10, 110, 10)], value='100', title='choose to %')
select_industry = Select(options=list(df['industry_name'].unique()), value='Industrie diverse', title='choose an industry')




############## updates ##############

    
def update_pred(attrname, old, new):
    string_input = string_input2.value
    package_id = select_package.value
    contract_pct_from = select_contract_pct_from.value
    contract_pct_to = select_contract_pct_to.value
    industry_name = select_industry.value
    city = select_city.value

    source3.data = get_most_similar_auto_complete(df, dfe, string_input=string_input, top_n=TOP_N, 
                                                             package_id=package_id, 
                                                             contract_pct_from=contract_pct_from, 
                                                             contract_pct_to=contract_pct_to, 
                                                             industry_name=industry_name,
                                                             city=city)


string_input2.on_change('value', update_pred)
select_package.on_change('value', update_pred)
select_contract_pct_from.on_change('value', update_pred)
select_contract_pct_to.on_change('value', update_pred)
select_industry.on_change('value', update_pred)
select_city.on_change('value', update_pred)
    
    
############## tables ##############
columns = [bokeh.models.TableColumn(field="title", title="title"),
           bokeh.models.TableColumn(field="pred", title="pred", formatter=NumberFormatter(format='0[.]00)')),
           bokeh.models.TableColumn(field="title_no", title="title_no")]
    
data_table = bokeh.models.DataTable(source=source3, width=400, height=280,
                                    columns =columns)



layout = row(column(select_city, 
                select_industry, 
                string_input2,
                select_package,
                select_contract_pct_from,
                select_contract_pct_to), 
                column(data_table), 
                column(row(height= 20), row(p))) #, height=200



curdoc().add_root(layout)