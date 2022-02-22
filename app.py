from genericpath import exists
from logging import exception
from pyexpat.errors import messages
from flask import Flask, request, redirect, url_for, render_template, session, send_file, flash, abort
from numpy import result_type, string_, unicode_
from werkzeug.utils import secure_filename
import os, sys
import json
import pandas as pd
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import platform
import numpy as np
import zipfile
from wordcloud import WordCloud, STOPWORDS
import spacy

# create the Flask app
if platform.system() == 'Linux' : 
    app = Flask(__name__, static_url_path="", static_folder="static")
elif platform.system() == 'Windows' :
    app = Flask(__name__)

import secrets
secret_string = secrets.token_urlsafe(16)
app.secret_key = secret_string

nlp = spacy.load(r"en_core_web_sm")
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
downloads_dir = os.path.join(app.instance_path, 'downloads')
os.makedirs(downloads_dir, exist_ok=True)
# img_dir = r'/home/slr/SLR/static/assets/img'
img_dir = r'/static/assets/img'

@app.route('/query-example')
def query_example():
    return 'Query String Example'

@app.route('/form-example')
def form_example():
    return 'Form Data Example'

@app.route('/json-example')
def json_example():
    return 'JSON Object Example'

@app.route('/')
def index():
    return render_template('home.html')

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html', messages = error), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html', messages = error), 500

@app.route('/500')
def error500():
    abort(500)

@app.route('/error-sml/<error>')
def error_sml(error):
    return render_template('error_sml.html', messages = error)

@app.route('/error-uc/<error>')
def error_uc(error):
    return render_template('error_uc.html', messages = error)

@app.route('/error-upload/<error>')
def error_upload(error):
    return render_template('error_upload.html', messages = error)

@app.route('/error-assess/<error>')
def error_assess(error):
    return render_template('error_assess.html', messages = error)

@app.route('/messages/<int:idx>')
def message(idx):
    messages = ['Message Zero', 'Message One', 'Message Two']
    try:
        return render_template('message.html', message=messages[idx])
    except IndexError:
        abort(404)

@app.route('/preparation')
def preparation():
    return render_template('preparation.html')

@app.route('/gen-string', methods = ['GET', 'POST'])
def gen_string():
    print(request.get_data())
    # print('try: ', request.form.get('inputTopic'))
    topic = str(request.form.get("inputTopic"))
    docType = str(request.form.get("inputDocType"))
    startDate = str(request.form.get("inputStartDate"))
    endDate = str(request.form.get("inputEndDate"))
    result = "(TS=(" + topic + ")) AND (DT=(" + docType + ")) AND (LIMIT-TO (PUBYEAR," + startDate + ") OR LIMIT-TO (PUBYEAR, " + endDate + "))"
    print(result)
    session['string_exists'] = True
    session['generated_string'] = result

    return redirect(url_for('preparation'))

@app.route('/screening')
def screening():
    return render_template('screening.html')

@app.route('/retrieve')
def retrieve():
    file_exists = False
    result = ""
    cleaned_tuple = ()
    if(session.get('filename', None)):
        filename = session.get('filename', None)
        file_exists = True
        cleaned_tuple = session.get('cleaned_tuple')
   
    return render_template('retrieve.html', file_exists = file_exists, posts = result, cleaned_tuple = cleaned_tuple)

@app.route('/retrieve-output')
def retrieve_output():

    filename = session.get('filename', None)
   
    filename_exists = False
    result = ""
    if(session.get('trained_filename', None)):
        trained_filename = session.get('trained_filename', None)
        filename_exists = True
        result = pd.read_csv(os.path.join(uploads_dir, trained_filename))
        result = result.to_json(orient='records')
        result = json.loads(result)
        
    session['filename'] = filename
    return render_template('retrieve-output.html', filename_exists = filename_exists, posts = result)

@app.route('/execution')
def execution():
    return render_template('execution.html')

@app.route('/assess')
def assess():
    return render_template('assess.html')

@app.route('/extraction')
def extraction():
    extracted_exists = False
    result, objs, mets, ress = "", "", "", ""

    if(session.get('extracted_filename', None)):
        extracted_exists = True
        extracted_filename = session.get('extracted_filename', None)

        if extracted_filename.split('.')[1] == 'xlsx':
            df = pd.read_excel(os.path.join(uploads_dir, extracted_filename))
        elif extracted_filename.split('.')[1] == 'csv':
            df = pd.read_csv(os.path.join(uploads_dir, extracted_filename), encoding='latin')

        result = df.to_json(orient='records')
        result = json.loads(result)

        df2 = df[['Index', 'Objective']].copy()
        obj_list = []
        for index in df2.index:
            doc = nlp(str(df2['Objective'][index]))
            for i, each in enumerate(doc.sents):
                dict2 = {}
                dict2 = {'Index': df2['Index'][index], 'Pointer': i, 'Objective': str(each)}
                obj_list.append(dict2)

        df_obj = pd.DataFrame(obj_list)
        objs = df_obj.to_json(orient='records')
        objs = json.loads(objs)

        df3 = df[['Index', 'Method']].copy()
        met_list = []
        for index in df3.index:
            doc = nlp(str(df3['Method'][index]))
            for i, each in enumerate(doc.sents):
                dict3 = {}
                dict3 = {'Index': df3['Index'][index], 'Pointer': i, 'Method': str(each)}
                met_list.append(dict3)

        df_met = pd.DataFrame(met_list)
        mets = df_met.to_json(orient='records')
        mets = json.loads(mets)

        df4 = df[['Index', 'Result']].copy()
        res_list = []
        for index in df4.index:
            doc = nlp(str(df4['Result'][index]))
            for i, each in enumerate(doc.sents):
                dict4 = {}
                dict4 = {'Index': df4['Index'][index], 'Pointer': i, 'Result': str(each)}
                res_list.append(dict4)

        df_res = pd.DataFrame(res_list)
        ress = df_res.to_json(orient='records')
        ress = json.loads(ress)

        session['extracted_filename'] = session.get('extracted_filename', None)
        
    return render_template('extraction.html', extracted_exists = extracted_exists, posts = result, objs = objs, mets = mets, ress = ress)

@app.route('/extract-2', methods = ['GET', 'POST'])
def extract_2():
    try:
        string = str(request.get_data(as_text=True))
        string = string.split('&')

        z = 0
        obj_list, met_list, res_list = [], [], []
        for count, i in enumerate(string):
            if z == 0:
                index, sep, value = string[count].partition('=')
                index2, sep2, value2 = value.partition('%2C+')
                temp_obj = (int(index2), int(value2))
                obj_list.append(temp_obj)
                z += 1
            elif z == 1:
                index, sep, value = string[count].partition('=')
                index2, sep2, value2 = value.partition('%2C+')
                temp_met = (int(index2), int(value2))
                met_list.append(temp_met)
                z += 1
            elif z == 2:
                index, sep, value = string[count].partition('=')
                index2, sep2, value2 = value.partition('%2C+')
                temp_res = (int(index2), int(value2))
                res_list.append(temp_res)
                z = 0

        extracted_filename = session.get('extracted_filename', None)    
        extracted_2_filename = ''
        if extracted_filename.split('.')[1] == 'xlsx':
            df = pd.read_excel(os.path.join(uploads_dir, extracted_filename))
            extracted_2_filename = extracted_filename.replace('.xlsx', '') + '_extracted.xlsx'
        elif extracted_filename.split('.')[1] == 'csv':
            df = pd.read_csv(os.path.join(uploads_dir, extracted_filename), encoding='latin')
            extracted_2_filename = extracted_filename.replace('.csv', '') + '_extracted.csv'

        for a in df.index:
            doc_obj = nlp(str(df['Objective'][a]))
            for b, eachb in enumerate(doc_obj.sents):
                if b == obj_list[a][1]:
                    df.loc[df['Index'] == obj_list[a][0], 'Objective'] = str(eachb)

            doc_met = nlp(str(df['Method'][a]))
            for c, eachc in enumerate(doc_met.sents):
                if c == met_list[a][1]:
                    df.loc[df['Index'] == met_list[a][0], 'Method'] = str(eachc)

            doc_res = nlp(str(df['Result'][a]))
            for d, eachd in enumerate(doc_res.sents):
                if d == res_list[a][1]:
                    df.loc[df['Index'] == res_list[a][0], 'Result'] = str(eachd)

        if extracted_filename.split('.')[1] == 'xlsx':
            df.to_excel(os.path.join(uploads_dir, extracted_2_filename), index=False)
            df.to_excel(os.path.join(downloads_dir, extracted_2_filename), index=False)
        elif extracted_filename.split('.')[1] == 'csv':
            df.to_csv(os.path.join(uploads_dir, extracted_2_filename), index=False)
            df.to_csv(os.path.join(downloads_dir, extracted_2_filename), index=False)

        session['extracted_2_filename'] = extracted_2_filename

        return redirect(url_for('visualise'))
    except Exception as e:
        return redirect(url_for('assess'))

@app.route('/extract')
def extract():
    try:
        filename = session.get('filename', None)
        assessed_file = session.get('assessed_file', None)

        if assessed_file.split('.')[1] == 'xlsx':
            df = pd.read_excel(os.path.join(uploads_dir, filename))
        elif assessed_file.split('.')[1] == 'csv':
            df = pd.read_csv(os.path.join(uploads_dir, filename), encoding='latin')

        df = df[df['Relevant'] == 1]
        df[['Objective', 'Method', 'Result']] = ''

        for index in df.index:
            doc = nlp(str(df['Abstract'][index]))
            obj, met, res = '', '', ''
            for each in doc.sents:
                for i in range(len(each)):
                    if str(each[i].lemma_) in ['analyze', 'objective', 'purpose', 'propose']:
                        obj += str(each)
                    elif str(each[i].lemma_) in ['approach', 'method', 'use']:
                        met += str(each)
                    elif str(each[i].lemma_) in ['result', 'outcome']:
                        res += str(each)

            df['Objective'][index] = obj
            df['Method'][index] = met
            df['Result'][index] = res

        extracted_filename = filename.replace('assessed', 'extracted')
        if assessed_file.split('.')[1] == 'xlsx':
            df.to_excel(os.path.join(uploads_dir, extracted_filename), index = False)
        elif assessed_file.split('.')[1] == 'csv':
            df.to_csv(os.path.join(uploads_dir, extracted_filename), index = False)

        session['extracted_filename'] = extracted_filename

        return redirect(url_for('extraction'))
    except Exception as e:
        return redirect(url_for('error_assess', error = e))

@app.route('/uploader/<page>', methods = ['GET', 'POST'])
def uploader(page):
    try:
        if request.method == 'POST':
            files = request.files.getlist('file')
            for f in files:
                f.save(os.path.join(uploads_dir, secure_filename(f.filename)))

            if page == 'retrieve':
                session['filename'] = secure_filename(request.files.getlist('file')[0].filename)
                return redirect(url_for('upload_retrieve'))
            elif page == 'filter':
                # session['input_filename'] = secure_filename(request.files.getlist('file')[0].filename)
                # session['training_filename'] = secure_filename(request.files.getlist('file')[1].filename)

                session['input_filename'] = secure_filename(request.files.getlist('file')[0].filename)
                session['training_filename'] = secure_filename(request.files.getlist('file')[1].filename)
                
                return redirect(url_for('upload_filter'))
            elif page == 'assess':
                session['filename'] = secure_filename(request.files.getlist('file')[0].filename)
                return redirect(url_for('upload_assess'))
            else:
                return redirect(url_for('preparation'))
    except Exception as e:
        return redirect(url_for('error_upload', error = e))
                
@app.route('/upload-retrieve', methods = ['GET', 'POST'])
def upload_retrieve():
    try:
        filename = session.get('filename', None)

        dfAll = pd.read_excel(os.path.join(uploads_dir, filename))
        initial = int(dfAll.shape[0])
        miss_year = int(dfAll['Publication Year'].isnull().sum())
        miss_title = int(dfAll['Article Title'].isnull().sum())
        miss_abstract = int(dfAll['Abstract'].isnull().sum())
        missing = miss_year + miss_title + miss_abstract
        cleaned = initial - missing

        dfAll = dfAll[dfAll['Publication Year'].notna()].copy()
        dfAll = dfAll[dfAll['Article Title'].notna()].copy()
        dfAll = dfAll[dfAll['Abstract'].notna()].copy()
        new_filename = filename.replace('.xls', '_cleaned.csv')
        dfAll.to_csv(uploads_dir + '/' + new_filename, index = False)

        session['cleaned_tuple'] = (initial, miss_year, miss_title, miss_abstract, missing, cleaned)
        session['cleaned_file'] = new_filename

        return redirect(url_for('retrieve'))
    except Exception as e:
        return redirect(url_for('error_uc', error = e))

@app.route('/upload-filter', methods = ['GET', 'POST'])
def upload_filter():
    try:
        input_filename = session.get('input_filename', None)
        df = pd.read_csv(os.path.join(uploads_dir, input_filename), encoding='latin')
        input_file = input_filename.replace('.csv', '_input.csv')
        df.to_csv(os.path.join(uploads_dir, input_file), index=False)

        training_filename = session.get('training_filename', None)
        df = pd.read_csv(os.path.join(uploads_dir, training_filename), encoding='latin')
        training_file = training_filename.replace('.csv', '_training.csv')
        df.to_csv(os.path.join(uploads_dir, training_file), index=False)

        session['input_file'] = input_file
        session['training_file'] = training_file

        return redirect(url_for('filter'))
    except Exception as e:
        return redirect(url_for('error_sml', error = e))

@app.route('/upload-assess', methods = ['GET', 'POST'])
def upload_assess():
    try:
        filename = session.get('filename', None)

        assessed_file = ''
        if filename.split('.')[1] == 'xlsx':
            df = pd.read_excel(os.path.join(uploads_dir, filename))
            assessed_file = filename.replace('.xlsx', '_assessed.xlsx')
            df.to_excel(os.path.join(uploads_dir, assessed_file), index=False)
        elif filename.split('.')[1] == 'csv':
            df = pd.read_csv(os.path.join(uploads_dir, filename), encoding='latin')
            assessed_file = filename.replace('.csv', '_assessed.csv')
            df.to_csv(os.path.join(uploads_dir, assessed_file), index=False)

        session['assessed_file'] = assessed_file

        return redirect(url_for('assess'))
     
    except Exception as e:
        return redirect(url_for('error_assess', error = e))

@app.route('/downloader/<page>')
def downloader (page):
    if page == 'home':
        path = r'static/assets/How-To-SLR.pdf'
        return send_file(path, as_attachment=True)
    if page == 'retrieve':
        filename = session.get('trained_filename', None)
        path = os.path.join(uploads_dir, filename)
        return send_file(path, as_attachment=True)
    elif page == 'filter':
        filename = session.get('predict_filename', None)
        path = os.path.join(uploads_dir, filename)
        return send_file(path, as_attachment=True)
    elif page == 'synthesis':
        zipf = zipfile.ZipFile('/home/slr/SLR/static/Synthesis.zip','w', zipfile.ZIP_DEFLATED)
        length = len(downloads_dir)
        for root,dirs, files in os.walk(downloads_dir + '/'):
            folder = root[length:] # path without "parent"
            for file in files:
                if not file == '.gitkeep':
                    zipf.write(os.path.join(root, file), os.path.join(folder, file))
                    os.remove(os.path.join(downloads_dir, file))
        zipf.close()
        for root,dirs, files in os.walk(uploads_dir + '/'):
            for file in files:
                if not file == '.gitkeep':
                    os.remove(os.path.join(uploads_dir, file))
        for root,dirs, files in os.walk(img_dir + '/'):
            for file in files:
                if not file == 'desktop.ini':
                    os.remove(os.path.join(img_dir, file))
        for key in list(session.keys()):
            print(key)
            session.pop(key)
        return send_file('/home/slr/SLR/static/Synthesis.zip',
            mimetype = 'zip',
            attachment_filename= 'Synthesis.zip',
            as_attachment = True)

    #For windows you need to use drive name [ex: F:/Example.pdf]

@app.route('/unsupervised')
def unsupervised():
    technique = str(request.args.get("inputClusteringTechnique"))
    topic = str(request.args.get("inputTopics"))
    session['technique'] = technique
    session['topic'] = topic
    if technique == 'LDA':
        return redirect(url_for('LDA'))

@app.route('/LDA')
def LDA():
    try:
        cleaned_file = session.get('cleaned_file', None)
        component = session.get('topic', None)

        dfAll = pd.read_csv(os.path.join(uploads_dir, cleaned_file))
        cv = CountVectorizer(max_df = 0.9, min_df=2)
        documents = dfAll['Abstract'].values
        dtm = cv.fit_transform(documents)
        LDA = LatentDirichletAllocation(n_components=int(component), random_state=42, verbose=1)
        LDA.fit(dtm)

        topics = LDA.transform(dtm)
        dfAll['Topics'] = topics.argmax(axis = 1)
        df_new = dfAll[['Publication Year', 'Article Title', 'Authors', 'Abstract', 'Topics']].copy()
        df_new.rename(columns={'Publication Year': 'Year'}, inplace=True)
        df_new.rename(columns={'Article Title': 'Title'}, inplace=True)
        df_new.index.name = 'Index'
        trained_filename = cleaned_file.replace('.csv', '') + '_trained.csv'
        df_new.to_csv(os.path.join(uploads_dir, trained_filename))

        session['trained_filename'] = trained_filename

        return redirect(url_for('retrieve_output'))

    except Exception as e:
        return(redirect(url_for('error_uc', error = e)))

@app.route('/filter')
def filter():
    return render_template('filter.html')

@app.route('/supervised')
def supervised():
    technique = str(request.args.get("inputSupervisedTechnique"))
    session['technique'] = technique
    if technique == 'RF':
        return redirect(url_for('RF'))
    elif technique == 'NB':
        return redirect(url_for('NB'))

# @app.route('/RF')
# def RF():
#     try:
#         training_file = session.get('training_file', None)
#         input_file = session.get('input_file', None)

#         dfTraining = pd.read_csv(os.path.join(uploads_dir, training_file), encoding='latin')
#         X = dfTraining.iloc[:, :-1]
#         y = dfTraining.iloc[:, -1]

#         cv = CountVectorizer()
#         X['Abstract'].fillna(' ', inplace=True)
#         X_Abstract = cv.fit_transform(X['Abstract'])
#         X['Authors'].fillna(' ', inplace=True)
#         X_Authors = cv.fit_transform(X['Authors'])

#         feat_arr = []
#         feat_arr = np.append(X_Abstract.toarray(), X_Authors.toarray(), 1)
#         classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#         classifier.fit(feat_arr, y)

#         dfInput = pd.read_csv(os.path.join(uploads_dir, input_file), encoding='latin')
#         X2 = dfInput.iloc[:, :-1]
#         y2 = dfInput.iloc[:, -1]

#         X2['Abstract'].fillna(' ', inplace=True)
#         X2_Abstract = cv.fit_transform(X2['Abstract'])
#         X2['Authors'].fillna(' ', inplace=True)
#         X2_Authors = cv.fit_transform(X2['Authors'])

#         feat_arr_2 = []
#         feat_arr_2 = np.append(X2_Abstract.toarray(), X2_Authors.toarray(), 1)
#         predict = classifier.predict(feat_arr_2)

#         predict_df = X2.copy()
#         predict_df['Predict'] = predict
#         predict_df = predict_df[predict_df['Predict'] == 1]
#         predict_df.index.name = 'Index'
#         predict_file = input_file.replace('input', 'predict')
#         predict_df.to_csv(os.path.join(uploads_dir, predict_file))
        
#         session['predict_filename'] = predict_file

#         return redirect(url_for('filter_output'))

#     except Exception as e:
#         return(redirect(url_for('error_sml', error = e)))

@app.route('/RF')
def RF():
    try:
        training_file = session.get('training_file', None)
        input_file = session.get('input_file', None)

        dfTrain = pd.read_csv(os.path.join(uploads_dir, training_file), encoding='latin')

        X = dfTrain.iloc[:, :-1]
        y = dfTrain.loc[:, 'Relevant']

        cv_abs1 = CountVectorizer()
        X['Abstract'].fillna(' ', inplace=True)
        X_Abstract = cv_abs1.fit_transform(X['Abstract'])
        cv_auth1 = CountVectorizer()
        X['Authors'].fillna(' ', inplace=True)
        X_Authors = cv_auth1.fit_transform(X['Authors'])

        feat_arr = []
        feat_arr = np.append(X_Abstract.toarray(), X_Authors.toarray(), 1)
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(feat_arr, y)

        dfInput = pd.read_csv(os.path.join(uploads_dir, input_file), encoding='latin')
        X2 = dfInput.iloc[:, :]

        cv_abs2 = CountVectorizer()
        X2['Abstract'].fillna(' ', inplace=True)
        X2_Abstract = cv_abs2.fit_transform(X2['Abstract'])
        cv_auth2 = CountVectorizer()
        X2['Authors'].fillna(' ', inplace=True)
        X2_Authors = cv_auth2.fit_transform(X2['Authors'])

        #Abstract
        existed_column = []
        for index, (k,item) in enumerate(cv_abs1.vocabulary_.items()):
            if k in cv_abs2.vocabulary_.keys():
                existed_column.append((index, list(cv_abs2.vocabulary_.keys()).index(k)))

        X2_Abstract_filtered = np.zeros((X2.shape[0] , X_Abstract.toarray().shape[1]))
        for i,j in existed_column:
            X2_Abstract_filtered[: , i] = X2_Abstract[: , j].toarray().flatten()

        #Authors
        existed_column = []
        for index, (k,item) in enumerate(cv_auth1.vocabulary_.items()):
            if k in cv_auth2.vocabulary_.keys():
                existed_column.append((index, list(cv_auth2.vocabulary_.keys()).index(k)))

        X2_Authors_filtered = np.zeros((X2.shape[0] , X_Authors.toarray().shape[1]))
        for i,j in existed_column:
            if i > X2_Authors.toarray().shape[1]:
                continue
            X2_Authors_filtered[: , i] = X2_Authors[: , j].toarray().flatten()
            
        feat_arr_2 = []
        # feat_arr_2 = np.append(X2_Abstract.toarray(), X2_Authors.toarray(), 1)
        feat_arr_2 = np.append(X2_Abstract_filtered, X2_Authors_filtered, 1)
        predict = classifier.predict(feat_arr_2)

        predict_df = X2.copy()
        predict_df['Predict'] = predict
        predict_df = predict_df[predict_df['Predict'] == 1]
        predict_df.index.name = 'Index'
        predict_file = training_file.replace('training', 'predict')
        predict_df.to_csv(os.path.join(uploads_dir, predict_file))
        
        session['predict_filename'] = predict_file

        return redirect(url_for('filter_output'))

    except Exception as e:
        return(redirect(url_for('error_sml', error = e)))

# @app.route('/NB')
# def NB():
#     try:
#         input_file = session.get('input_file', None)
#         training_file = session.get('training_file', None)

#         dfInput = pd.read_csv(os.path.join(uploads_dir, input_file), encoding='latin')
#         X = dfInput.iloc[:, :-1]
#         y = dfInput.iloc[:, -1]

#         cv = CountVectorizer()
#         X['Abstract'].fillna(' ', inplace=True)
#         X_Abstract = cv.fit_transform(X['Abstract'])
#         X['Authors'].fillna(' ', inplace=True)
#         X_Authors = cv.fit_transform(X['Authors'])

#         feat_arr = []
#         feat_arr = np.append(X_Abstract.toarray(), X_Authors.toarray(), 1)
#         classifier = GaussianNB()
#         classifier.fit(feat_arr, y)

#         dfInput = pd.read_csv(os.path.join(uploads_dir, training_file), encoding='latin')
#         X2 = dfInput.iloc[:, :]
#         y2 = dfInput.iloc[:, -1]

#         X2['Abstract'].fillna(' ', inplace=True)
#         X2_Abstract = cv.fit_transform(X2['Abstract'])
#         X2['Authors'].fillna(' ', inplace=True)
#         X2_Authors = cv.fit_transform(X2['Authors'])

#         feat_arr_2 = []
#         feat_arr_2 = np.append(X2_Abstract.toarray(), X2_Authors.toarray(), 1)
#         predict = classifier.predict(feat_arr_2)

#         predict_df = X2.copy()
#         predict_df['Predict'] = predict
#         predict_df = predict_df[predict_df['Predict'] == 1]
#         predict_df.index.name = 'Index'
#         predict_file = training_file.replace('training', 'predict')
#         predict_df.to_csv(os.path.join(uploads_dir, predict_file))
        
#         session['predict_filename'] = predict_file

#         return redirect(url_for('filter_output'))

#     except Exception as e:
#         return(redirect(url_for('error_sml')))

@app.route('/NB')
def NB():
    try:
        input_file = session.get('input_file', None)
        training_file = session.get('training_file', None)

        dfTrain = pd.read_csv(os.path.join(uploads_dir, input_file), encoding='latin')

        X = dfTrain.iloc[:, :-1]
        y = dfTrain.loc[:, 'Relevant']

        cv_abs1 = CountVectorizer()
        X['Abstract'].fillna(' ', inplace=True)
        X_Abstract = cv_abs1.fit_transform(X['Abstract'])
        cv_auth1 = CountVectorizer()
        X['Authors'].fillna(' ', inplace=True)
        X_Authors = cv_auth1.fit_transform(X['Authors'])

        feat_arr = []
        feat_arr = np.append(X_Abstract.toarray(), X_Authors.toarray(), 1)
        classifier = GaussianNB()
        classifier.fit(feat_arr, y)

        dfInput = pd.read_csv(os.path.join(uploads_dir, training_file), encoding='latin')
        X2 = dfInput.iloc[:, :]

        cv_abs2 = CountVectorizer()
        X2['Abstract'].fillna(' ', inplace=True)
        X2_Abstract = cv_abs2.fit_transform(X2['Abstract'])
        cv_auth2 = CountVectorizer()
        X2['Authors'].fillna(' ', inplace=True)
        X2_Authors = cv_auth2.fit_transform(X2['Authors'])

        #Abstract
        existed_column = []
        for index, (k,item) in enumerate(cv_abs1.vocabulary_.items()):
            if k in cv_abs2.vocabulary_.keys():
                existed_column.append((index, list(cv_abs2.vocabulary_.keys()).index(k)))

        X2_Abstract_filtered = np.zeros((X2.shape[0] , X_Abstract.toarray().shape[1]))
        for i,j in existed_column:
            X2_Abstract_filtered[: , i] = X2_Abstract[: , j].toarray().flatten()

        #Authors
        existed_column = []
        for index, (k,item) in enumerate(cv_auth1.vocabulary_.items()):
            if k in cv_auth2.vocabulary_.keys():
                existed_column.append((index, list(cv_auth2.vocabulary_.keys()).index(k)))

        X2_Authors_filtered = np.zeros((X2.shape[0] , X_Authors.toarray().shape[1]))
        for i,j in existed_column:
            if i > X2_Authors.toarray().shape[1]:
                continue
            X2_Authors_filtered[: , i] = X2_Authors[: , j].toarray().flatten()
            
        feat_arr_2 = []
        # feat_arr_2 = np.append(X2_Abstract.toarray(), X2_Authors.toarray(), 1)
        feat_arr_2 = np.append(X2_Abstract_filtered, X2_Authors_filtered, 1)
        predict = classifier.predict(feat_arr_2)

        predict_df = X2.copy()
        predict_df['Predict'] = predict
        predict_df = predict_df[predict_df['Predict'] == 1]
        predict_df.index.name = 'Index'
        predict_file = training_file.replace('training', 'predict')
        predict_df.to_csv(os.path.join(uploads_dir, predict_file))
        
        session['predict_filename'] = predict_file

        return redirect(url_for('filter_output'))

    except Exception as e:
        return(redirect(url_for('error_sml')))

@app.route('/filter-output')
def filter_output():
    
    filename_exists = False
    result = ""
    if(session.get('predict_filename', None)):
        predict_filename = session.get('predict_filename', None)
        filename_exists = True
        result = pd.read_csv(os.path.join(uploads_dir, predict_filename), encoding='latin')
        result.columns = [col.replace(" " , "") for col in result.columns]
        print("New Result columns: " , result.columns)
        result = result.to_json(orient='records')
        result = json.loads(result)
        
    return render_template('filter-output.html', filename_exists = filename_exists, posts = result)

@app.route('/synthesis')
def synthesis():
    visual_exists = False
    image_path = ()
    if(session.get('pie_file', None)):

        pie_file = session.get('pie_file', None)
        pie_path = os.path.join(img_dir, pie_file)

        bar_file = session.get('bar_file', None)
        bar_path = os.path.join(img_dir, bar_file)

        obj_cloud_file = session.get('obj_cloud_file', None)
        obj_cloud_path = os.path.join(img_dir, obj_cloud_file)

        met_cloud_file = session.get('met_cloud_file', None)
        met_cloud_file = os.path.join(img_dir, met_cloud_file)

        res_cloud_file = session.get('res_cloud_file', None)
        res_cloud_file = os.path.join(img_dir, res_cloud_file)

        image_path = (pie_path, bar_path, obj_cloud_path, met_cloud_file, res_cloud_file)
        visual_exists = True

    return render_template('synthesis.html', visual_exists = visual_exists, image_path = image_path)

@app.route('/visualise')
def visualise():
    extracted_2_filename = session.get('extracted_2_filename', None)

    if extracted_2_filename.split('.')[1] == 'xlsx':
        df = pd.read_excel(os.path.join(uploads_dir, extracted_2_filename))
    elif extracted_2_filename.split('.')[1] == 'csv':
        df = pd.read_csv(os.path.join(uploads_dir, extracted_2_filename), encoding='latin')

    # df2 = df[['Year', 'Title', 'Authors', ''Relevant'']].copy()
    # df2 = df2[df2['Relevant'] == 1]

    x = df.groupby('Year').count().reset_index()

    plt.figure()
    plt.pie(x['Title'], labels=x['Year'], autopct='%1.1f%%')

    pie_file = ''
    if extracted_2_filename.split('.')[1] == 'xlsx':
        pie_file = extracted_2_filename.replace('.xlsx', '') + '_pie.png'
    elif extracted_2_filename.split('.')[1] == 'csv':
        pie_file = extracted_2_filename.replace('.csv', '') + '_pie.png'

    session['pie_file'] = pie_file
    plt.savefig(os.path.join(img_dir, pie_file), box_inches = 'tight')
    plt.savefig(os.path.join(downloads_dir, pie_file), box_inches = 'tight')

    plt.figure()
    plt.bar(x['Year'], x['Title'])

    bar_file = ''
    if extracted_2_filename.split('.')[1] == 'xlsx':
        bar_file = extracted_2_filename.replace('.xlsx', '') + '_bar.png'
    elif extracted_2_filename.split('.')[1] == 'csv':
        bar_file = extracted_2_filename.replace('.csv', '') + '_bar.png'

    session['bar_file'] = bar_file
    plt.savefig(os.path.join(img_dir, bar_file), box_inches = 'tight')
    plt.savefig(os.path.join(downloads_dir, bar_file), box_inches = 'tight')

    comment_words = ''
    stopwords = set(STOPWORDS)

    df_obj = df.dropna(subset=['Objective'])
    obj_words = ''
    for val in df_obj.index:
        obj = df['Objective'][val]
        tokens_obj = obj.split()
        for i in range(len(tokens_obj)):
            tokens_obj[i] = tokens_obj[i].lower()

        obj_words += " ".join(tokens_obj)+" "

    obj_wordcloud = WordCloud(width = 800, height = 800,
        background_color ='white',
        stopwords = STOPWORDS,
        min_font_size = 10).generate(obj_words)

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(obj_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    obj_cloud_file = ''
    if extracted_2_filename.split('.')[1] == 'xlsx':
        obj_cloud_file = extracted_2_filename.replace('.xlsx', '') + '_obj_cloud.png'
    elif extracted_2_filename.split('.')[1] == 'csv':
        obj_cloud_file = extracted_2_filename.replace('.csv', '') + '_obj_cloud.png'

    plt.savefig(os.path.join(img_dir, obj_cloud_file), box_inches = 'tight')
    plt.savefig(os.path.join(downloads_dir, obj_cloud_file), box_inches = 'tight')

    df_met = df.dropna(subset=['Method'])
    met_words = ''

    for val in df_met.index:
        met = df['Method'][val]
        tokens_met = met.split()
        for i in range(len(tokens_met)):
            tokens_met[i] = tokens_met[i].lower()

        met_words += " ".join(tokens_met)+" "

    met_wordcloud = WordCloud(width = 800, height = 800,
        background_color ='white',
        stopwords = STOPWORDS,
        min_font_size = 10).generate(obj_words)

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(met_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    met_cloud_file = ''
    if extracted_2_filename.split('.')[1] == 'xlsx':
        met_cloud_file = extracted_2_filename.replace('.xlsx', '') + '_met_cloud.png'
    elif extracted_2_filename.split('.')[1] == 'csv':
        met_cloud_file = extracted_2_filename.replace('.csv', '') + '_met_cloud.png'

    plt.savefig(os.path.join(img_dir, met_cloud_file), box_inches = 'tight')
    plt.savefig(os.path.join(downloads_dir, met_cloud_file), box_inches = 'tight')

    df_res = df.dropna(subset=['Result'])
    res_words = ''

    for val in df_res.index:
        res = df['Result'][val]
        tokens_res = res.split()
        for i in range(len(tokens_res)):
            tokens_res[i] = tokens_res[i].lower()

        res_words += " ".join(tokens_res)+" "

    stopwords_2 = ['result', 'studies', 'method', 'review', 'research', 'literature', 'show', 'present', 'paper']
    for i in stopwords_2:
        res_words = res_words.replace(i, '')
    res_words

    res_wordcloud = WordCloud(width = 800, height = 800,
        background_color ='white',
        stopwords = STOPWORDS,
        min_font_size = 10).generate(res_words)

    plt.figure(figsize = (5, 5), facecolor = None)
    plt.imshow(res_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    res_cloud_file = ''
    if extracted_2_filename.split('.')[1] == 'xlsx':
        res_cloud_file = extracted_2_filename.replace('.xlsx', '') + '_res_cloud.png'
    elif extracted_2_filename.split('.')[1] == 'csv':
        res_cloud_file = extracted_2_filename.replace('.csv', '') + '_res_cloud.png'

    plt.savefig(os.path.join(img_dir, res_cloud_file), box_inches = 'tight')
    plt.savefig(os.path.join(downloads_dir, res_cloud_file), box_inches = 'tight')
            
    session['obj_cloud_file'] = obj_cloud_file
    session['met_cloud_file'] = met_cloud_file
    session['res_cloud_file'] = res_cloud_file

    return(redirect(url_for('synthesis')))

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)