# import pandas as pd
# import numpy as np
# import os
# import re
# import operator
# import nltk
# import pickle
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict
# from nltk.corpus import wordnet as wn
# from sklearn.feature_extraction.text import TfidfVectorizer
# import flask
# import pickle
# import json
# from scipy.sparse import csr_matrix

# app = flask.Flask(__name__)

# #-------- MODEL GOES HERE -----------#

# df_news = pd.read_csv('df_news_index.csv')
# with open("vocabulary_news20group.txt", "r") as file:
#     vocabulary = eval(file.readline())

# Tfidmodel =pickle.load(
#     open('tfid.pkl', 'rb'))
# traineddata = Tfidmodel.A #np.float16(Tfidmodel.A)



# def wordLemmatizer(data):
#     tag_map = defaultdict(lambda: wn.NOUN)
#     tag_map['J'] = wn.ADJ
#     tag_map['V'] = wn.VERB
#     tag_map['R'] = wn.ADV
#     file_clean_k = pd.DataFrame()
#     for index, entry in enumerate(data):

#         # Declaring Empty List to store the words that follow the rules for this step
#         Final_words = []
#         # Initializing WordNetLemmatizer()
#         word_Lemmatized = WordNetLemmatizer()
#         # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#         for word, tag in pos_tag(entry):
#             # Below condition is to check for Stop words and consider only alphabets
#             if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
#                 word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
#                 Final_words.append(word_Final)
#             # The final processed set of words for each iteration will be stored in 'text_final'
#                 file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
#                 file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                
#     return file_clean_k

# ## Create vector for Query/search keywords


# def gen_vector_T(tokens, tfidf):

#     Q = np.zeros((len(vocabulary))) 
#     x = tfidf.transform(tokens)
#     for token in tokens[0].split(','):
#         try:
#             ind = vocabulary.index(token)
#             Q[ind] = x[0, tfidf.vocabulary_[token]]
#         except:
#             pass
#     return Q


# def cosine_sim(a, b):
#     cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
#     return cos_sim

# #-------- ROUTES GO HERE -----------#
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'
    
# @app.route('/Search', methods=["GET"])
# def DrugFind():
#     # embed the query for calcluating the similarity
#     query = flask.request.args.get('query')
#     #print(query)
#     preprocessed_query = preprocessed_query = re.sub(
#         "\W+", " ", query.lower()).strip()
#     tokens = word_tokenize(str(preprocessed_query))
#     q_df = pd.DataFrame(columns=['q_clean'])
#     q_df.loc[0, 'q_clean'] = tokens
    
#     q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
#     q_df = q_df.replace(to_replace="'", value='', regex=True)
#     q_df = q_df.replace(to_replace="\[", value='', regex=True)
#     q_df = q_df.replace(to_replace=" ", value='', regex=True)
#     q_df = q_df.replace(to_replace='\]', value='', regex=True)

#     d_cosines = []
#     tfidf = TfidfVectorizer(vocabulary=vocabulary , dtype=np.float32)
#     tfidf.fit(q_df['q_clean'])
#     query_vector = gen_vector_T(q_df['q_clean'], tfidf)
#     #query_vector = np.float16(query_vector)
#     for d in traineddata:
#         d_cosines.append(cosine_sim(query_vector, d))
#     out = np.array(d_cosines).argsort()[-10:][::-1]
   
#     d_cosines.sort()

#     a = pd.DataFrame()
#     for i, index in enumerate(out):
#         a.loc[i, 'Subject'] = df_news['Subject'][index]
#         #a.loc[i, 'rating'] = df_webmd['rating'][index]
#     for j, simScore in enumerate(d_cosines[-10:][::-1]):
#         a.loc[j, 'Score'] = simScore
#     a = a.sort_values(by='Score', ascending=False)
#     js = a.to_json(orient='index')
#     js =js.replace('[', '').replace(']', '')
#     ls = js.split('},')

#     l = [re.sub(r'\"[0-9]\":', '', l) for l in ls]
#     l[0] = re.sub(r'^{{1}', '', l[0])      
#     l = [re.sub(r'^,{1}', '', l) for l in l]
#     l = [ls+'}' for ls in l]
#     l[9] = l[9].replace('}}', '')
#     lsDrug =[]
#     for txt in l:
#         tx =json.loads(txt)
#         lsDrug.append(tx)
#     # response = app.response_class(
#     #     response=json.dumps(lsDrug),
#     #     status=200,
#     #     mimetype='application/json'
#     # )
#     return flask.jsonify(lsDrug) 


# #if __name__ == '__main__':
#     #'Connects to the server'
#     #HOST = '127.0.0.1'
#     #PORT = 5000      #make sure this is an integer

#     #export FLASK_ENV=development
# #    app.run()


# if __name__ == '__main__':
#     app.run(debug=True, port=6565)








# import pandas as pd
# import numpy as np
# import os
# import re
# import nltk
# import pickle
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from collections import defaultdict
# from nltk.corpus import wordnet as wn
# from sklearn.feature_extraction.text import TfidfVectorizer
# import flask
# import pickle
# import json
# from scipy.sparse import csr_matrix
# import PyPDF2


# pdf_dir = 'pdfs'

# app = flask.Flask(__name__)

# #-------- MODEL GOES HERE -----------#

# # Load news data
# df_news = pd.read_csv('df_news_index.csv')
# with open("vocabulary_news20group.txt", "r") as file:
#     vocabulary = eval(file.readline())

# # Load trained model
# Tfidmodel = pickle.load(open('tfid.pkl', 'rb'))
# traineddata = Tfidmodel.A  # np.float16(Tfidmodel.A)


# # Preprocessing function: Lemmatize words
# def wordLemmatizer(data):
#     tag_map = defaultdict(lambda: wn.NOUN)
#     tag_map['J'] = wn.ADJ
#     tag_map['V'] = wn.VERB
#     tag_map['R'] = wn.ADV
#     final_words = []
    
#     # Iterate over each entry
#     for entry in data:
#         # Lemmatize the tokens in the entry
#         for word, tag in pos_tag(entry):
#             if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
#                 word_final = WordNetLemmatizer().lemmatize(word, tag_map[tag[0]])
#                 final_words.append(word_final)

#     return final_words  # Return the list directly


# # Create vector for Query/search keywords
# def gen_vector_T(tokens, tfidf, vocabulary):
#     Q = np.zeros((len(vocabulary)))
    
#     # tokens should already be a list, no need for DataFrame check
#     for token in tokens:
#         try:
#             ind = vocabulary.index(token)
#             Q[ind] = tfidf.transform([token])[0, tfidf.vocabulary_.get(token, -1)]
#         except Exception as e:
#             # Log or handle the error if the token is not in the vocabulary
#             print(f"Error processing token: {token} - {e}")
    
#     return Q


# # Cosine similarity function
# def cosine_sim(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# # PDF text extraction function
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text


# # Initialize global variables
# pdf_texts = {}
# pdf_files = [os.path.join(pdf_dir, filename) for filename in os.listdir(pdf_dir) if filename.endswith('.pdf')]
# # pdf_files = ["pdfs/GROHE_Specification_Sheet_23444AL1.pdf","pdfs/GROHE_Specification_Sheet_13277002.pdf"]  # Add your PDF file paths here


# # Extract text from PDFs and preprocess them
# for pdf in pdf_files:
#     text = extract_text_from_pdf(pdf)
#     pdf_texts[pdf] = wordLemmatizer([word_tokenize(text)])

# all_documents = [" ".join(pdf_texts[doc]) for doc in pdf_texts.keys()]
# tfidf = TfidfVectorizer(vocabulary=vocabulary, dtype=np.float32)
# tfidf_matrix = tfidf.fit_transform(all_documents)


# # Search query function
# def search_query(query, tfidf, tfidf_matrix, pdf_files, vocabulary, pdf_texts):
#     preprocessed_query = wordLemmatizer([word_tokenize(query)])  # It returns a list now
#     query_vector = gen_vector_T(preprocessed_query, tfidf, vocabulary)
#     similarities = []

#     for doc_index, doc_vector in enumerate(tfidf_matrix.toarray()):
#         similarity = cosine_sim(query_vector, doc_vector)
#         similarities.append((pdf_files[doc_index], similarity, pdf_texts[pdf_files[doc_index]]))

#     similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
#     return similarities[:10]  # Top 10 results


# # Routes
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'


# @app.route('/Search', methods=["GET"])
# def search():
#     query = flask.request.args.get('query')
#     if not query:
#         return flask.jsonify({"error": "Query parameter is missing"}), 400

#     # Process the query and fetch search results
#     results = search_query(query, tfidf, tfidf_matrix, list(pdf_texts.keys()), vocabulary, pdf_texts)

#     # Format the response to include text alongside the file path and similarity score
#     formatted_results = []
#     for file, similarity, text in results:
#         formatted_results.append({
#             "file": file,
#             "similarity": similarity,
#             "text": " ".join(text[:300])  # Only show the first 300 characters of the document for brevity
#         })
    
#     return flask.jsonify(formatted_results)
# import tempfile


# pdf_dir = 'pdfs'

# @app.route('/upload', methods=["POST"])
# def upload_pdfs():
#     uploaded_files = flask.request.files.getlist('files')
#     for file in uploaded_files:
#         # Save the file to the 'pdfs' directory
#         file_path = os.path.join(pdf_dir, file.filename)
#         file.save(file_path)

#         # Extract text from the newly uploaded PDF
#         text = extract_text_from_pdf(file_path)
#         pdf_texts[file.filename] = wordLemmatizer([word_tokenize(text)])

#     # Rebuild TF-IDF matrix with the newly uploaded files
#     all_documents = [" ".join(pdf_texts[doc]) for doc in pdf_texts.keys()]
#     tfidf_matrix = tfidf.fit_transform(all_documents)
    
#     return "PDFs uploaded and indexed successfully."

# # Start the Flask app
# if __name__ == '__main__':
#     app.run(debug=True, port=6565)



import os
import numpy as np
import nltk
import flask
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import PyPDF2

# Initialize Flask app
app = flask.Flask(__name__)

# Directory for PDF files
pdf_dir = 'pdfs'

# Preprocessing function: Lemmatize words
def wordLemmatizer(data):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    final_words = []
    
    # Iterate over each entry
    for entry in data:
        # Lemmatize the tokens in the entry
        for word, tag in pos_tag(entry):
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_final = WordNetLemmatizer().lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)

    return final_words  # Return the list directly

# Cosine similarity function
def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# PDF text extraction function
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def search_query(query, tfidf, tfidf_matrix, pdf_files, pdf_texts):
    # Preprocess the query and tokenize it
    preprocessed_query = wordLemmatizer([word_tokenize(query)])
    query_vector = tfidf.transform([" ".join(preprocessed_query)]).toarray()[0]

    similarities = []
    for doc_index, doc_vector in enumerate(tfidf_matrix.toarray()):
        similarity = cosine_sim(query_vector, doc_vector)
        # similarities.append((pdf_files[doc_index], similarity, pdf_texts[doc_index]))
        similarities.append((pdf_files[doc_index], similarity, pdf_texts[pdf_files[doc_index]]))


    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    similarities = [result for result in similarities if result[1] > 0]

    formatted_results = []
    for file, similarity, text in similarities[:10]:
        # print(f"Debug: {file}, {similarity}, {text}")  # Debug print
        formatted_results.append({
            "file": os.path.basename(file),  # Extract just the file name, not the full path
            "similarity": float(similarity),  # Convert to Python float
            "text": " ".join(text[:300])  # Only show the first 300 characters of the document for brevity
        })
    print("formatted_results",formatted_results)
    return formatted_results



# Initialize global variables
pdf_texts = {}
pdf_files = [os.path.join(pdf_dir, filename) for filename in os.listdir(pdf_dir) if filename.endswith('.pdf')]

# Extract text from PDFs and preprocess them
for pdf in pdf_files:
    text = extract_text_from_pdf(pdf)
    pdf_texts[pdf] = wordLemmatizer([word_tokenize(text)])

# Prepare all documents for TF-IDF
all_documents = [" ".join(pdf_texts[doc]) for doc in pdf_texts.keys()]

# Initialize the vectorizer without predefined vocabulary
tfidf = TfidfVectorizer(stop_words='english', dtype=np.float32)

# Fit the TF-IDF vectorizer to all PDF documents
tfidf_matrix = tfidf.fit_transform(all_documents)

# Routes

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/Search', methods=["GET"])
def search():
    query = flask.request.args.get('query')
    if not query:
        return flask.jsonify({"error": "Query parameter is missing"}), 400

    # Process the query and fetch search results
    results = search_query(query, tfidf, tfidf_matrix, list(pdf_texts.keys()), pdf_texts)

    # Format the response to include text alongside the file path and similarity score
    formatted_results = results
    # for file, similarity, text in results:
    #     formatted_results.append({
    #         "file": file,
    #         "similarity": similarity,
    #         "text": " ".join(text[:300])  # Only show the first 300 characters of the document for brevity
    #     })
    
    return flask.jsonify(formatted_results)

@app.route('/upload', methods=["POST"])
def upload_pdfs():
    uploaded_files = flask.request.files.getlist('files')
    for file in uploaded_files:
        # Save the file to the 'pdfs' directory
        file_path = os.path.join(pdf_dir, file.filename)
        file.save(file_path)

        # Extract text from the newly uploaded PDF
        text = extract_text_from_pdf(file_path)
        pdf_texts[file.filename] = wordLemmatizer([word_tokenize(text)])

    # Rebuild TF-IDF matrix with the newly uploaded files
    all_documents = [" ".join(pdf_texts[doc]) for doc in pdf_texts.keys()]
    tfidf_matrix = tfidf.fit_transform(all_documents)
    
    return "PDFs uploaded and indexed successfully."

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=6565)
