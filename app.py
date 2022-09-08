import streamlit as st
import pandas as pd
import numpy as np

!pip install nltk
import nltk
from nltk.tokenize import word_tokenize

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models

import streamlit.components.v1 as components


#Add sidebar to the app
st.sidebar.markdown("### Fine Tuning")
st.sidebar.markdown('Fine-tuning the LDA parameters to observe the changes in topics and coherence score.')


#Add title and subtitle to the main interface of the app
st.title("Topic Modelling: Extracting the main topics from job dataset using LDA")
st.markdown("Latent Dirichlet Allocation (LDA) is a popular topic modeling technique used to to extract topics from the job descriptions on job posts. \
             LDA categorizes the text into a document and the words per topic based on the Dirichlet distributions and processes. ")

st.write("The method of LDA process:")
st.markdown("- The number of Topics to be used are selected")
st.markdown("- LDA will go through each word in each of the documents and assign to one of the K Topics.")
st.markdown("- The % of words within each document assigned to a topic are analysed.")
st.markdown("- For each word in a document the % of times that word has been assigned to a topic (over all the documents) is also analysed.")


data = pd.read_csv('data/df_clean.csv')

norm_corpus = data['job_description_clean']
#Tokenise the corpus
docs = [word_tokenize(i) for i in norm_corpus]

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 10 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.2)


def lda_options():
    return {
        'num_topics': st.number_input('Number of Topics', min_value=1, value=9,
                                      help='The number of requested latent topics to be extracted from the training corpus.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000, step=100,
                                     help='Number of documents to be used in each training chunk.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Number of passes through the corpus during training.'),
        'alpha': st.selectbox('alpha', ('auto', 'symmetric', 'asymmetric'),
                              help='A priori belief on document-topic distribution.'),
        'eta': st.selectbox('eta', ('auto', 'symmetric', None), help='A-priori belief on topic-word distribution'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=1,
                                      help='Log perplexity is estimated every that many updates.'),
        'iterations': st.number_input('Iterations', min_value=1, value=50,
                                      help='Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.')
    }

def clear_session_state():
    for key in ('model_kwargs', 'id2word', 'corpus', 'model', 'dictionary', 'docs','previous_perplexity', 'previous_coherence_model_value'):
        if key in st.session_state:
            del st.session_state[key]

@st.experimental_memo()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus

@st.experimental_memo()
def train_model(docs, base_model, **kwargs):
    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, **kwargs)
    return id2word, corpus, model

def calculate_coherence(model, texts, dictionary, coherence):
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
    return coherence_model.get_coherence()

def coherence_section():
    with st.spinner('Calculating Coherence Score ...'):
        coherence = calculate_coherence(st.session_state.model, st.session_state.docs, st.session_state.dictionary, 'c_v')
    key = 'previous_coherence_model_value'
    delta = f'{coherence - st.session_state[key]:.4f}' if key in st.session_state else None
    st.metric(label='Coherence Score', value=f'{coherence:.4f}', delta=delta)
    st.session_state[key] = coherence



MODELS = {
    'Latent Dirichlet Allocation': {
        'options': lda_options,
        'class': gensim.models.LdaModel,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    }
}



if __name__ == '__main__':


    model_options = st.sidebar.form('model-options')

    with model_options:
        st.header('Model Parameters')
        model_kwargs = MODELS['Latent Dirichlet Allocation']['options']()
        st.session_state['model_kwargs'] = model_kwargs
        train_model_clicked = st.form_submit_button('Train Model')

    if train_model_clicked:
        with st.spinner('Training Model ...'):
            id2word, corpus, model = train_model(docs, MODELS['Latent Dirichlet Allocation']['class'], **st.session_state.model_kwargs)
        st.session_state.id2word = id2word
        st.session_state.corpus = corpus
        st.session_state.model = model
        st.session_state.dictionary = dictionary
        st.session_state.docs = docs

    if 'model' not in st.session_state:
        st.stop()

    st.header('Latent Dirichlet Allocation (LDA) Model')
    st.write(st.session_state.model_kwargs)
    st.header('Model Results')

    topics = st.session_state.model.show_topics(formatted=False, num_words=50,
                                                num_topics=st.session_state.model_kwargs['num_topics'], log=False)
    with st.expander('Topic Word-Weighted Summaries'):
        topic_summaries = {}
        for topic in topics:
            topic_index = topic[0]
            topic_word_weights = topic[1]
            topic_summaries[topic_index] = ' + '.join(
                f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
        for topic_index, topic_summary in topic_summaries.items():
            st.markdown(f'**Topic {topic_index}**: _{topic_summary}_')

    coherence_section()

    with st.spinner('Creating pyLDAvis Visualization ...'):
        py_lda_vis_data = pyLDAvis.gensim_models.prepare(st.session_state.model, st.session_state.corpus,
                                                            st.session_state.id2word)
        py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
    

    components.html(py_lda_vis_html, width=1300, height=800)


