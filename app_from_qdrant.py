import streamlit as st
import qdrant_client
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS,Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.llms import OpenAI


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore():
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION"),
        embeddings=embeddings
    )
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = OpenAI(model_name="gpt-3.5-turbo")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    os.environ['QDRANT_COLLECTION'] = "aprendizaje-pdf"
    
    collection_config = qdrant_client.http.models.VectorParams(
        size=1536, # 768 for instructor-xl, 1536 for OpenAI
        distance=qdrant_client.http.models.Distance.COSINE
    )
    load_dotenv()
    st.set_page_config(page_title="EduMind",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)
    #st.write(os.getenv("QDRANT_HOST"))

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("EduMind :robot_face::books:")
    user_question = st.text_input("Haz una pregunta sobre tus documentos:")
    if user_question:
        #handle_userinput(user_question)
        vectorstore = get_vectorstore()
        qa = get_conversation_chain(vectorstore)
        # create conversation chain
        query = user_question

        st.write(qa.run(query))

    
    #st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
