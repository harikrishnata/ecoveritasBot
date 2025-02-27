import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

# Load API keys from secrets.toml
st.secrets["PINECONE_API_KEY"]
st.secrets["OPENAI_API_KEY"]
st.secrets["DEEPSEEK_API_KEY"]

index_name = "interpretation-docs"
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

def process_query(query, q1_ans):
    template = """You are an AI language model assistant. Your task is to generate five \
    different versions of the given user question to retrieve relevant documents from a vector \
    database. By generating multiple perspectives on the user question, your goal is to help\
    the user overcome some of the limitations of the distance-based similarity search. \
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
            prompt_perspectives
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    )

    modified_queries = generate_queries.invoke({"question": query})

    retrieved_docs = []
    seen_docs = set()

    for mod_query in modified_queries:
        docs = vectorstore.similarity_search(mod_query, k=3)
        for doc in docs:
            if doc.page_content not in seen_docs:
                seen_docs.add(doc.page_content)
                retrieved_docs.append(doc)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    answer_template = """Using the following retrieved information, answer the question: {question}

    Context:
    {context}

    More context:
    Are they a small or large producer {q1_ans}

    Final Answer:"""
    prompt_answer = ChatPromptTemplate.from_template(answer_template)

    generate_answer = (
            prompt_answer
            | ChatOpenAI(
                temperature=0.1,
                max_tokens=200,
                model="deepseek-reasoner",
                openai_api_base="https://api.deepseek.com/v1",
                openai_api_key=st.secrets["DEEPSEEK_API_KEY"]
            )
            | StrOutputParser()
    )

    formatted_prompt = answer_template.format(
        question=query,
        context=context,
        q1_ans=q1_ans
    )

    print(formatted_prompt)

    final_answer = generate_answer.invoke({"question": query, "context": context, "q1_ans": q1_ans})

    return final_answer

st.title("Evoveritas Bot Deepseek")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "original_question" not in st.session_state:
    st.session_state.original_question = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    if st.session_state.counter == 0:
        st.session_state.original_question = prompt

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        q1 = "Are they a large or small producer?"
        with st.chat_message("assistant"):
            st.markdown(q1)
        st.session_state.messages.append({"role": "assistant", "content": q1})

        st.session_state.counter = 1

    elif st.session_state.counter == 1:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = process_query(st.session_state.original_question, prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.counter = 0
