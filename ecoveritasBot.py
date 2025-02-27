import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader  # Install with: pip install pymupdf
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
import streamlit as st

os.environ['PINECONE_API_KEY'] = "pcsk_76WQU2_77N63EQhEq88Br5Skj4d1fQU9TzLeEsBzV9FmWUKET2mcvr9N97Bdkt9uddgcsT"
os.environ[
    'OPENAI_API_KEY'] = "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"

index_name = "interpretation-docs"
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)


def process_query(query, q1_ans):
    # Multi Query: Different Perspectives
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

    # Step 2: Perform similarity search for each modified query
    retrieved_docs = []  # List to store unique documents
    seen_docs = set()  # Set to track unique document contents

    for mod_query in modified_queries:
        docs = vectorstore.similarity_search(mod_query, k=3)
        for doc in docs:
            if doc.page_content not in seen_docs:  # Avoid duplicates
                seen_docs.add(doc.page_content)
                retrieved_docs.append(doc)

    # Step 3: Generate a final answer using retrieved documents
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
                model="deepseek-reasoner",  # Specify DeepSeek model
                openai_api_base="https://api.deepseek.com/v1",  # DeepSeek API endpoint
                openai_api_key="sk-aae615c845c14eae866d0238275c4c78"  # Your DeepSeek API key
            )
            | StrOutputParser()
    )

    # The rest of your code remains the same
    formatted_prompt = answer_template.format(
        question=query,
        context=context,
        q1_ans=q1_ans
    )

    print(formatted_prompt)

    final_answer = generate_answer.invoke({"question": query, "context": context, "q1_ans": q1_ans})

    return final_answer

st.title("Evoveritas Bot Deepseek")

# Initialize chat history and counter
if "messages" not in st.session_state:
    st.session_state.messages = []
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "original_question" not in st.session_state:
    st.session_state.original_question = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    if st.session_state.counter == 0:
        st.session_state.original_question = prompt

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Ask the first question
        q1 = "Are they a large or small producer?"
        with st.chat_message("assistant"):
            st.markdown(q1)
        st.session_state.messages.append({"role": "assistant", "content": q1})

        # Update counter
        st.session_state.counter = 1

    elif st.session_state.counter == 1:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = process_query(st.session_state.original_question, prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Reset counter
        st.session_state.counter = 0

# query_input = input("Enter query: ")
# print(process_query(query_input))
#
# st.title("Evoveritas Bot")
#
# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# counter = 0
# prompt = st.chat_input("What is up?")
# if prompt and counter == 0:
#     prompt_original_question = prompt
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#
#     q1 = "Are they a large or small producer?"
#     with st.chat_message("assistant"):
#         st.markdown(q1)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "assistant", "content": q1})
#     prompt=""
#     counter = 1
#
#
#
# if prompt and counter == 1:
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#
#     response = process_query(prompt_original_question)
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         st.markdown(response)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     counter = 0

# query = "
#
# "
# query="What are the consequences of not reporting aspects of the business? The aspect they're concerned they can't capture is takeaway cups. brand of the cups small producer large producer"
#
# # Multi Query: Different Perspectives
# template = """You are an AI language model assistant. Your task is to generate five
# different versions of the given user question to retrieve relevant documents from a vector
# database. By generating multiple perspectives on the user question, your goal is to help
# the user overcome some of the limitations of the distance-based similarity search.
# Provide these alternative questions separated by newlines. Original question: {question}"""
# prompt_perspectives = ChatPromptTemplate.from_template(template)
#
#
# print(f"Original Query: {query}")
# # print(f"Improved Query: {improved_query}")
#
#
# generate_queries = (
#     prompt_perspectives
#     | ChatOpenAI(temperature=0)
#     | StrOutputParser()
#     | (lambda x: x.split("\n"))
# )
#
# modified_queries = generate_queries.invoke({"question":query})
# print(modified_queries)
#
#
# # results = vectorstore.similarity_search(query, k=3)
#
#
# # Step 2: Perform similarity search for each modified query
# retrieved_docs = []  # List to store unique documents
# seen_docs = set()  # Set to track unique document contents
#
# for mod_query in modified_queries:
#     docs = vectorstore.similarity_search(mod_query, k=3)
#     for doc in docs:
#         if doc.page_content not in seen_docs:  # Avoid duplicates
#             seen_docs.add(doc.page_content)
#             retrieved_docs.append(doc)
#
# print(len(retrieved_docs))
#
# # Step 3: Generate a final answer using retrieved documents
# context = "\n\n".join([doc.page_content for doc in retrieved_docs])
#
# answer_template = """Using the following retrieved information, answer the question: {question}
#
# Context:
# {context}
#
# Final Answer:"""
# prompt_answer = ChatPromptTemplate.from_template(answer_template)
#
# generate_answer = (
#     prompt_answer
#     | ChatOpenAI(temperature=0.5)
#     | StrOutputParser()
# )
#
# final_answer = generate_answer.invoke({"question": query, "context": context})
#
# print("\nFinal Answer:", final_answer)
