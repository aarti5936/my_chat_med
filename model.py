import json
import os
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstore/db_faiss'
CHAT_LOG_FILE = "chat_history.json"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval."""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Load the model
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

# Ensure JSON file exists
def initialize_json():
    """Creates an empty JSON file if it doesn't exist."""
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "w") as f:
            json.dump([], f)

# Save chat history in JSON
def save_chat_json(query, answer):
    chat_data = {"query": query, "answer": answer}

    try:
        # Load existing chat history
        with open(CHAT_LOG_FILE, "r") as f:
            chat_history = json.load(f)

        # Append new chat data
        chat_history.append(chat_data)

        # Save updated history
        with open(CHAT_LOG_FILE, "w") as f:
            json.dump(chat_history, f, indent=4)

        print(f"Chat successfully saved in JSON: {chat_data}")

    except Exception as e:
        print(f"Error saving chat: {e}")

# Chainlit Start
@cl.on_chat_start
async def start():
    initialize_json()  # Ensure JSON file is ready
    chain = qa_bot()
    cl.user_session.set("chain", chain)

    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

# Chat Message Handling
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSources:\n" + "\n".join(str(src) for src in sources)
    else:
        answer += "\n\nNo sources found."

    # Save chat to JSON file
    save_chat_json(message.content, answer)

    await cl.Message(content=answer).send()
