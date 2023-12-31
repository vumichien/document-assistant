import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pinecone
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import glob
import chainlit as cl

STREAMING = True

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAME_SPACE = os.getenv("NAME_SPACE")
DEBUG = os.getenv("DEBUG")

pdf_data = []
for doc in glob.glob("data/*.PDF"):
    print(doc)
    loader = PyMuPDFLoader(doc)
    loaded_pdf = loader.load()
    for document in loaded_pdf:
        pdf_data.append(document)


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "Assistant"}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
def main():
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api key in console
    )

    index = pinecone.GRPCIndex(PINECONE_INDEX)
    print(index.describe_index_stats())
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if index.describe_index_stats().total_vector_count == 0:
        print("Adding documents")
        docsearch = Pinecone.from_documents(
            pdf_data, embeddings, index_name=PINECONE_INDEX, namespace=NAME_SPACE
        )
        print("Done adding documents")
    else:
        docsearch = Pinecone.from_existing_index(
            PINECONE_INDEX, embeddings, namespace=NAME_SPACE
        )
        print("Loaded index documents")

    # Create a chain that uses the Chroma vector store
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=STREAMING)

    # docsearch = FAISS.from_documents(pdf_data, embeddings)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    system_template = """Use the following pieces of context to answer the users question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        *箇条書きで回答してください．
        Answer in japanease:
        {summaries}
        """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.01},
        ),
        # retriever=docsearch.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": 10},
        # ),
        memory=memory,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        reduce_k_below_max_tokens=True,
    )

    # Store the chain in the user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"],
    )
    # cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    if DEBUG == "True":
        print(res)
        # Post processing here
        print(len(res["source_documents"]))
    answer = res["answer"]
    # source_elements = []
    await cl.Message(content=answer).send()
    for idx, source in enumerate(res["source_documents"]):
        title = source.metadata["title"]
        page_number = int(source.metadata["page"]) + 1
        path = source.metadata["file_path"]
        content = source.page_content
        answer_source = f"タイトル: {title}。 ページ: {page_number}\nファイルパス: {path}"
        source_elements = [cl.Text(name=f"{path}", content=content, display="side")]
        await cl.Message(content=answer_source, elements=source_elements).send()
