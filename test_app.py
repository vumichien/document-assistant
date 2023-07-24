import pytest
import pinecone
import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
load_dotenv()


def test_data():
    pdf_data = []
    for doc in glob.glob("data/*.PDF"):
        print(doc)
        loader = PyMuPDFLoader(doc)
        loaded_pdf = loader.load()
        for document in loaded_pdf:
            pdf_data.append(document)
    assert len(pdf_data) > 0
