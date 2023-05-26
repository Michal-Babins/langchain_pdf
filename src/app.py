import pickle
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title(':smile: :frog: LLM Chat App')
    st.markdown(
        '''
        # LLM Chat App
        This is a simple chat app that allows users to send and receive messages.'''
    )
    add_vertical_space(5)
    st.write('Made with :heart:')
    
def main():
    st.write('Chat with pdf :talk:')
    
    #upload a pdf file
    pdf = st.file_uploader('Upload your pdf file', type=['pdf'])
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        pdf_basename = pdf.name[:-4]
        
        load_dotenv()
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        
        chunks = text_splitter.split_text(text=text)
        
        if os.path.exists(f"{pdf_basename}.pkl"):
            with open(f"{pdf_basename}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings loaded successfully')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{pdf_basename}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                
            # st.write('Embeddings saved successfully')
        
        #Accept user question/query
        query = st.text_input("Ask a question about your document:")
        # st.write(query)
        
        #get semantic search for the top 3 chuncks
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
            llm = OpenAI(temperature=0,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb: 
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
    
if __name__ == '__main__':
    main()