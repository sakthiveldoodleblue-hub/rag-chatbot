import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from utils import mongodb_to_searchable_text
from intent_classifier import EmbeddingIntentClassifier

def build_rag_model(api_key, transactions_collection):
    """Build RAG model with embeddings and retrieval chain"""
    
    try:
        with st.spinner("🔄 Building embeddings..."):
            # Convert MongoDB transactions to searchable text
            chunks = mongodb_to_searchable_text(transactions_collection)
            
            if not chunks:
                raise ValueError("No chunks generated from transactions")
            
            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            
            # Build FAISS vector store
            vectorstore = FAISS.from_texts(chunks, embeddings)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 documents
            )
            
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                max_output_tokens=1500,
                google_api_key=api_key
            )
            
            # Define QA prompt
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful e-commerce customer service assistant. 
Use the provided transaction data to answer questions accurately.

Context (Transaction Data):
{context}

Customer Question: {question}

Instructions:
- Answer based on the context provided
- If information is not in the context, say "I don't have that information"
- Be concise and helpful
- Format currency as USD with $ symbol
- Include specific transaction details when relevant

Answer:"""
            )
            
            # Build conversational retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                verbose=False
            )
            
            # Initialize intent classifier
            intent_classifier = EmbeddingIntentClassifier(embeddings)
            
            st.success("✅ Models loaded successfully")
            
            return qa_chain, llm, intent_classifier
    
    except Exception as e:
        st.error(f"Error building RAG model: {str(e)}")
        raise


