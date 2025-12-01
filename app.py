import os
import json
import pickle
import streamlit as st
from datetime import datetime
from typing import List, Tuple
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vectorstore_cached' not in st.session_state:
    st.session_state.vectorstore_cached = False

# MongoDB
@st.cache_resource
def get_mongodb():
    try:
        uri = os.getenv("MONGODB_URI", st.secrets.get("MONGODB_URI", ""))
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client[st.secrets.get("DB_NAME", "rag_chatbot_db")]
    except Exception as e:
        st.error(f"❌ MongoDB: {e}")
        return None

db = get_mongodb()
if db is not None:
    transactions_collection = db["transactions"]
    customers_collection = db["customers"]
    products_collection = db["products"]
    support_tickets_collection = db["support_tickets"]
else:
    st.stop()

# Upload JSON (limit to 50)
def upload_json(data):
    if isinstance(data, list):
        docs = data[:50]  # Only 50 records
    else:
        docs = [data]
    
    transactions_collection.delete_many({})
    customers_collection.delete_many({})
    products_collection.delete_many({})
    
    customers = {}
    products = {}
    txns = []
    
    for d in docs:
        cid = d.get("Customer ID", "UNK")
        pid = d.get("ID_product", "UNK")
        
        if cid not in customers:
            customers[cid] = {
                "customer_id": cid,
                "name": d.get("Customer name", "Unknown"),
                "email": d.get("Email", "N/A"),
                "phone": d.get("Phone", "N/A"),
                "city": d.get("City", "N/A"),
                "loyalty_tier": d.get("Loyalty_Tier", "Regular"),
                "created_at": datetime.now()
            }
        
        if pid not in products:
            products[pid] = {
                "product_id": pid,
                "name": d.get("Product", "Unknown"),
                "category": d.get("Category", "N/A"),
                "created_at": datetime.now()
            }
        
        txns.append({
            "invoice_number": d.get("Invoice Number", "N/A"),
            "customer_id": cid,
            "customer_name": d.get("Customer name", "Unknown"),
            "product_id": pid,
            "product_name": d.get("Product", "Unknown"),
            "category": d.get("Category", "N/A"),
            "quantity": d.get("Quantity_piece", 0),
            "total_amount": d.get("Total Amount", 0),
            "payment_mode": d.get("Payment_mode", "N/A"),
            "date_of_purchase": d.get("Date_of_purchase", ""),
            "channel": d.get("Channel", "N/A"),
            "status": "completed",
            "created_at": datetime.now()
        })
    
    if customers:
        customers_collection.insert_many(list(customers.values()))
    if products:
        products_collection.insert_many(list(products.values()))
    if txns:
        transactions_collection.insert_many(txns)
    
    return len(txns)

# Get text chunks
def get_chunks():
    txns = list(transactions_collection.find())
    texts = []
    for t in txns:
        text = f"""Transaction {t['invoice_number']}:
Customer: {t['customer_name']} ({t['customer_id']})
Product: {t['product_name']} (Category: {t['category']})
Quantity: {t['quantity']} | Amount: ${t['total_amount']:.2f}
Payment: {t['payment_mode']} | Date: {t['date_of_purchase']}"""
        texts.append(text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_text("\n\n".join(texts))

# Build RAG with caching
def build_rag():
    key = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY", ""))
    if not key:
        st.error("❌ No API key")
        return None, None
    
    try:
        # Check if vectorstore is already cached
        if st.session_state.vectorstore_cached:
            st.info("📦 Using cached embeddings (saving API quota!)")
            vectorstore = st.session_state.cached_vectorstore
        else:
            st.info("🔨 Building embeddings (this uses API quota)...")
            chunks = get_chunks()
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=key,
            )
            
            # Create and cache vectorstore
            vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.cached_vectorstore = vectorstore
            st.session_state.vectorstore_cached = True
            st.success("✅ Embeddings created and cached!")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=key,
            max_output_tokens=1000
        )
        
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Based on this data:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        return qa_chain, llm
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        
        # If quota exceeded, offer light mode
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("""
⚠️ **Quota Exceeded!**

**Solutions:**
1. **Wait 24 hours** for quota reset
2. **Create new API key** at https://aistudio.google.com/app/apikey
3. **Use Light Mode** (no embeddings) - see sidebar
            """)
        
        return None, None

# Classify intent
def classify_intent(q):
    q = q.lower()
    if any(w in q for w in ["history", "purchase", "order", "invoice", "my"]):
        return "HISTORY"
    elif any(w in q for w in ["problem", "issue", "help", "support", "broken"]):
        return "SUPPORT"
    else:
        return "SEARCH"

# Handlers
def handle_search(question, qa_chain, history):
    result = qa_chain.invoke({"question": question, "chat_history": history})
    return result.get("answer", "No answer found")

def handle_history(question, llm):
    prompt = f"Extract customer name/ID or invoice from: '{question}'. Reply ONLY with identifier or 'NONE'"
    response = llm.invoke(prompt)
    identifier = response.content.strip()
    
    if identifier == "NONE" or not identifier:
        return "❌ Please provide customer name, ID, or invoice number."
    
    # Try invoice
    txn = transactions_collection.find_one({"invoice_number": identifier})
    if txn:
        return f"""**Invoice Details:**
📄 **Invoice:** {txn['invoice_number']}
👤 **Customer:** {txn['customer_name']}
📦 **Product:** {txn['product_name']}
💰 **Amount:** ${txn['total_amount']:.2f}
📅 **Date:** {txn['date_of_purchase']}"""
    
    # Try customer
    customer = customers_collection.find_one({
        "$or": [
            {"name": {"$regex": identifier, "$options": "i"}},
            {"customer_id": identifier}
        ]
    })
    
    if not customer:
        return f"❌ '{identifier}' not found"
    
    txns = list(transactions_collection.find({"customer_id": customer['customer_id']}).limit(5))
    
    result = f"""**Customer:** {customer['name']}
📧 {customer['email']} | 📞 {customer['phone']}

**Transactions:**
"""
    for i, t in enumerate(txns, 1):
        result += f"{i}. {t['product_name']} - ${t['total_amount']:.2f}\n"
    
    return result

def handle_support(question):
    ticket = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    support_tickets_collection.insert_one({
        "ticket_number": ticket,
        "issue": question,
        "status": "open",
        "created_at": datetime.now()
    })
    return f"✅ **Ticket Created:** {ticket}\n\nWe'll respond within 1-2 hours."

# UI
st.title("🤖 RAG Chatbot (Quota-Optimized)")
st.caption("💡 Caches embeddings to save API quota")
st.markdown("---")

with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded = st.file_uploader("Upload JSON (max 50 records)", type=['json'])
    
    if uploaded and not st.session_state.data_loaded:
        if st.button("Load Data & Build Model"):
            with st.spinner("Loading data..."):
                try:
                    data = json.load(uploaded)
                    count = upload_json(data)
                    st.success(f"✅ Loaded {count} transactions")
                    
                    with st.spinner("Building model..."):
                        qa_chain, llm = build_rag()
                        
                        if qa_chain and llm:
                            st.session_state.qa_chain = qa_chain
                            st.session_state.llm = llm
                            st.session_state.data_loaded = True
                            st.success("✅ Ready!")
                            st.rerun()
                        else:
                            st.error("Failed to build model")
                
                except Exception as e:
                    st.error(f"❌ {e}")
    
    if st.session_state.data_loaded:
        st.success("✅ System Ready")
        
        # Show cache status
        if st.session_state.vectorstore_cached:
            st.info("💾 Embeddings are cached")
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("⚠️ Clear Cache & Rebuild"):
            st.session_state.vectorstore_cached = False
            st.session_state.data_loaded = False
            st.rerun()
    
    st.markdown("---")
    st.info("""
**Quota Saving Tips:**
- ✅ Embeddings cached after first build
- ✅ Only 50 records processed
- ✅ Click "Load Data" only ONCE
- ⚠️ Don't repeatedly rebuild!
    """)

if not st.session_state.data_loaded:
    st.info("👈 Upload JSON file to start")
    
    st.warning("""
⚠️ **IMPORTANT: Avoid Quota Errors**

1. Only click "Load Data & Build Model" **ONCE**
2. Embeddings are cached after first build
3. Use **max 50 records** from your file
4. If you see quota error:
   - Wait 24 hours, OR
   - Create new API key at https://aistudio.google.com/app/apikey
    """)
else:
    # Chat
    for user, bot in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user)
        with st.chat_message("assistant"):
            st.markdown(bot)
    
    if prompt := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            try:
                intent = classify_intent(prompt)
                
                if intent == "SEARCH":
                    answer = handle_search(prompt, st.session_state.qa_chain, st.session_state.chat_history)
                elif intent == "HISTORY":
                    answer = handle_history(prompt, st.session_state.llm)
                else:
                    answer = handle_support(prompt)
                
                st.markdown(answer)
                st.session_state.chat_history.append((prompt, answer))
            
            except Exception as e:
                st.error(f"❌ {e}")


