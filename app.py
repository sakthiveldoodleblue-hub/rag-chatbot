import os
import json
import streamlit as st
from datetime import datetime
from typing import List, Tuple
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional imports
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="RAG Sales Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'intent_classifier' not in st.session_state:
    st.session_state.intent_classifier = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- MongoDB Connection ---
@st.cache_resource
def get_mongodb_connection():
    """Establish MongoDB connection"""
    try:
        MONGODB_URI = os.getenv("MONGODB_URI", st.secrets.get("MONGODB_URI", ""))
        if not MONGODB_URI:
            st.error("❌ MongoDB URI not configured. Please add it in Streamlit secrets.")
            st.info("Go to Settings → Secrets and add: MONGODB_URI = 'your-connection-string'")
            return None
        
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        DB_NAME = os.getenv("DB_NAME", st.secrets.get("DB_NAME", "rag_chatbot_db"))
        return client[DB_NAME]
    except ConnectionFailure as e:
        st.error(f"❌ Failed to connect to MongoDB: {e}")
        st.info("Please check your MongoDB Atlas connection string and network access settings.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

db = get_mongodb_connection()

if db is not None:
    transactions_collection = db["transactions"]
    products_collection = db["products"]
    customers_collection = db["customers"]
    support_tickets_collection = db["support_tickets"]
else:
    st.warning("⚠️ MongoDB not connected. Please configure secrets.")
    st.stop()

# --- Upload JSON to MongoDB ---
def upload_json_to_mongodb(json_data: dict) -> int:
    """Upload JSON data to MongoDB"""
    if isinstance(json_data, list):
        documents = json_data
    else:
        documents = [json_data]
    
    documents = documents[:50]
    
    # Clear existing data
    transactions_collection.delete_many({})
    customers_collection.delete_many({})
    products_collection.delete_many({})
    
    customers_dict = {}
    products_dict = {}
    transactions = []
    
    for doc in documents:
        customer_id = doc.get("Customer ID", "UNKNOWN")
        customer_name = doc.get("Customer name", "Unknown")
        
        if customer_id not in customers_dict:
            customers_dict[customer_id] = {
                "customer_id": customer_id,
                "name": customer_name,
                "email": doc.get("Email", "N/A"),
                "phone": doc.get("Phone", "N/A"),
                "city": doc.get("City", "N/A"),
                "loyalty_tier": doc.get("Loyalty_Tier", "Regular"),
                "created_at": datetime.now()
            }
        
        product_id = doc.get("ID_product", "UNKNOWN")
        product_name = doc.get("Product", "Unknown")
        
        if product_id not in products_dict:
            products_dict[product_id] = {
                "product_id": product_id,
                "name": product_name,
                "category": doc.get("Category", "N/A"),
                "sku": doc.get("SKUs", "N/A"),
                "cogs": doc.get("COGS", 0),
                "margin_percent": doc.get("Margin_per_piece_percent", 0),
                "created_at": datetime.now()
            }
        
        transaction = {
            "invoice_number": doc.get("Invoice Number", "N/A"),
            "txn_number": doc.get("Txn_No", "N/A"),
            "customer_id": customer_id,
            "customer_name": customer_name,
            "product_id": product_id,
            "product_name": product_name,
            "category": doc.get("Category", "N/A"),
            "quantity": doc.get("Quantity_piece", 0),
            "gross_amount": doc.get("Gross_Amount", 0),
            "discount_percentage": doc.get("Discount_Percentage", 0),
            "total_amount": doc.get("Total Amount", 0),
            "gst": doc.get("GST", 0),
            "payment_mode": doc.get("Payment_mode", "N/A"),
            "date_of_purchase": doc.get("Date_of_purchase", datetime.now().isoformat()),
            "channel": doc.get("Channel", "N/A"),
            "store_location": doc.get("Store_location", "N/A"),
            "mode": doc.get("Mode", "N/A"),
            "status": "completed",
            "created_at": datetime.now()
        }
        transactions.append(transaction)
    
    if customers_dict:
        customers_collection.insert_many(list(customers_dict.values()))
    if products_dict:
        products_collection.insert_many(list(products_dict.values()))
    if transactions:
        result = transactions_collection.insert_many(transactions)
        return len(result.inserted_ids)
    return 0

# --- Convert MongoDB to Searchable Text ---
def mongodb_to_searchable_text() -> List[str]:
    """Convert MongoDB transactions to searchable text chunks"""
    transactions = list(transactions_collection.find())
    if not transactions:
        raise ValueError("No transactions found.")
    
    texts = []
    for txn in transactions:
        text = f"""
Transaction Details:
- Invoice: {txn.get('invoice_number')}
- Customer: {txn.get('customer_name')} (ID: {txn.get('customer_id')})
- Product: {txn.get('product_name')} (Category: {txn.get('category')})
- Quantity: {txn.get('quantity')} pieces
- Total Amount: ${txn.get('total_amount'):.2f}
- GST: ${txn.get('gst'):.2f}
- Gross Amount: ${txn.get('gross_amount'):.2f}
- Discount: {txn.get('discount_percentage'):.2f}%
- Payment Mode: {txn.get('payment_mode')}
- Store Location: {txn.get('store_location')}
- Channel: {txn.get('channel')}
- Purchase Date: {txn.get('date_of_purchase')}
"""
        texts.append(text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text("\n".join(texts))
    return chunks

# --- Embedding-Based Intent Classifier ---
class EmbeddingIntentClassifier:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.use_embeddings = SKLEARN_AVAILABLE
        
        self.intent_templates = {
            "SEARCH_DB": [
                "What products do you have?",
                "Show me sales data",
                "How many items sold?",
                "What is the price of product?",
                "List all products",
                "Show inventory",
                "Top selling products",
                "Total sales",
                "Product information"
            ],
            "CUSTOMER_HISTORY": [
                "Show my purchase history",
                "What did I buy?",
                "My previous orders",
                "Transaction history",
                "Orders for customer",
                "Show transactions",
                "My account purchases",
                "Invoice number",
                "My orders"
            ],
            "SUPPORT": [
                "I have a problem",
                "Need help",
                "Product is broken",
                "Issue with delivery",
                "Customer service",
                "File a complaint",
                "Not working",
                "Report an issue",
                "Refund request"
            ]
        }
        
        if self.use_embeddings:
            self.intent_embeddings = {}
            for intent, templates in self.intent_templates.items():
                embeddings = self.embeddings_model.embed_documents(templates)
                self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
    
    def classify(self, question: str) -> Tuple[str, float]:
        if self.use_embeddings:
            question_embedding = self.embeddings_model.embed_query(question)
            
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = cosine_similarity(
                    [question_embedding], 
                    [intent_embedding]
                )[0][0]
                similarities[intent] = similarity
            
            best_intent = max(similarities, key=similarities.get)
            confidence = similarities[best_intent]
        else:
            # Fallback: keyword-based
            q_lower = question.lower()
            if any(word in q_lower for word in ["history", "transaction", "purchase", "order", "invoice"]):
                best_intent = "CUSTOMER_HISTORY"
                confidence = 0.8
            elif any(word in q_lower for word in ["problem", "issue", "help", "support", "complaint", "broken"]):
                best_intent = "SUPPORT"
                confidence = 0.8
            else:
                best_intent = "SEARCH_DB"
                confidence = 0.7
        
        return best_intent, confidence

# --- Build RAG Model ---
@st.cache_resource
def build_rag_model():
    """Build RAG model"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY", ""))
    if not GOOGLE_API_KEY:
        st.error("❌ Google API Key not configured.")
        st.info("Go to Settings → Secrets and add: GOOGLE_API_KEY = 'your-api-key'")
        return None, None, None
    
    try:
        chunks = mongodb_to_searchable_text()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
            max_output_tokens=1500
        )
        
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful customer service assistant. "
                "Based on the transaction data, provide a clear answer.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        intent_classifier = EmbeddingIntentClassifier(embeddings)
        
        return qa_chain, llm, intent_classifier
    except Exception as e:
        st.error(f"❌ Error building RAG model: {e}")
        return None, None, None

# --- Intent Handlers ---
def handle_search_db(question: str, qa_chain, chat_history: List[Tuple[str, str]]) -> str:
    """Search database"""
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    answer = result.get("answer") or result.get("result") or "⚠️ No relevant information found"
    
    with st.expander("📎 View Source Documents"):
        for i, doc in enumerate(result.get("source_documents", []), 1):
            st.text(f"Source {i}:")
            st.code(doc.page_content, language="text")
    
    return answer

def handle_customer_history(question: str, llm) -> str:
    """Retrieve customer purchase history"""
    extract_prompt = f"""Extract from this question:
1. Customer identifier (name, ID, email)
2. Invoice number

Question: "{question}"

Format:
CUSTOMER: [identifier or NOT_FOUND]
INVOICE: [number or NOT_FOUND]"""
    
    response = llm.invoke(extract_prompt)
    extraction = response.content.strip()
    
    lines = extraction.split('\n')
    customer_id = "NOT_FOUND"
    invoice_num = "NOT_FOUND"
    
    for line in lines:
        if "CUSTOMER:" in line:
            customer_id = line.split("CUSTOMER:")[-1].strip()
        elif "INVOICE:" in line:
            invoice_num = line.split("INVOICE:")[-1].strip()
    
    # Search by invoice
    if invoice_num != "NOT_FOUND" and invoice_num:
        txn = transactions_collection.find_one({"invoice_number": invoice_num})
        if not txn:
            return f"❌ Invoice '{invoice_num}' not found."
        
        customer = customers_collection.find_one({"customer_id": txn.get("customer_id")})
        
        return f"""**Invoice Details**
📄 **Invoice:** {txn.get('invoice_number')}
📅 **Date:** {txn.get('date_of_purchase')}

**Customer:** {customer.get('name', 'Unknown') if customer else 'Unknown'}
**Product:** {txn.get('product_name')}
**Category:** {txn.get('category')}
**Quantity:** {txn.get('quantity')} pieces
**Amount:** ${txn.get('total_amount'):.2f}
**Status:** {txn.get('status')}"""
    
    # Search by customer
    if customer_id == "NOT_FOUND" or not customer_id:
        return "❌ Please provide a customer name, ID, or invoice number."
    
    customer = customers_collection.find_one({
        "$or": [
            {"name": {"$regex": customer_id, "$options": "i"}},
            {"customer_id": customer_id},
            {"email": {"$regex": customer_id, "$options": "i"}}
        ]
    })
    
    if not customer:
        return f"❌ Customer '{customer_id}' not found."
    
    txn_list = list(transactions_collection.find(
        {"customer_id": customer.get("customer_id")}
    ).sort("date_of_purchase", -1).limit(10))
    
    if not txn_list:
        return f"✅ Customer: {customer.get('name')}\n❌ No purchase history found."
    
    history = f"""**Customer: {customer.get('name')}**
📧 {customer.get('email')} | 📞 {customer.get('phone')}
🏙️ {customer.get('city')} | ⭐ {customer.get('loyalty_tier')}

**Recent Transactions ({len(txn_list)}):**
"""
    
    total = 0
    for i, txn in enumerate(txn_list[:5], 1):
        history += f"""
{i}. **Invoice:** {txn.get('invoice_number')}
   - **Product:** {txn.get('product_name')}
   - **Amount:** ${txn.get('total_amount'):.2f}
   - **Date:** {txn.get('date_of_purchase')}
"""
        total += txn.get('total_amount', 0)
    
    history += f"\n**Total Spent (shown):** ${total:.2f}"
    return history

def handle_support_request(question: str) -> str:
    """Handle support request"""
    ticket_num = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    ticket_doc = {
        "ticket_number": ticket_num,
        "issue": question,
        "status": "open",
        "priority": "normal",
        "created_at": datetime.now()
    }
    
    support_tickets_collection.insert_one(ticket_doc)
    
    return f"""✅ **Support Ticket Created**

**Ticket #:** {ticket_num}
**Issue:** {question}
**Status:** Open
**Priority:** Normal

Our support team will respond within 1-2 business hours.

📞 **Phone:** 1-800-SUPPORT
📧 **Email:** support@company.com
💬 **Live Chat:** www.company.com/support"""

# --- Streamlit UI ---
st.title("🤖 RAG Sales & Support Chatbot")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Data Management")
    
    uploaded_file = st.file_uploader("Upload JSON Transaction File", type=['json'])
    
    if uploaded_file and not st.session_state.data_loaded:
        if st.button("Load Data & Build Model"):
            with st.spinner("Processing data..."):
                try:
                    json_data = json.load(uploaded_file)
                    count = upload_json_to_mongodb(json_data)
                    st.success(f"✅ Loaded {count} transactions")
                    
                    with st.spinner("Building RAG model..."):
                        qa_chain, llm, intent_clf = build_rag_model()
                        
                        if qa_chain and llm and intent_clf:
                            st.session_state.qa_chain = qa_chain
                            st.session_state.llm = llm
                            st.session_state.intent_classifier = intent_clf
                            st.session_state.data_loaded = True
                            st.success("✅ Model ready!")
                            st.rerun()
                        else:
                            st.error("Failed to build model. Check your API key.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    if st.session_state.data_loaded:
        st.success("✅ System Ready")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.info("""
**Capabilities:**
- 🔍 Product & sales queries
- 📊 Customer purchase history
- 📄 Invoice lookup
- 🆘 Support ticket creation
        """)

# Main chat interface
if not st.session_state.data_loaded:
    st.info("👈 **Please upload a JSON file to get started**")
    
    with st.expander("ℹ️ How to use this chatbot"):
        st.markdown("""
        1. **Upload your transaction JSON file** using the sidebar
        2. **Click "Load Data & Build Model"** to process the data
        3. **Start chatting!** Ask questions about:
           - Products and sales data
           - Customer purchase history
           - Specific invoices
           - Get customer support
        """)
else:
    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
    
    # Chat input
    if prompt := st.chat_input("Ask about sales, customer history, or get support..."):
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Classify intent
                    intent, confidence = st.session_state.intent_classifier.classify(prompt)
                    
                    # Show classification
                    with st.expander("🎯 Intent Classification"):
                        st.write(f"**Detected Intent:** {intent}")
                        st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Handle based on intent
                    if intent == "SEARCH_DB":
                        answer = handle_search_db(prompt, st.session_state.qa_chain, st.session_state.chat_history)
                    elif intent == "CUSTOMER_HISTORY":
                        answer = handle_customer_history(prompt, st.session_state.llm)
                    elif intent == "SUPPORT":
                        answer = handle_support_request(prompt)
                    else:
                        answer = "I'm not sure how to help with that. Can you rephrase?"
                    
                    st.markdown(answer)
                    
                    # Add to history
                    st.session_state.chat_history.append((prompt, answer))
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

