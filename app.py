import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Tuple, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter        

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env file.")

# --- MongoDB Connection ---
def get_mongodb_connection():
    """Establish MongoDB connection"""
    try:
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        print(f"✓ Connected to MongoDB successfully")
        return client[DB_NAME]
    except ConnectionFailure as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        raise

db = get_mongodb_connection()

# Collections
transactions_collection = db["transactions"]
products_collection = db["products"]
customers_collection = db["customers"]
support_tickets_collection = db["support_tickets"]

# --- JSON Upload to MongoDB ---
def upload_json_to_mongodb(json_file_path: str) -> int:
    """
    Upload JSON file data to MongoDB
    Handles both array and single object formats
    Extracts and normalizes customer and product data
    """
    print(f"\n📁 Reading JSON file: {json_file_path}")
   
    if not Path(json_file_path).exists():
        raise FileNotFoundError(f"JSON file not found at: {json_file_path}")
   
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
   
    # Handle array or single object
    if isinstance(data, list):
        documents = data
    else:
        documents = [data]
   
    # Take only first 100 records
    documents = documents[:100]
   
    print(f"   📊 Found {len(documents)} records to process (max 100)...")
   
    # Check if data already exists
    existing_count = transactions_collection.count_documents({})
    if existing_count > 0:
        response = input(f"\n⚠️  Found {existing_count} existing transactions. Replace? (yes/no): ").strip().lower()
        if response == 'yes':
            transactions_collection.delete_many({})
            customers_collection.delete_many({})
            products_collection.delete_many({})
            print("   Cleared existing data.")
        else:
            print("   Keeping existing data. Merging new data...")
   
    # Process and insert documents
    customers_dict = {}
    products_dict = {}
    transactions = []
   
    for doc in documents:
        # Extract customer info
        customer_id = doc.get("Customer ID", "UNKNOWN")
        customer_name = doc.get("Customer name", "Unknown")
        email = doc.get("Email", "N/A")
        phone = doc.get("Phone", "N/A")
        city = doc.get("City", "N/A")
        loyalty_tier = doc.get("Loyalty_Tier", "Regular")
       
        if customer_id not in customers_dict:
            customers_dict[customer_id] = {
                "customer_id": customer_id,
                "name": customer_name,
                "email": email,
                "phone": phone,
                "city": city,
                "loyalty_tier": loyalty_tier,
                "created_at": datetime.now()
            }
       
        # Extract product info
        product_id = doc.get("ID_product", "UNKNOWN")
        product_name = doc.get("Product", "Unknown")
        category = doc.get("Category", "N/A")
        sku = doc.get("SKUs", "N/A")
        cogs = doc.get("COGS", 0)
       
        if product_id not in products_dict:
            products_dict[product_id] = {
                "product_id": product_id,
                "name": product_name,
                "category": category,
                "sku": sku,
                "cogs": cogs,
                "margin_percent": doc.get("Margin_per_piece_percent", 0),
                "created_at": datetime.now()
            }
       
        # Create transaction document
        transaction = {
            "invoice_number": doc.get("Invoice Number", "N/A"),
            "txn_number": doc.get("Txn_No", "N/A"),
            "customer_id": customer_id,
            "customer_name": customer_name,
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
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
   
    # Insert customers
    if customers_dict:
        customers_collection.insert_many(list(customers_dict.values()))
        print(f"   ✓ Added {len(customers_dict)} unique customers")
   
    # Insert products
    if products_dict:
        products_collection.insert_many(list(products_dict.values()))
        print(f"   ✓ Added {len(products_dict)} unique products")
   
    # Insert transactions
    if transactions:
        result = transactions_collection.insert_many(transactions)
        print(f"   ✓ Added {len(result.inserted_ids)} transactions")
   
    return len(result.inserted_ids)

# --- Convert MongoDB Documents to Searchable Text ---
def mongodb_to_searchable_text() -> List[str]:
    """Convert MongoDB transactions to searchable text chunks"""
    print("\n📄 Converting MongoDB transactions to searchable format...")
   
    transactions = list(transactions_collection.find())
    if not transactions:
        raise ValueError("No transactions found in MongoDB. Please upload JSON file first.")
   
    texts = []
    for txn in transactions:
        # Create readable format for each transaction
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
- Status: {txn.get('status')}
"""
        texts.append(text)
   
    # Split into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text("\n".join(texts))
   
    print(f"✓ Created {len(chunks)} searchable chunks from {len(transactions)} transactions")
    return chunks

# --- Build RAG Model ---
def build_rag_model(api_key: str):
    """Build RAG model with Gemini API and FAISS"""
    print("\n🔨 Building RAG Model...")
   
    # Get searchable text from MongoDB
    chunks = mongodb_to_searchable_text()
   
    # Initialize embeddings
    print("   Initializing Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
   
    # Create vector store
    print("   Creating FAISS Vector Store...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
   
    # Initialize LLM
    print("   Initializing Gemini Chat Model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=1500
    )
   
    # Create QA chain
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful customer service assistant for an e-commerce platform. "
            "Based on the following transaction data from our database, "
            "provide a clear and helpful answer to the customer's question. "
            "Keep your answer concise and relevant to sales and transactions.\n\n"
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
   
    print("✓ RAG Model Built Successfully")
    return qa_chain, llm

# --- Intent Classification ---
def classify_intent(question: str, llm) -> str:
    """Classify user intent into 3 categories"""
    intent_prompt = f"""Classify the following customer question into EXACTLY ONE category:

1. SEARCH_DB - User is asking general questions about products, sales, inventory, prices, or other business information
2. CUSTOMER_HISTORY - User is asking about their purchase history, transactions, orders, or providing customer name/ID
3. SUPPORT - User has an issue, problem, complaint, or needs customer care assistance

Analyze the question carefully and respond with ONLY the category name.

Question: "{question}"

Response (only write one: SEARCH_DB, CUSTOMER_HISTORY, or SUPPORT):"""

    response = llm.invoke(intent_prompt)
    intent = response.content.strip().upper()
   
    # Fallback keyword matching
    if intent not in ["SEARCH_DB", "CUSTOMER_HISTORY", "SUPPORT"]:
        q_lower = question.lower()
       
        if any(word in q_lower for word in ["history", "transaction", "purchase", "order", "my orders", "customer id", "customer name", "my account", "my purchases", "past orders", "invoice"]):
            intent = "CUSTOMER_HISTORY"
        elif any(word in q_lower for word in ["problem", "issue", "help", "support", "complaint", "broken", "not working", "error", "fix", "call care", "contact"]):
            intent = "SUPPORT"
        else:
            intent = "SEARCH_DB"
   
    return intent

# --- Intent Handlers ---

def handle_search_db(question: str, qa_chain, chat_history: List[Tuple[str, str]]) -> str:
    """Search database and answer questions about products/sales"""
    print("\n🔍 Intent: SEARCH DATABASE")
    print("   Searching transaction data...")
   
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    answer = result.get("answer") or result.get("result") or "⚠️ No relevant information found in database"
   
    print("\n   📎 Source Transactions:")
    for i, doc in enumerate(result.get("source_documents", [])[:2], 1):
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"      [{i}] {snippet}...")
   
    return answer

def handle_customer_history(question: str, llm) -> str:
    """Retrieve customer purchase history and transactions"""
    print("\n📊 Intent: CUSTOMER HISTORY")
   
    # Extract customer identifier from question
    extract_prompt = f"""From this question, extract ONLY the customer identifier (name, ID, or email).
Question: "{question}"
Extract and return ONLY the customer name, ID, or email. If multiple found, list all.
If not found, respond with 'NOT_FOUND'."""
   
    response = llm.invoke(extract_prompt)
    identifier = response.content.strip()
   
    if identifier == "NOT_FOUND":
        print("   Please provide your customer name or customer ID")
        identifier = input("   Enter customer name or ID: ").strip()
   
    print(f"   🔍 Searching for customer: {identifier}")
   
    # Search for customer
    customer = customers_collection.find_one({
        "$or": [
            {"name": {"$regex": identifier, "$options": "i"}},
            {"customer_id": identifier},
            {"email": {"$regex": identifier, "$options": "i"}}
        ]
    })
   
    if not customer:
        return f"❌ Customer '{identifier}' not found in database."
   
    customer_id = customer.get("customer_id")
    customer_name = customer.get("name", "Unknown")
   
    # Get all transactions for this customer
    txn_list = list(transactions_collection.find(
        {"customer_id": customer_id}
    ).sort("date_of_purchase", -1))
   
    if not txn_list:
        return f"""
✓ Customer Found: {customer_name}
Email: {customer.get('email', 'N/A')}
Phone: {customer.get('phone', 'N/A')}
City: {customer.get('city', 'N/A')}
Loyalty Tier: {customer.get('loyalty_tier', 'N/A')}

❌ No purchase history found for this customer."""
   
    # Format transaction history
    history = f"""
{'='*70}
CUSTOMER PURCHASE HISTORY
{'='*70}
Name: {customer_name}
Email: {customer.get('email', 'N/A')}
Phone: {customer.get('phone', 'N/A')}
City: {customer.get('city', 'N/A')}
Loyalty Tier: {customer.get('loyalty_tier', 'N/A')}

{'='*70}
TRANSACTIONS ({len(txn_list)} total):
{'='*70}
"""
   
    total_spent = 0
    for idx, txn in enumerate(txn_list, 1):
        history += f"""
{idx}. Invoice: {txn.get('invoice_number')} | TXN: {txn.get('txn_number')}
   Product: {txn.get('product_name')} (Category: {txn.get('category')})
   Quantity: {txn.get('quantity')} pieces
   Amount: ${txn.get('total_amount'):.2f} | Gross: ${txn.get('gross_amount'):.2f}
   Discount: {txn.get('discount_percentage'):.2f}% | GST: ${txn.get('gst'):.2f}
   Payment: {txn.get('payment_mode')} | Channel: {txn.get('channel')}
   Date: {txn.get('date_of_purchase')}
   Status: {txn.get('status')}
"""
        total_spent += txn.get('total_amount', 0)
   
    history += f"""
{'='*70}
SUMMARY:
Total Transactions: {len(txn_list)}
Total Amount Spent: ${total_spent:.2f}
Last Purchase: {txn_list[0].get('date_of_purchase') if txn_list else 'N/A'}
{'='*70}
"""
    return history

def handle_support_request() -> str:
    """Handle customer support issues"""
    print("\n🆘 Intent: CUSTOMER SUPPORT")
    print("=" * 70)
    print("📞 Connecting you to customer care...\n")
   
    # Collect issue details
    print("Please describe your issue:")
    issue = input("> ").strip()
   
    email = input("Your email: ").strip()
    name = input("Your name: ").strip()
    phone = input("Your phone (optional): ").strip()
   
    # Create support ticket
    ticket_num = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
   
    # Determine priority
    priority = "high" if any(w in issue.lower() for w in ["urgent", "critical", "emergency", "broken"]) else "normal"
   
    ticket_doc = {
        "ticket_number": ticket_num,
        "customer_name": name,
        "customer_email": email,
        "customer_phone": phone,
        "issue": issue,
        "status": "open",
        "priority": priority,
        "created_at": datetime.now()
    }
   
    support_tickets_collection.insert_one(ticket_doc)
   
    response = f"""
{'='*70}
✓ SUPPORT TICKET CREATED
{'='*70}
Ticket Number: {ticket_num}
Priority: {priority.upper()}
Status: OPEN

Issue: {issue}

Your support ticket has been logged and assigned to our team.
Expected Response Time: 1-2 business hours

📞 Phone: 1-800-SUPPORT
📧 Email: support@company.com
💬 Live Chat: www.company.com/support

Reference your ticket number when following up.
{'='*70}
"""
    print(response)
    return response

# --- Main Chatbot Loop ---
def main():
    try:
        print("\n" + "="*70)
        print("📊 JSON-BASED TRANSACTION RAG CHATBOT")
        print("="*70)
       
        # Step 1: Upload JSON file
        json_file = input("\n📁 Enter JSON file path: ").strip()
       
        if not json_file:
            print("❌ JSON file path is required!")
            return
       
        try:
            upload_json_to_mongodb(json_file)
        except Exception as e:
            print(f"❌ Error uploading JSON: {e}")
            return
       
        # Step 2: Build RAG model
        qa_chain, llm = build_rag_model(GOOGLE_API_KEY)
       
    except Exception as e:
        print(f"\n❌ SETUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
   
    chat_history: List[Tuple[str, str]] = []
   
    print("\n" + "="*70)
    print("🤖 INTELLIGENT SALES & SUPPORT CHATBOT")
    print("="*70)
    print("\n✨ I can help you with:")
    print("   1. 🔍 Answer questions about products, sales, and transactions")
    print("   2. 📊 Show your complete purchase history")
    print("   3. 🆘 Connect with customer support")
    print("\n   Type 'quit' or 'exit' to end conversation")
    print("="*70)
   
    while True:
        try:
            question = input("\n💬 You: ").strip()
           
            if question.lower() in ("quit", "exit"):
                print("\n👋 Thank you for using our chatbot. Goodbye!")
                break
           
            if not question:
                continue
           
            # Classify intent
            print("   🤔 Analyzing your request...")
            intent = classify_intent(question, llm)
           
            # Route to handler
            if intent == "SEARCH_DB":
                answer = handle_search_db(question, qa_chain, chat_history)
               
            elif intent == "CUSTOMER_HISTORY":
                answer = handle_customer_history(question, llm)
               
            elif intent == "SUPPORT":
                answer = handle_support_request()
           
            print(f"\n🤖 Assistant:\n{answer}")
            chat_history.append((question, answer))
            print("\n" + "-"*70)
           
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()


