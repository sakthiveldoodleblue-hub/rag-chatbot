import streamlit as st
import os
import logging
from datetime import datetime
from db import init_collections
from upload import upload_json_to_mongodb
from rag_model import build_rag_model
from search_db import handle_search_db
from customer_history import handle_customer_history
from support import handle_support_request
from utils import token_counter
from complex_query import handle_complex_query
from dotenv import load_dotenv
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for maximum detail
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="E-commerce Chatbot", layout="wide")

def log_session_state():
    """Log current session state for debugging"""
    logger.debug("="*60)
    logger.debug("SESSION STATE SNAPSHOT:")
    for key, value in st.session_state.items():
        if key == "chat_history":
            logger.debug(f"  {key}: {len(value)} messages")
        elif key in ["qa_chain", "llm", "intent_classifier"]:
            logger.debug(f"  {key}: <object initialized>")
        else:
            logger.debug(f"  {key}: {value}")
    logger.debug("="*60)

def main():
    st.title("🛍️ E-commerce Sales & Support Chatbot")
    logger.info("="*80)
    logger.info("APPLICATION STARTED")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*80)
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        logger.info("✓ Initialized new chat history")
    else:
        logger.info(f"✓ Chat history exists with {len(st.session_state.chat_history)} messages")
    
    if "models_ready" not in st.session_state:
        st.session_state.models_ready = False
        logger.info("✓ Initialized models_ready flag to False")
    
    log_session_state()
    
    # Database connection
    try:
        logger.info("Attempting to initialize database collections...")
        collections = init_collections()
        logger.info("✓ Database collections initialized successfully")
        
        # Log collection counts
        try:
            customer_count = collections['customers'].count_documents({})
            product_count = collections['products'].count_documents({})
            transaction_count = collections['transactions'].count_documents({})
            ticket_count = collections['support_tickets'].count_documents({})
            
            logger.info(f"Database Status:")
            logger.info(f"  - Customers: {customer_count}")
            logger.info(f"  - Products: {product_count}")
            logger.info(f"  - Transactions: {transaction_count}")
            logger.info(f"  - Support Tickets: {ticket_count}")
        except Exception as e:
            logger.warning(f"Could not fetch collection counts: {e}")
            
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}", exc_info=True)
        st.error(f"Database connection error: {e}")
        st.info("Please configure MONGODB_URI in Streamlit secrets")
        return
    
    # Model initialization
    if not st.session_state.get("models_ready", False):
        logger.info("Models not ready - showing upload interface")
        st.info("📊 Ready to upload sales data")
        
        # File upload options
        st.subheader("📁 Upload Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            upload_mode = st.radio(
                "Select upload mode:",
                options=["New File (Clear Existing)", "Existing File (Append)"],
                help="Choose whether to replace existing data or append to it"
            )
            logger.debug(f"Upload mode selected: {upload_mode}")
        
        with col2:
            st.info(f"""
            **{upload_mode.split('(')[0].strip()}**
            
            {'🗑️ Will delete all existing data' if 'New' in upload_mode else '➕ Will keep existing data'}
            
            📝 Only first 100 transactions will be processed
            """)
        
        clear_existing = "New" in upload_mode
        logger.info(f"Upload configuration: clear_existing={clear_existing}")
        
        json_file = st.file_uploader(
            "Upload sales JSON data", 
            type=["json"],
            help="Upload your sales transaction data in JSON format"
        )
        
        if json_file:
            logger.info("="*60)
            logger.info("FILE UPLOADED:")
            logger.info(f"  - Name: {json_file.name}")
            logger.info(f"  - Size: {json_file.size} bytes ({json_file.size / 1024:.2f} KB)")
            logger.info(f"  - Type: {json_file.type}")
            logger.info("="*60)
            
            # Show file info
            st.info(f"📄 File: **{json_file.name}** ({json_file.size / 1024:.2f} KB)")
            
            # Confirmation for new file mode
            if clear_existing:
                st.warning("⚠️ This will delete all existing data in the database!")
                confirm = st.checkbox("I confirm I want to delete existing data")
                logger.debug(f"Delete confirmation checkbox: {confirm}")
                if not confirm:
                    logger.info("User did not confirm deletion - stopping")
                    st.stop()
            
            if st.button("🚀 Process File", type="primary"):
                logger.info("="*80)
                logger.info("FILE PROCESSING INITIATED")
                logger.info(f"  - File: {json_file.name}")
                logger.info(f"  - Mode: {'NEW (clear existing)' if clear_existing else 'APPEND (keep existing)'}")
                logger.info("="*80)
                
                try:
                    with st.spinner("Uploading and processing data..."):
                        # Save temporary file
                        temp_file = "temp.json"
                        logger.info(f"Creating temporary file: {temp_file}")
                        
                        with open(temp_file, "wb") as f:
                            bytes_written = f.write(json_file.getbuffer())
                        logger.info(f"✓ Temporary file created: {bytes_written} bytes written")
                        
                        # Upload to MongoDB
                        logger.info("Starting MongoDB upload process...")
                        uploaded_count = upload_json_to_mongodb(
                            temp_file, 
                            collections,
                            clear_existing=clear_existing
                        )
                        
                        logger.info(f"✓ MongoDB upload completed: {uploaded_count} transactions")
                        st.success(f"✅ Uploaded {uploaded_count} transactions (first 100 from file)")
                        
                        # Clean up
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            logger.info(f"✓ Temporary file deleted: {temp_file}")
                        
                        # Build RAG model
                        st.info("🤖 Building AI models...")
                        logger.info("="*60)
                        logger.info("BUILDING RAG MODELS")
                        logger.info("="*60)
                        
                        api_key = st.secrets.get("GOOGLE_API_KEY")
                        if not api_key:
                            logger.error("❌ GOOGLE_API_KEY not found in secrets")
                            st.error("GOOGLE_API_KEY not found in secrets")
                            return
                        
                        logger.info("✓ Google API key found")
                        logger.info(f"API key length: {len(api_key)} characters")
                        
                        qa_chain, llm, intent_classifier = build_rag_model(
                            api_key, 
                            collections["transactions"]
                        )
                        
                        st.session_state.models_ready = True
                        st.session_state.qa_chain = qa_chain
                        st.session_state.llm = llm
                        st.session_state.intent_classifier = intent_classifier
                        
                        logger.info("✓ Models stored in session state")
                        logger.info("✓ System fully initialized and ready")
                        log_session_state()
                        
                        st.success("🎉 System ready! You can now ask questions.")
                        st.rerun()
                
                except Exception as e:
                    logger.error("="*80)
                    logger.error("FILE PROCESSING ERROR")
                    logger.error(f"Error: {str(e)}")
                    logger.error("="*80, exc_info=True)
                    st.error(f"Error processing file: {str(e)}")
                    return
    
    # Chatbot interface
    if st.session_state.get("models_ready", False):
        logger.debug("Rendering chatbot interface (models ready)")
        st.divider()
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me about products, sales, or support:",
                placeholder="e.g., Show top selling products, What did customer John buy?",
                key="user_input"
            )
        
        with col2:
            if st.button("🔄 Clear History"):
                previous_count = len(st.session_state.chat_history)
                st.session_state.chat_history = []
                logger.info(f"✓ Chat history cleared by user (removed {previous_count} messages)")
                st.rerun()
        
        with col3:
            if st.button("📋 View Logs"):
                st.session_state.show_logs = not st.session_state.get("show_logs", False)
                logger.debug(f"Log viewer toggled: {st.session_state.show_logs}")
        
        # Show logs if enabled
        if st.session_state.get("show_logs", False):
            with st.expander("📋 Application Logs", expanded=True):
                try:
                    with open("chatbot.log", "r", encoding='utf-8') as f:
                        logs = f.readlines()
                        # Show last 100 lines
                        st.text_area(
                            "Recent Logs (Last 100 lines)", 
                            "".join(logs[-100:]), 
                            height=400,
                            disabled=True
                        )
                        st.caption(f"Total log lines: {len(logs)}")
                except FileNotFoundError:
                    logger.warning("Log file not found")
                    st.info("No logs available yet")
        
        if user_input:
            logger.info("="*80)
            logger.info("NEW USER QUERY")
            logger.info(f"Query: '{user_input}'")
            logger.info(f"Query length: {len(user_input)} characters")
            logger.info(f"Timestamp: {datetime.now().isoformat()}")
            logger.info("="*80)
            
            try:
                # Intent classification
                logger.info("Starting intent classification...")
                intent, conf = st.session_state.intent_classifier.classify(user_input)
                logger.info(f"✓ Intent classified: {intent}")
                logger.info(f"✓ Confidence score: {conf:.4f}")
                
                with st.expander(f"🎯 Intent: {intent} (Confidence: {conf:.2f})"):
                    st.write(f"The system identified this as a **{intent}** query")
                
                with st.spinner("Processing..."):
                    logger.info(f"Handling intent: {intent}")
                    
                    if intent == "SEARCH_DB":
                        logger.info("Route: SEARCH_DB (RAG-based database search)")
                        answer = handle_search_db(
                            user_input,
                            st.session_state.qa_chain,
                            st.session_state.chat_history
                        )

                    elif intent == "CUSTOMER_HISTORY":
                        logger.info("Route: CUSTOMER_HISTORY (customer lookup)")
                        answer = handle_customer_history(
                            user_input,
                            st.session_state.llm,
                            collections
                        )

                    elif intent == "SUPPORT":
                        logger.info("Route: SUPPORT (ticket creation)")
                        answer = handle_support_request()
                        
                    elif intent == "COMPLEX_QUERY":
                         logger.info("Route: COMPLEX_QUERY (LLM MongoDB Query)")
                         answer = handle_complex_query(user_input, st.session_state.llm, collections)

                    else:
                        logger.warning(f"Unknown intent received: {intent}")
                        answer = "Sorry, I couldn't understand your request. Please try again."
                
                # Store in chat history
                chat_entry = {
                    "user": user_input,
                    "bot": answer,
                    "intent": intent,
                    "confidence": conf,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(chat_entry)
                
                logger.info(f"✓ Added to chat history (total messages: {len(st.session_state.chat_history)})")
                logger.debug(f"Chat entry: {chat_entry}")
                
                # Display response
                st.write(answer)
                
                # Token counting
                tokens = token_counter.count_tokens(str(answer))
                logger.info(f"Token count: {tokens}")
                st.caption(f"📊 Tokens used: {tokens}")
                
                logger.info("="*80)
                logger.info("QUERY PROCESSING COMPLETED SUCCESSFULLY")
                logger.info("="*80)
            
            except Exception as e:
                logger.error("="*80)
                logger.error("QUERY PROCESSING ERROR")
                logger.error(f"User query: '{user_input}'")
                logger.error(f"Error: {str(e)}")
                logger.error("="*80, exc_info=True)
                st.error(f"Error processing request: {str(e)}")
    
    # Display chat history
    if st.session_state.get("chat_history"):
        logger.debug(f"Rendering chat history ({len(st.session_state.chat_history)} messages)")
        st.divider()
        st.subheader("💬 Chat History")
        
        for i, msg in enumerate(reversed(st.session_state.chat_history), 1):
            idx = len(st.session_state.chat_history) - i + 1
            with st.expander(f"Q{idx}: {msg['user'][:60]}{'...' if len(msg['user']) > 60 else ''}"):
                st.write(f"**Question:** {msg['user']}")
                st.write(f"**Answer:** {msg['bot']}")
                st.caption(f"Intent: {msg['intent']} | Confidence: {msg.get('confidence', 0):.2f}")
                if 'timestamp' in msg:
                    st.caption(f"Time: {msg['timestamp']}")

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("E-COMMERCE CHATBOT APPLICATION")
    logger.info(f"Session Start: {datetime.now().isoformat()}")
    logger.info("="*80 + "\n")
    
    try:
        main()
    except Exception as e:
        logger.critical("="*80)
        logger.critical("CRITICAL APPLICATION ERROR")
        logger.critical(f"Error: {str(e)}")
        logger.critical("="*80, exc_info=True)
        st.error("A critical error occurred. Please check the logs.")
    
    logger.info("\n" + "="*80)
    logger.info("Application session ended")
    logger.info("="*80 + "\n")
