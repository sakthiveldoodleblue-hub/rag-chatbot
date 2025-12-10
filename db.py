import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

def get_mongodb_connection():
    """Establish MongoDB Atlas connection using Streamlit secrets"""
    logger.info("Attempting MongoDB connection...")
    
    try:
        MONGODB_URI = st.secrets.get("MONGODB_URI") or os.getenv("MONGODB_URI")
        DB_NAME = st.secrets.get("DB_NAME", "rag_chatbot_db")
        
        logger.debug(f"Database name: {DB_NAME}")
        
        if not MONGODB_URI:
            logger.error("MONGODB_URI not found in secrets or environment")
            raise ValueError("MONGODB_URI not found in Streamlit secrets or environment")
        
        logger.info(f"MongoDB URI found (length: {len(MONGODB_URI)} chars)")
        logger.info("Connecting to MongoDB Atlas...")
        
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        logger.info("✓ Successfully connected to MongoDB Atlas")
        logger.info(f"✓ Using database: {DB_NAME}")
        
        st.success("✓ Connected to MongoDB Atlas")
        return client[DB_NAME]
    
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}", exc_info=True)
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB connection: {str(e)}", exc_info=True)
        raise

@st.cache_resource
def init_collections():
    """Initialize database collections with indexes"""
    logger.info("Initializing database collections...")
    
    db = get_mongodb_connection()
    
    collections = {
        "transactions": db["transactions"],
        "products": db["products"],
        "customers": db["customers"],
        "support_tickets": db["support_tickets"]
    }
    
    logger.info("Creating indexes for collections...")
    
    try:
        # Create indexes for faster queries
        collections["customers"].create_index("customer_id", unique=True)
        logger.debug("✓ Index created: customers.customer_id")
        
        collections["products"].create_index("product_id", unique=True)
        logger.debug("✓ Index created: products.product_id")
        
        collections["transactions"].create_index("customer_id")
        logger.debug("✓ Index created: transactions.customer_id")
        
        collections["support_tickets"].create_index("ticket_number", unique=True)
        logger.debug("✓ Index created: support_tickets.ticket_number")
        
        logger.info("✓ All indexes created successfully")
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {str(e)}")
    
    logger.info("✓ Collections initialized successfully")
    return collections

