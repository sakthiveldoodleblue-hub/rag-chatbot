import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class TokenCounter:
    """Count tokens using tiktoken for accurate token usage"""
    
    def __init__(self):
        logger.debug("Initializing TokenCounter...")
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.info("✓ TokenCounter initialized with cl100k_base encoding")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}")
            logger.warning("Falling back to estimation method")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            if self.encoding:
                token_count = len(self.encoding.encode(text))
                logger.debug(f"Token count (tiktoken): {token_count}")
                return token_count
            # Fallback: rough estimation (1 token ≈ 4 characters)
            estimated_count = len(text) // 4
            logger.debug(f"Token count (estimated): {estimated_count}")
            return estimated_count
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            estimated_count = len(text) // 4
            logger.debug(f"Token count (fallback): {estimated_count}")
            return estimated_count

token_counter = TokenCounter()

def mongodb_to_searchable_text(transactions_collection):
    """Convert MongoDB transactions to searchable text chunks"""
    
    logger.info("Converting MongoDB transactions to searchable text...")
    
    try:
        logger.debug("Fetching transactions from MongoDB (limit: 200)...")
        transactions = list(transactions_collection.find().limit(200))
        
        if not transactions:
            logger.error("No transactions found in MongoDB")
            raise ValueError("No transactions found in MongoDB")
        
        logger.info(f"✓ Retrieved {len(transactions)} transactions")
        
        # Create searchable text from transactions
        texts = []
        logger.debug("Converting transactions to text format...")
        
        for idx, txn in enumerate(transactions, 1):
            text = f"""
Transaction Details:
Invoice Number: {txn.get("invoice_number", "N/A")}
Transaction Number: {txn.get("txn_number", "N/A")}
Customer: {txn.get("customer_name", "Unknown")} (ID: {txn.get("customer_id", "N/A")})
Email: {txn.get("customer_email", "N/A") if "customer_email" in txn else "N/A"}
Product: {txn.get("product_name", "Unknown")} (ID: {txn.get("product_id", "N/A")})
Category: {txn.get("category", "N/A")}
Quantity Purchased: {txn.get("quantity", 0)} units
Gross Amount: ${txn.get("gross_amount", 0):.2f}
Discount: {txn.get("discount_percentage", 0)}%
Total Amount: ${txn.get("total_amount", 0):.2f}
GST: ${txn.get("gst", 0):.2f}
Payment Mode: {txn.get("payment_mode", "N/A")}
Purchase Date: {txn.get("date_of_purchase", "N/A")}
Channel: {txn.get("channel", "N/A")}
Store Location: {txn.get("store_location", "N/A")}
Status: {txn.get("status", "N/A")}
"""
            texts.append(text)
            
            if idx % 50 == 0:
                logger.debug(f"Processed {idx}/{len(transactions)} transactions...")
        
        logger.info(f"✓ Converted {len(texts)} transactions to text")
        
        # Split text into chunks for better retrieval
        logger.info("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.debug("Performing text splitting...")
        combined_text = "\n".join(texts)
        logger.debug(f"Combined text length: {len(combined_text)} characters")
        
        chunks = splitter.split_text(combined_text)
        
        if not chunks:
            logger.error("No chunks created from transactions")
            raise ValueError("No chunks created from transactions")
        
        logger.info(f"✓ Created {len(chunks)} text chunks")
        logger.debug(f"Chunk size range: {min(len(c) for c in chunks)} - {max(len(c) for c in chunks)} chars")
        logger.debug(f"First chunk preview: {chunks[0][:200]}...")
        
        return chunks
    
    except Exception as e:
        logger.error("="*60)
        logger.error("Error converting MongoDB to searchable text")
        logger.error(f"Error: {str(e)}")
        logger.error("="*60, exc_info=True)
        raise
