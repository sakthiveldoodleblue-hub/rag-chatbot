from datetime import datetime
from pathlib import Path
import json
import streamlit as st
import logging
from pymongo.errors import BulkWriteError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def upload_json_to_mongodb(json_file_path: str, collections, clear_existing: bool = True) -> int:
    """Upload and parse JSON file into MongoDB collections
    
    Args:
        json_file_path: Path to JSON file
        collections: MongoDB collections dict
        clear_existing: If True, delete existing data before upload
    
    Returns:
        Number of transactions uploaded
    """
    
    logger.info(f"Starting upload process for file: {json_file_path}")
    logger.info(f"Clear existing data: {clear_existing}")
    
    if not Path(json_file_path).exists():
        logger.error(f"File not found: {json_file_path}")
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    try:
        # Read JSON file
        logger.info("Reading JSON file...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        documents = data if isinstance(data, list) else [data]
        logger.info(f"Total documents in file: {len(documents)}")
        
        # Limit to first 100 documents
        documents = documents[:100]
        logger.info(f"Processing first 100 documents (actual: {len(documents)})")
        
        if not documents:
            logger.error("No documents found in JSON file")
            raise ValueError("No documents found in JSON file")
        
        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing data from collections...")
            customers_deleted = collections["customers"].delete_many({}).deleted_count
            products_deleted = collections["products"].delete_many({}).deleted_count
            transactions_deleted = collections["transactions"].delete_many({}).deleted_count
            logger.info(f"Deleted: {customers_deleted} customers, {products_deleted} products, {transactions_deleted} transactions")
        else:
            logger.info("Keeping existing data (append mode)")
        
        # Initialize storage dicts
        customers_dict = {}
        products_dict = {}
        transactions = []
        
        # Process each document
        logger.info("Processing documents...")
        processed_count = 0
        error_count = 0
        
        for idx, doc in enumerate(documents, 1):
            try:
                # Extract customer data
                cid = str(doc.get("Customer ID", "UNKNOWN")).strip()
                if cid and cid not in customers_dict:
                    customers_dict[cid] = {
                        "customer_id": cid,
                        "name": str(doc.get("Customer name", "Unknown")).strip(),
                        "email": str(doc.get("Email", "N/A")).strip(),
                        "phone": str(doc.get("Phone", "N/A")).strip(),
                        "city": str(doc.get("City", "N/A")).strip(),
                        "loyalty_tier": str(doc.get("Loyalty_Tier", "Regular")).strip(),
                        "created_at": datetime.now()
                    }
                    logger.debug(f"Added new customer: {cid}")
                
                # Extract product data
                pid = str(doc.get("ID_product", "UNKNOWN")).strip()
                if pid and pid not in products_dict:
                    products_dict[pid] = {
                        "product_id": pid,
                        "name": str(doc.get("Product", "Unknown")).strip(),
                        "category": str(doc.get("Category", "N/A")).strip(),
                        "sku": str(doc.get("SKUs", "N/A")).strip(),
                        "cogs": float(doc.get("COGS", 0)) if doc.get("COGS") else 0,
                        "margin_percent": float(doc.get("Margin_per_piece_percent", 0)) 
                                        if doc.get("Margin_per_piece_percent") else 0,
                        "created_at": datetime.now()
                    }
                    logger.debug(f"Added new product: {pid}")
                
                # Extract transaction data
                transactions.append({
                    "invoice_number": str(doc.get("Invoice Number", "N/A")).strip(),
                    "txn_number": str(doc.get("Txn_No", "N/A")).strip(),
                    "customer_id": cid,
                    "customer_name": str(doc.get("Customer name", "Unknown")).strip(),
                    "product_id": pid,
                    "product_name": str(doc.get("Product", "Unknown")).strip(),
                    "category": str(doc.get("Category", "N/A")).strip(),
                    "quantity": int(doc.get("Quantity_piece", 0)) if doc.get("Quantity_piece") else 0,
                    "gross_amount": float(doc.get("Gross_Amount", 0)) if doc.get("Gross_Amount") else 0,
                    "discount_percentage": float(doc.get("Discount_Percentage", 0)) 
                                          if doc.get("Discount_Percentage") else 0,
                    "total_amount": float(doc.get("Total Amount", 0)) if doc.get("Total Amount") else 0,
                    "gst": float(doc.get("GST", 0)) if doc.get("GST") else 0,
                    "payment_mode": str(doc.get("Payment_mode", "N/A")).strip(),
                    "date_of_purchase": str(doc.get("Date_of_purchase", datetime.now().isoformat())).strip(),
                    "channel": str(doc.get("Channel", "N/A")).strip(),
                    "store_location": str(doc.get("Store_location", "N/A")).strip(),
                    "mode": str(doc.get("Mode", "N/A")).strip(),
                    "status": "completed",
                    "created_at": datetime.now()
                })
                processed_count += 1
                
                if idx % 25 == 0:
                    logger.info(f"Processed {idx}/{len(documents)} documents...")
            
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing document {idx}: {str(e)}")
                continue
        
        logger.info(f"Document processing complete. Success: {processed_count}, Errors: {error_count}")
        
        # Insert data into MongoDB with upsert for customers and products
        inserted_counts = {"customers": 0, "products": 0, "transactions": 0}
        
        # UPSERT customers (update if exists, insert if new)
        if customers_dict:
            logger.info(f"Upserting {len(customers_dict)} customers...")
            upserted_count = 0
            updated_count = 0
            
            for customer in customers_dict.values():
                try:
                    result = collections["customers"].update_one(
                        {"customer_id": customer["customer_id"]},
                        {"$set": customer},
                        upsert=True
                    )
                    if result.upserted_id:
                        upserted_count += 1
                    elif result.modified_count > 0:
                        updated_count += 1
                except Exception as e:
                    logger.warning(f"Error upserting customer {customer['customer_id']}: {e}")
            
            inserted_counts["customers"] = upserted_count
            logger.info(f"✓ Customers - Inserted: {upserted_count}, Updated: {updated_count}")
        
        # UPSERT products (update if exists, insert if new)
        if products_dict:
            logger.info(f"Upserting {len(products_dict)} products...")
            upserted_count = 0
            updated_count = 0
            
            for product in products_dict.values():
                try:
                    result = collections["products"].update_one(
                        {"product_id": product["product_id"]},
                        {"$set": product},
                        upsert=True
                    )
                    if result.upserted_id:
                        upserted_count += 1
                    elif result.modified_count > 0:
                        updated_count += 1
                except Exception as e:
                    logger.warning(f"Error upserting product {product['product_id']}: {e}")
            
            inserted_counts["products"] = upserted_count
            logger.info(f"✓ Products - Inserted: {upserted_count}, Updated: {updated_count}")
        
        # INSERT transactions (always insert new transactions)
        if transactions:
            logger.info(f"Inserting {len(transactions)} transactions...")
            try:
                result = collections["transactions"].insert_many(transactions, ordered=False)
                inserted_counts["transactions"] = len(result.inserted_ids)
                logger.info(f"✓ Inserted {inserted_counts['transactions']} transactions")
            except BulkWriteError as e:
                # Some transactions may have been inserted before error
                inserted_counts["transactions"] = e.details.get('nInserted', 0)
                logger.warning(f"Partial transaction insert: {inserted_counts['transactions']} succeeded")
                logger.error(f"BulkWriteError: {e.details}")
        
        logger.info(f"Upload complete! Total transactions: {inserted_counts['transactions']}")
        return inserted_counts["transactions"]
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}", exc_info=True)
        raise Exception(f"Error uploading data: {str(e)}")
