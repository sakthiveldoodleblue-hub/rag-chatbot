import streamlit as st
from db import init_collections

def handle_customer_history(question, llm, collections):
    """Handle customer history queries with improved search"""
    
    st.header("👤 Customer Purchase History")
    
    # Get collections
    customers_col = collections['customers']
    transactions_col = collections['transactions']
    
    # Search input
    search_term = st.text_input(
        "🔍 Enter Customer ID/Name/Email:",
        placeholder="e.g., john@example.com or CUST001"
    )
    
    if not search_term:
        return "Please enter customer details to search"
    
    try:
        # Enhanced search with multiple fields
        customer = customers_col.find_one({
            "$or": [
                {"name": {"$regex": search_term, "$options": "i"}},
                {"customer_id": {"$regex": search_term, "$options": "i"}},
                {"email": {"$regex": search_term, "$options": "i"}},
                {"phone": {"$regex": search_term, "$options": "i"}}
            ]
        })
        
        if not customer:
            return f"❌ No customer found matching '{search_term}'"
        
        customer_id = customer["customer_id"]
        
        # Get transactions
        txns = list(transactions_col.find(
            {"customer_id": customer_id}
        ).sort("date_of_purchase", -1))
        
        if not txns:
            st.warning(f"No transactions found for {customer['name']}")
            return f"Customer {customer['name']} has no purchase history"
        
        # Display customer info
        st.subheader(f"Customer: {customer['name']}")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(txns))
        
        with col2:
            total_spent = sum(t.get("total_amount", 0) for t in txns)
            st.metric("Total Spent", f"${total_spent:,.2f}")
        
        with col3:
            st.metric("Email", customer.get("email", "N/A"))
        
        with col4:
            st.metric("Loyalty Tier", customer.get("loyalty_tier", "Regular"))
        
        # Display transactions
        st.subheader("📋 Recent Transactions")
        
        display_data = []
        for txn in txns:
            display_data.append({
                "Date": txn.get("date_of_purchase", "N/A"),
                "Invoice": txn.get("invoice_number", "N/A"),
                "Product": txn.get("product_name", "N/A"),
                "Category": txn.get("category", "N/A"),
                "Quantity": txn.get("quantity", 0),
                "Amount": f"${txn.get('total_amount', 0):,.2f}",
                "Status": txn.get("status", "N/A")
            })
        
        st.dataframe(display_data, use_container_width=True)
        
        return f"✅ Found {len(txns)} transactions for {customer['name']}"
    
    except Exception as e:
        st.error(f"Error searching customer: {str(e)}")
        return f"Error: {str(e)}"
