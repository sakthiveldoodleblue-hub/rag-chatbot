import streamlit as st
from datetime import datetime
from db import init_collections

def handle_support_request():
    """Handle support ticket creation with validation"""
    
    st.header("🆘 Create Support Ticket")
    
    with st.form("support_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", placeholder="John Doe")
        
        with col2:
            email = st.text_input("Email", placeholder="john@example.com")
        
        category = st.selectbox(
            "Issue Category",
            ["Order Issue", "Product Defect", "Delivery Problem", "Billing", "Other"]
        )
        
        issue = st.text_area(
            "Describe your issue",
            placeholder="Please provide details about your issue...",
            height=150
        )
        
        priority = st.select_slider(
            "Priority Level",
            options=["Low", "Normal", "High", "Urgent"],
            value="Normal"
        )
        
        submitted = st.form_submit_button("📤 Submit Ticket", use_container_width=True)
        
        if submitted:
            # Validation
            if not name or not name.strip():
                st.error("❌ Please enter your name")
                return "Name is required"
            
            if not email or "@" not in email:
                st.error("❌ Please enter a valid email")
                return "Valid email is required"
            
            if not issue or not issue.strip():
                st.error("❌ Please describe your issue")
                return "Issue description is required"
            
            try:
                collections = init_collections()
                
                ticket = {
                    "ticket_number": f"TKT-{int(datetime.now().timestamp())}",
                    "customer_name": name.strip(),
                    "customer_email": email.strip(),
                    "category": category,
                    "issue": issue.strip(),
                    "priority": priority.lower(),
                    "status": "open",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                result = collections['support_tickets'].insert_one(ticket)
                
                st.success(f"✅ Ticket created successfully!")
                st.info(f"**Ticket Number:** {ticket['ticket_number']}")
                st.info(f"**Status:** Open - Our team will respond soon")
                st.balloons()
                
                return f"Support ticket {ticket['ticket_number']} created successfully"
            
            except Exception as e:
                st.error(f"❌ Error creating ticket: {str(e)}")
                return f"Error: {str(e)}"
    
    return "Fill out the form to create a support ticket"
