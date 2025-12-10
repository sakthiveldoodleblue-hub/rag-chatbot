import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingIntentClassifier:
    """Classify user intents using embeddings and cosine similarity"""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.intent_templates = {
            "SEARCH_DB": [
                # Product-related queries
                "What products do you have?",
                "Show me sales data",
                "How many items sold?",
                "What is the price of product?",
                "List all products in category",
                "Show inventory",
                "What are the top selling products?",
                "Total sales amount",
                "How much revenue?",
                "Product information",
                "Category details",
                "Stock availability",
                "Product pricing",
                "Sales statistics",
                "Business analytics",
                "Show me transaction history",
                "What are our best sellers?",
                
                # Customer-related queries (general database queries)
                "Which customer purchased"
                "How many customers do we have?",
                "List all customers",
                "Show customer information",
                "What customers are from a specific city?",
                "Customer demographics",
                "Total number of customers",
                "Customer list",
                "Show all customer names",
                "Which customers bought the most?",
                "Customer purchase patterns",
                "Top customers by revenue",
                "Customer details",
                "Search for customer by name",
                "Find customer information",
                "Show customer data",
                "Customer analytics",
                "Customer statistics",
                "Who are our biggest customers?",
                "Customer segmentation data",
                "List customers by location",
                "Customer contact information",
                "Show customer emails",
                "Customer phone numbers"
            ], 
            "CUSTOMER_HISTORY": [
                "Show my purchase history",
                "What did I buy?",
                "My previous orders",
                "My transaction history",
                "Orders for customer John Doe",
                "Show transactions for customer ID",
                "My account purchases",
                "What have I ordered before?",
                "My past invoices",
                "Show my receipts",
                "Customer order history",
                "My shopping history",
                "Previous purchases",
                "Order history for email",
                "Track my orders",
                "Look up customer orders",
                "Find customer transactions",
                "What did customer X purchase?",
                "Show orders for this customer",
                "Customer purchase history"
            ],
            "SUPPORT": [
                "I have a problem",
                "Need help with my order",
                "Product is broken",
                "Issue with delivery",
                "Customer service needed",
                "Contact support team",
                "File a complaint",
                "Not working properly",
                "Report an issue",
                "Need assistance",
                "Something went wrong",
                "Call customer care",
                "Urgent help required",
                "Refund request",
                "Product defect",
                "Create support ticket",
                "I need support"
            ],
            "COMPLEX_QUERY": [
                # Top customers
                "Show top 10 customers by revenue",
                "Which customers spent the most?",
                "Who are the high-value customers?",
                
                # Category analysis
                "How much revenue did each category generate?",
                "Which category has the most sales?",
                "Show sales breakdown by category",
                
                # Location analysis
                "Which store location performed best?",
                "Compare sales across locations",
                "Show revenue by store location",
                
                # Product analysis
                "What are the top 10 best-selling products?",
                "Which products have the highest profit?",
                "Show product sales performance",
                
                # Customer analysis
                "How many customers bought in multiple categories?",
                "Find customers in specific cities",
                "Show customer distribution by loyalty tier",
                
                # Payment analysis
                "Which payment method is most used?",
                "Show payment mode preferences",
                "Compare revenue by payment method",
                
                # Time-based analysis
                "Show sales trends over time",
                "Compare sales this month vs last month",
                "Show daily/weekly sales patterns",
                
                # Custom queries
                "Find customers who spent more than 10000",
                "Show orders with discount above 20%",
                "List products in electronics with price above 5000",
                "Show customers from specific cities",
                "Find transactions with GST above 1000",
]
        }
        
        print("\n🧠 Pre-computing intent template embeddings...")
        self.intent_embeddings = {}
        
        try:
            for intent, templates in self.intent_templates.items():
                embeddings = self.embeddings_model.embed_documents(templates)
                self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
            
            print(f"✅ Intent templates loaded: {list(self.intent_templates.keys())}")
        
        except Exception as e:
            print(f"Error initializing intent classifier: {str(e)}")
            raise

    def classify(self, question):
        """Classify question intent and return intent and confidence"""
        
        try:
            question_embedding = self.embeddings_model.embed_query(question)
            
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = cosine_similarity(
                    [question_embedding],
                    [intent_embedding]
                )[0][0]
                similarities[intent] = float(similarity)
            
            best_intent = max(similarities, key=similarities.get)
            best_score = similarities[best_intent]
            
            return best_intent, best_score
        
        except Exception as e:
            print(f"Error during intent classification: {str(e)}")

            return "SEARCH_DB", 0.5
