import streamlit as st
import pandas as pd
import logging
import json
from pymongo import MongoClient
import re

logger = logging.getLogger(__name__)

class MongoQueryGenerator:
    """Generate MongoDB queries from natural language using LLM"""
    
    def __init__(self, collections, llm):
        self.collections = collections
        self.llm = llm
        self.transactions_df = None
        logger.info("MongoQueryGenerator initialized")
    
    def get_database_schema(self):
        """Get schema information from MongoDB collections"""
        try:
            logger.info("Fetching database schema...")
            
            schema_info = """
MONGODB DATABASE SCHEMA:
=======================

COLLECTIONS AND FIELDS:

1. transactions:
   - _id: ObjectId
   - invoice_number: string
   - txn_number: string
   - customer_id: string
   - customer_name: string
   - product_id: string
   - product_name: string
   - category: string
   - quantity: number
   - gross_amount: number
   - discount_percentage: number
   - total_amount: number
   - gst: number
   - payment_mode: string
   - date_of_purchase: date
   - channel: string
   - store_location: string
   - mode: string
   - status: string
   - created_at: date

2. customers:
   - _id: ObjectId
   - customer_id: string (unique)
   - name: string
   - email: string
   - phone: string
   - city: string
   - loyalty_tier: string
   - created_at: date

3. products:
   - _id: ObjectId
   - product_id: string (unique)
   - name: string
   - category: string
   - sku: string
   - cogs: number
   - margin_percent: number
   - created_at: date

4. support_tickets:
   - _id: ObjectId
   - ticket_number: string (unique)
   - customer_name: string
   - customer_email: string
   - category: string
   - issue: string
   - priority: string
   - status: string
   - created_at: date
   - updated_at: date

COMMON MONGODB OPERATIONS:
- db.transactions.find(query): Find documents matching query
- db.transactions.aggregate([stages]): Pipeline aggregation
- db.transactions.countDocuments(query): Count matching documents
- db.transactions.distinct(field): Get unique values
- $match: Filter documents
- $group: Group documents
- $sort: Sort results
- $limit: Limit results
- $project: Select fields
- $sum: Sum values
- $avg: Calculate average
- $max/$min: Get max/min values
"""
            return schema_info
        except Exception as e:
            logger.error(f"Error fetching schema: {e}")
            return ""
    
    def generate_query(self, question, schema_info):
        """Use LLM to generate MongoDB query from natural language"""
        try:
            logger.info(f"Generating query for: {question}")
            
            prompt = f"""You are a MongoDB query expert. Convert the user's natural language question into a MongoDB query.

{schema_info}

USER QUESTION: {question}

INSTRUCTIONS:
1. Generate a VALID MongoDB query (in Python dict format for MongoDB)
2. Return ONLY the query, no explanation
3. Use $match, $group, $sort, $limit, $project stages in aggregation pipeline when needed
4. For simple queries, use db.collection.find() format
5. For complex queries, use aggregation pipeline
6. Include sorting and limiting when appropriate
7. Format the query as valid Python code

QUERY FORMAT EXAMPLES:

For simple find:
{{"query_type": "find", "collection": "transactions", "query": {{"category": "Electronics"}}, "projection": {{"product_name": 1, "total_amount": 1}}}}

For aggregation:
{{"query_type": "aggregate", "collection": "transactions", "pipeline": [{{"$match": {{"category": "Electronics"}}}}, {{"$group": {{"_id": "$customer_name", "total": {{"$sum": "$total_amount"}}}}}}, {{"$sort": {{"total": -1}}}}, {{"$limit": 10}}]}}

GENERATE THE QUERY:"""
            
            from langchain_core.messages import HumanMessage
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            query_text = response.content.strip()
            
            logger.info(f"Generated query text: {query_text[:200]}...")
            
            # Parse the query
            try:
                query_dict = json.loads(query_text)
                logger.info(f"Query parsed successfully: {query_dict.get('query_type')}")
                return query_dict
            except json.JSONDecodeError:
                logger.error("Failed to parse query as JSON")
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', query_text, re.DOTALL)
                if json_match:
                    query_dict = json.loads(json_match.group())
                    return query_dict
                raise ValueError("Could not parse LLM response as valid JSON")
        
        except Exception as e:
            logger.error(f"Error generating query: {e}", exc_info=True)
            raise
    
    def execute_find_query(self, query_dict):
        """Execute a find() query"""
        try:
            logger.info("Executing find query...")
            
            collection_name = query_dict.get('collection')
            find_query = query_dict.get('query', {})
            projection = query_dict.get('projection', None)
            limit = query_dict.get('limit', 100)
            
            logger.info(f"Collection: {collection_name}, Query: {find_query}")
            
            collection = self.collections[collection_name]
            
            if projection:
                results = list(collection.find(find_query, projection).limit(limit))
            else:
                results = list(collection.find(find_query).limit(limit))
            
            logger.info(f"Found {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Error executing find query: {e}", exc_info=True)
            raise
    
    def execute_aggregate_query(self, query_dict):
        """Execute an aggregation pipeline"""
        try:
            logger.info("Executing aggregation query...")
            
            collection_name = query_dict.get('collection')
            pipeline = query_dict.get('pipeline', [])
            
            logger.info(f"Collection: {collection_name}, Pipeline stages: {len(pipeline)}")
            logger.info(f"Pipeline: {pipeline}")
            
            collection = self.collections[collection_name]
            results = list(collection.aggregate(pipeline))
            
            logger.info(f"Aggregation returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error executing aggregation query: {e}", exc_info=True)
            raise
    
    def execute_query(self, query_dict):
        """Execute the generated query"""
        try:
            query_type = query_dict.get('query_type')
            
            if query_type == 'find':
                results = self.execute_find_query(query_dict)
            elif query_type == 'aggregate':
                results = self.execute_aggregate_query(query_dict)
            else:
                raise ValueError(f"Unknown query type: {query_type}")
            
            logger.info(f"Query executed successfully, got {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            raise
    
    def generate_text_answer(self, question, results, llm):
        """Use LLM to generate text answer from results"""
        try:
            logger.info("Generating text answer from results...")
            
            # Convert results to JSON for context
            results_json = json.dumps(results, indent=2, default=str)
            
            prompt = f"""You are a business analyst. Based on the database query results provided, give a clear, concise text answer to the user's question.

USER QUESTION: {question}

DATABASE QUERY RESULTS:
{results_json}

INSTRUCTIONS:
1. Answer the question directly using the results
2. Provide specific numbers, names, and values from the results
3. Format the answer in clear, readable text
4. Use bullet points or numbers only if necessary
5. Include insights and observations
6. Keep it concise and professional
7. Do not show raw data - present information naturally

ANSWER:"""
            
            logger.info("Calling LLM to generate text answer...")
            from langchain_core.messages import HumanMessage
            
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
            
            logger.info("Text answer generated successfully")
            return answer
        
        except Exception as e:
            logger.error(f"Error generating text answer: {e}", exc_info=True)
            raise
    
    def process_question(self, question, llm):
        """Process user question and return text answer"""
        try:
            logger.info("="*60)
            logger.info(f"Processing question: {question}")
            logger.info("="*60)
            
            # Get schema
            logger.info("Step 1: Getting database schema...")
            schema_info = self.get_database_schema()
            
            # Generate query
            logger.info("Step 2: Generating MongoDB query...")
            query_dict = self.generate_query(question, schema_info)
            
            logger.info(f"Step 3: Generated query type: {query_dict.get('query_type')}")
            
            # Show generated query for debugging
            with st.expander("🔍 View Generated Query"):
                st.code(json.dumps(query_dict, indent=2), language="json")
            
            # Execute query
            logger.info("Step 4: Executing query...")
            results = self.execute_query(query_dict)
            
            logger.info(f"Step 5: Query returned {len(results)} results")
            
            # Generate text answer
            logger.info("Step 6: Generating text answer...")
            answer = self.generate_text_answer(question, results, llm)
            
            logger.info("="*60)
            
            return answer
        
        except Exception as e:
            logger.error(f"Error in process_question: {e}", exc_info=True)
            raise


def handle_complex_query(question, llm, collections):
    """Handle complex queries using LLM-generated MongoDB queries"""
    
    st.header("📊 Query Analysis - MongoDB")
    logger.info(f"Handling complex query: {question}")
    
    try:
        engine = MongoQueryGenerator(collections, llm)
        
        with st.spinner("Analyzing question and generating query..."):
            logger.info("Starting query processing...")
            answer = engine.process_question(question, llm)
        
        # Display text answer only
        if answer:
            logger.info("Displaying answer")
            st.write(answer)
        
        else:
            st.info("No results found for your query")
            logger.info("Query returned empty results")
            return "No results found for your query"
    
    except Exception as e:
        logger.error(f"Error handling complex query: {e}", exc_info=True)
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        return error_msg