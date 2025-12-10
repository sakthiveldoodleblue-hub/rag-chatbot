import streamlit as st
import logging

logger = logging.getLogger(__name__)

def handle_search_db(question, qa_chain, chat_history):
    """Handle database search using RAG chain for both products and customers"""
    
    try:
        logger.info(f"Search DB Query: {question}")
        logger.info(f"Chat history length: {len(chat_history) if chat_history else 0}")
        
        # Validate inputs
        if question is None:
            question = ""
        
        question = str(question).strip()
        
        if not question:
            error_msg = "Please enter a valid question"
            logger.error(error_msg)
            st.error(error_msg)
            return error_msg
        
        if qa_chain is None:
            error_msg = "QA chain is not initialized. Please upload data first."
            logger.error(error_msg)
            st.error(error_msg)
            return error_msg
        
        logger.info(f"Processing question: {question}")
        
        # Prepare chat history - SAFE VERSION
        formatted_history = []
        
        try:
            if chat_history and isinstance(chat_history, list):
                for item in chat_history[-5:]:  # Last 5 exchanges
                    try:
                        if not isinstance(item, dict):
                            continue
                        
                        user_msg = item.get("user")
                        bot_msg = item.get("bot")
                        
                        # Convert None to empty string, then to string
                        user_msg = "" if user_msg is None else str(user_msg).strip()
                        bot_msg = "" if bot_msg is None else str(bot_msg).strip()
                        
                        # Only add if both exist
                        if user_msg and bot_msg:
                            formatted_history.append((user_msg, bot_msg))
                    
                    except Exception as e:
                        logger.warning(f"Error processing chat history item: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error preparing chat history: {e}")
            formatted_history = []
        
        logger.info(f"Formatted history: {len(formatted_history)} items")
        
        # Invoke QA chain
        try:
            logger.info("Invoking QA chain...")
            
            invoke_input = {
                "question": question,
                "chat_history": formatted_history
            }
            
            logger.info(f"Invoke input - Question: {question}, History items: {len(formatted_history)}")
            
            result = qa_chain.invoke(invoke_input)
            
            logger.info("QA chain invoked successfully")
        
        except Exception as e:
            logger.error(f"Error invoking QA chain: {e}", exc_info=True)
            return f"Error during analysis: {str(e)}"
        
        # Extract answer SAFELY
        answer = "No data found"
        
        try:
            if result is None:
                answer = "No data found"
                logger.warning("QA chain returned None")
            
            elif isinstance(result, dict):
                answer = result.get("answer")
                if answer is None:
                    answer = "No data found"
                else:
                    answer = str(answer).strip()
            
            elif isinstance(result, str):
                answer = result.strip()
            
            else:
                answer = str(result).strip() if result else "No data found"
        
        except Exception as e:
            logger.error(f"Error extracting answer: {e}", exc_info=True)
            answer = "Error processing response"
        
        logger.info(f"Answer obtained - Length: {len(answer)}")
        
        # Display source documents SAFELY
        try:
            if isinstance(result, dict):
                source_docs = result.get("source_documents")
                
                if source_docs and isinstance(source_docs, list):
                    logger.info(f"Found {len(source_docs)} source documents")
                    
                    with st.expander("📚 Source Documents", expanded=False):
                        for i, doc in enumerate(source_docs, 1):
                            try:
                                # Get content safely
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                else:
                                    content = str(doc)
                                
                                # Convert to string safely
                                if content is None:
                                    content = ""
                                else:
                                    content = str(content).strip()
                                
                                if content:
                                    # Truncate if too long
                                    if len(content) > 400:
                                        display_content = content[:400]
                                        st.write(f"**Document {i}:**")
                                        st.text(display_content)
                                        st.caption("... (truncated)")
                                    else:
                                        st.write(f"**Document {i}:**")
                                        st.text(content)
                                    
                                    st.divider()
                            
                            except Exception as e:
                                logger.warning(f"Error displaying doc {i}: {e}")
                                continue
        
        except Exception as e:
            logger.warning(f"Error displaying source documents: {e}")
        
        logger.info("Search completed successfully")
        return answer
    
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return error_msg