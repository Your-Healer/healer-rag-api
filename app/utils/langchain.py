from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from typing import List
from langchain.schema import Document

from app.config import logger, RAG_OPENAI_API_KEY, RAG_OPENAI_BASEURL, RAG_OPENAI_PROXY
from app.utils.vector_store_init import ensure_vector_store_initialized
from app.utils.enhanced_retrieval import EnhancedRetrieval

prompt_template = """
    You are an advanced medical assistant specializing in Vietnamese diseases with access to a comprehensive medical database. Your purpose is to provide accurate, helpful, and contextually relevant information.

    You have two primary functions:
    1. DISEASE INFORMATION: When users ask about specific diseases, provide detailed, organized information
    2. DISEASE PREDICTION: When users describe symptoms, analyze them and suggest potential diseases that match

    CONTEXT INFORMATION:
    {context}

    USER QUERY:
    {question}

    RESPONSE GUIDELINES:
    - Respond ONLY in {input_language} language
    - Be compassionate but professional in your tone
    - For serious conditions, encourage seeking professional medical advice
    - NEVER invent or hallucinate information not present in the context provided
    - If information is incomplete or unavailable, acknowledge limitations clearly
    - Avoid medical jargon when possible; explain specialized terms if necessary
    - Maintain cultural sensitivity relevant to Vietnamese medical practices

    FOR DISEASE INFORMATION QUERIES:
    - Provide a concise definition of the disease first
    - Organize information into clear sections: Symptoms, Causes, Diagnosis, Treatment, Prevention
    - Include common local names or alternative names used in Vietnam if available
    - Highlight important warning signs that require urgent medical attention
    - Mention relevant statistics or epidemiology specific to Vietnam when available

    FOR SYMPTOM-BASED QUERIES:
    - Summarize and confirm the symptoms described by the user
    - List potential diseases in order of likelihood based on the symptoms
    - For each disease suggestion, clearly explain WHY it matches the symptoms described
    - Include severity assessment and urgency indicators when appropriate
    - Suggest what additional symptoms might help narrow down the diagnosis
    - Recommend appropriate types of medical specialists to consult if needed
    - Provide practical next steps based on symptom severity
    
    YOUR RESPONSE ({input_language}):
    """

def create_llm(temperature: float = 0, max_tokens: int = 2000, model: str = "gpt-4.1-mini"):
    """Create a language model for RAG responses"""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=RAG_OPENAI_API_KEY,
        openai_api_base=RAG_OPENAI_BASEURL,
        openai_proxy=RAG_OPENAI_PROXY
    )

def create_prompt(input_language: str = 'vietnamese'):
    """Create prompt template with the specified language"""
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={"input_language": input_language}
    )

async def create_async_rag_chain(input_language: str = 'vietnamese', k: int = 5):
    """Create a RAG chain for query processing with async initialization"""
    try:
        # Ensure vector store is initialized
        await ensure_vector_store_initialized()
        
        llm = create_llm()
        prompt = create_prompt(input_language)
        
        # Make sure we're using the properly initialized vector store from config
        from app.config import vector_store
        
        standard_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=standard_retriever,
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_separator": "\n\n---\n\n",
                "verbose": True
            }
        )

        enhanced_retriever = EnhancedRetrieval(vector_store)
   
        return qa_chain, enhanced_retriever
    except Exception as e:
        print(f"Error creating async RAG chain: {str(e)}")
        raise

def is_symptom_query(query: str, language: str = "vietnamese") -> bool:
    """
    Determine if a query is asking about symptoms to predict a disease.
    """
    # Common phrases that indicate symptom queries in Vietnamese
    vn_symptom_phrases = [
        "triệu chứng", "dấu hiệu", "bị đau", "cảm thấy", "có những", 
        "tôi bị", "tôi có", "tôi đang", "tôi thấy", "bệnh gì",
        "mắc bệnh", "mắc phải", "tôi bị làm sao", "tôi bị sao",
        "chẩn đoán", "có phải tôi bị", "nghi ngờ bị", "đau", "nhức",
        "sốt", "nóng", "lạnh", "phát ban", "nổi mẩn", "buồn nôn",
        "nôn", "mệt mỏi", "mệt", "khó thở", "ho", "đau đầu",
        "chóng mặt", "ăn không ngon"
    ]
    
    # Common phrases that indicate symptom queries in English
    en_symptom_phrases = [
        "symptom", "sign", "feel", "having", "suffering from",
        "i have", "i am", "i feel", "what disease", "diagnose",
        "do i have", "might i have", "could i have", "is it possible",
        "pain", "ache", "fever", "hot", "cold", "rash", "nausea",
        "vomiting", "tired", "fatigue", "breathing", "cough", "headache",
        "dizzy", "no appetite", "hurts", "sore", "swollen", "swelling"
    ]
    
    symptom_phrases = vn_symptom_phrases if language == "vietnamese" else en_symptom_phrases
    
    # Check if any of the phrases are in the query
    lower_query = query.lower()
    return any(phrase in lower_query for phrase in symptom_phrases)

def enhance_query(query: str, language: str = "vietnamese") -> str:
    """
    Enhance the query based on its type and language.
    """
    if is_symptom_query(query, language):
        if language == "vietnamese":
            return f"Phân tích các triệu chứng sau và dự đoán bệnh có thể mắc phải: {query}"
        else:
            return f"Analyze the following symptoms and predict possible diseases: {query}"
    return query
    
def retrieve_relevant_documents(query: str, retriever=None, enhanced_retriever=None, language: str = "vietnamese", k: int = 5) -> List[Document]:
    """
    Retrieve relevant documents based on the query.
    
    Args:
        query (str): The user query
        retriever: The standard retriever object
        enhanced_retriever: The enhanced retriever object (optional)
        language (str): The language of the query
        k (int): Number of documents to retrieve
        
    Returns:
        List[Document]: List of relevant documents
    """
    try:
        symptom_based = is_symptom_query(query, language)

        processed_query = query
        if symptom_based:
            if language == "vietnamese":
                processed_query = f"Triệu chứng: {query}"
            else:
                processed_query = f"Symptoms: {query}"
            k = max(k, 7)

        if enhanced_retriever is not None:
            try:
                relevant_docs = enhanced_retriever.retrieve(
                    processed_query, 
                    language=language, 
                    k=k
                )
            except Exception as e:
                print(f"Enhanced retrieval failed: {str(e)}, falling back to standard retrieval")
                relevant_docs = retriever.get_relevant_documents(processed_query, k=k)
        else:
            relevant_docs = retriever.get_relevant_documents(processed_query, k=k)

        return relevant_docs
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return []
    
def format_retrieved_documents(docs: List[Document], language: str = "vietnamese") -> str:
    """
    Format retrieved documents for display.
    
    Args:
        docs (List[Document]): The retrieved documents
        language (str): The language for formatting
        
    Returns:
        str: Formatted document summary
    """
    if not docs:
        return "No relevant documents found."
    
    # Format based on language
    header = "Tài liệu tham khảo:" if language == "vietnamese" else "Reference Documents:"
    
    formatted_text = [header]
    
    for i, doc in enumerate(docs):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        disease_name = meta.get("disease_name", "Unknown disease")
        section = meta.get("section", "General information")
        
        # Format based on language
        if language == "vietnamese":
            formatted_text.append(f"{i+1}. Bệnh: {disease_name}, Phần: {section}")
        else:
            formatted_text.append(f"{i+1}. Disease: {disease_name}, Section: {section}")
    
    return "\n".join(formatted_text)

async def query_rag_system(qa_chain, query, enhanced_retriever=None, language="vietnamese", show_references=False):
    """
    Query the RAG system with enhanced handling for symptom-based queries.
    
    Args:
        qa_chain: The RAG system chain
        query (str): User query
        enhanced_retriever: The enhanced retriever (optional)
        language (str): The language to use for the response
        show_references (bool): Whether to show reference documents
        
    Returns:
        dict: Response from the RAG system with additional retrieval information
    """
    try:
        retriever = qa_chain.retriever
        
        relevant_docs = retrieve_relevant_documents(
            query=query,
            retriever=retriever,
            enhanced_retriever=enhanced_retriever,
            language=language,
            k=5 if not is_symptom_query(query, language) else 7
        )
        
        symptom_based = is_symptom_query(query, language)
        
        if symptom_based:
            # enhanced_query = f"Phân tích các triệu chứng sau và dự đoán bệnh có thể mắc phải: {query}"
            enhanced_query = query
            try:
                result = qa_chain.invoke({
                    "query": enhanced_query,
                    "input_language": language,
                })
            except AttributeError:
                result = qa_chain({"query": enhanced_query, "input_language": language})
        else:
            try:
                result = qa_chain.invoke({
                    "query": query, 
                    "input_language": language,
                })
            except AttributeError:
                result = qa_chain({"query": query, "input_language": language})
        
        # Add retrieval information to the result
        result["retrieved_documents"] = relevant_docs
        result["formatted_references"] = format_retrieved_documents(relevant_docs, language)
        
        return result
    except Exception as e:
        error_msg = f"Error querying RAG system: {str(e)}"
        print(error_msg)
        # Print the full error stack trace for debugging
        import traceback
        print(traceback.format_exc())
        return {
            "result": f"Error processing your query. ({language}): {str(e)}",
            "retrieved_documents": [],
            "formatted_references": "",
            "error": error_msg
        }