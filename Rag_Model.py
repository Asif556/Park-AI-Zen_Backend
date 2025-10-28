# Flask-ready RAG chatbot functions

def initialize_rag_system():
    """
    Initialize the RAG system with MongoDB data and Qdrant Cloud.
    Call this once when your Flask app starts.
    Returns: qdrant_client, collection_name, model, mongo_uri
    """
    from pymongo import MongoClient
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import numpy as np
    import os
    from dotenv import load_dotenv
    import uuid
    
    # Load environment variables
    load_dotenv()
    
    MONGO_URI = os.getenv("MONGODB_URI")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    if not MONGO_URI:
        raise ValueError("MONGODB_URI not found in environment variables")
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL not found in environment variables")
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY not found in environment variables")
    
    # Initialize Qdrant Cloud client
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    print(f"✅ Connected to Qdrant Cloud")
    
    client = MongoClient(MONGO_URI)
    db = client["smart_parking_db"]
    collection = db["parking_records"]
    records = list(collection.find({}, {"_id": 0}))
    



    docs = []
    for rec in records:
        text = (
            f"Record: Token ID {rec.get('tokenId')}, "
            f"Vehicle Number {rec.get('vehicleNumber')}, "
            f"Slot Number {rec.get('slotNumber')}, "
            f"Entry Time {rec.get('entryTime')}, "
            f"Exit Time {rec.get('exitTime')}, "
            f"Charge ₹{rec.get('charge')}, "
            f"Status {rec.get('status')}."
        )
        docs.append(text)
    
    
    def text_splitter(text, chunk_size=300, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
    
    all_chunks = []
    for doc in docs:
        chunks = text_splitter(doc)
        all_chunks.extend(chunks)
    
    # Generate embeddings using SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print(f"✅ Generated embeddings for {len(all_chunks)} chunks")
    
    # Create or recreate Qdrant collection
    collection_name = "parking_records"
    embedding_size = len(embeddings[0])
    
    # Delete collection if it exists (to ensure fresh data)
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"🗑️ Deleted existing collection: {collection_name}")
    except Exception:
        pass  # Collection doesn't exist yet
    
    # Create collection with vector configuration
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    print(f"✅ Created Qdrant collection: {collection_name}")
    
    # Upload vectors to Qdrant
    points = []
    for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),  # Generate unique ID
            vector=embedding.tolist(),
            payload={"text": chunk, "chunk_index": idx}
        )
        points.append(point)
    
    # Upload in batches for better performance
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    print(f"✅ Uploaded {len(points)} vectors to Qdrant Cloud")
    
    return qdrant_client, collection_name, model, MONGO_URI


def get_chatbot_response(query, qdrant_client, collection_name, model, mongo_uri, top_k=10):
    """
    Get chatbot response using Qdrant for vector search and Gemini for generation.
    
    Args:
        query: User's question
        qdrant_client: Qdrant client instance
        collection_name: Name of Qdrant collection
        model: SentenceTransformer model
        mongo_uri: MongoDB connection URI
        top_k: Number of similar chunks to retrieve
    """
    import google.generativeai as genai
    import numpy as np
    from pymongo import MongoClient
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    client = MongoClient(mongo_uri)
    db = client["smart_parking_db"]
    collection = db["parking_records"]
    
    
    total_records = collection.count_documents({})
    active_records = list(collection.find({"status": "active"}))
    completed_records = collection.count_documents({"status": "completed"})
    total_revenue = sum([r.get('charge', 0) for r in collection.find({"status": "completed"})])
    
    
    db_summary = f"""
DATABASE SUMMARY (Real-time):
- Total parking records: {total_records}
- Currently active vehicles: {len(active_records)}
- Completed parkings: {completed_records}
- Total revenue: ₹{total_revenue}

ACTIVE VEHICLES RIGHT NOW:
"""
    
    if active_records:
        for rec in active_records:
            db_summary += f"  • Vehicle {rec.get('vehicleNumber')} in Slot {rec.get('slotNumber')}, Entry: {rec.get('entryTime')}\n"
    else:
        db_summary += "  • No vehicles currently parked\n"
    
    # Generate query embedding
    query_vector = model.encode([query])[0]
    
    # Search Qdrant for similar chunks
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k
    )
    
    # Extract retrieved context from search results
    retrieved_chunks = [hit.payload["text"] for hit in search_results]
    retrieved_context = "\n".join(retrieved_chunks)
    
    # Create comprehensive prompt
    prompt = f"""You are a Smart Parking Management AI Assistant.

{db_summary}

HISTORICAL RECORDS (Retrieved context):
{retrieved_context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Use the real-time database summary above for current status questions (active vehicles, current occupancy)
- Use the historical records for past data, trends, and specific vehicle histories
- Perform calculations when needed (sums, averages, counts)
- Be precise with numbers and vehicle details
- If information is not available, clearly state that

Provide a helpful, accurate answer:"""
    
    
    model_g = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model_g.generate_content(prompt)
    
    return response.text


