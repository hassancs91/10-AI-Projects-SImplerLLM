from SimplerLLM.language.llm import LLM,LLMProvider
from SimplerLLM.language.embeddings import LLM as EmbeddingLLM,EmbeddingsProvider
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.tools.text_chunker import chunk_by_max_chunk_size,chunk_by_semantics
from SimplerVectors_core import VectorDatabase

db = VectorDatabase('VDB')

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
llm_embedding_instance = EmbeddingLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-large")

youtube_url = "https://www.youtube.com/watch?v=CiyFuDcwaAo"

youtube_content = load_content(youtube_url).content

#chunk
#chunks = chunk_by_max_chunk_size(text=blog_content,max_chunk_size=1000,preserve_sentence_structure=True).chunk_list
chunks = chunk_by_semantics(text=youtube_content,llm_embeddings_instance=llm_embedding_instance,threshold_percentage=95).chunk_list

embedded_documents = []

for chunk in chunks:
    chunk_embedding = llm_embedding_instance.generate_embeddings(user_input=chunk.text,model_name="text-embedding-3-large")
    embedded_documents.append(chunk_embedding[0].embedding)


for idx, emb in enumerate(embedded_documents):
     db.add_vector(emb, {"doc_id": idx, "vector": chunks[idx].text}, normalize=True)

db.save_to_disk("db_yt")



print("Hello! Ask me any question or type 'exit' to leave:")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Embed the user query
    query_embedding = llm_embedding_instance.generate_embeddings(user_input=user_input,model_name="text-embedding-3-large")
    query_embedding = db.normalize_vector(query_embedding[0].embedding)  # Normalizing the query vector

    # Retrieving the top similar questions and their answers
    results = db.top_cosine_similarity(query_embedding, top_n=1)
    if results:
        top_match = results[0]
        context = top_match[0]["vector"]
        prompt = f"Answer the following question: {user_input} \n Based on this context only: \n" + top_match[0]["vector"]
        answer = llm_instance.generate_response(prompt=prompt)
        print(answer)

    else:
        print("Bot: I'm not sure how to answer that.")


