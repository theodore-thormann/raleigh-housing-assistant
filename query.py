import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load the vector database we already built
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = Chroma(
    persist_directory=".chroma",
    embedding_function=embeddings
)

# Set up the LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3-flash-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a friendly and knowledgeable Raleigh NC housing assistant. 
Answer the question based only on the following context.
Respond in a warm, conversational tone as if you are a local who knows the area well.
Do not use bullet points, headers, or bold text. Write in natural flowing paragraphs.

{context}

Question: {question}
""")

# Build the RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions in a loop
print("Raleigh Housing Assistant ready!\n")
while True:
    question = input("Ask a question (or type 'quit'): ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")