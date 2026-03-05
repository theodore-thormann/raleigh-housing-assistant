import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---- DISCLAIMER ----

DISCLAIMER = """
==================================================
DISCLAIMER: This tool provides general housing 
information and financial guidelines for 
educational purposes only. It is NOT personalized 
financial advice. Always consult a licensed 
financial advisor before making major financial 
decisions.
==================================================
"""

# ---- USER PROFILE COLLECTION ----

def collect_user_profile():
    """Ask the user a few questions to personalize responses"""
    print("\nBefore we get started, let me learn a little about you.")
    print("This helps me give you more relevant information.\n")

    profile = {}

    # Budget
    print("What is your monthly housing budget?")
    print("  1. Under $1,000")
    print("  2. $1,000 - $1,500")
    print("  3. $1,500 - $2,000")
    print("  4. $2,000 - $2,500")
    print("  5. $2,500+")
    choice = input("Enter 1-5: ").strip()
    budget_map = {
        "1": "under $1,000/month",
        "2": "$1,000-$1,500/month",
        "3": "$1,500-$2,000/month",
        "4": "$2,000-$2,500/month",
        "5": "$2,500+/month"
    }
    profile["budget"] = budget_map.get(choice, "unspecified")

    # Rent or buy
    print("\nAre you looking to rent or buy?")
    print("  1. Rent")
    print("  2. Buy")
    print("  3. Not sure yet")
    choice = input("Enter 1-3: ").strip()
    tenure_map = {"1": "rent", "2": "buy", "3": "not sure yet"}
    profile["tenure"] = tenure_map.get(choice, "not sure yet")

    # Annual income (only if buying)
    if profile["tenure"] == "buy":
        print("\nWhat is your approximate gross annual household income?")
        print("  1. Under $50,000")
        print("  2. $50,000 - $75,000")
        print("  3. $75,000 - $100,000")
        print("  4. $100,000 - $150,000")
        print("  5. $150,000+")
        choice = input("Enter 1-5: ").strip()
        income_map = {
            "1": "under $50,000",
            "2": "$50,000-$75,000",
            "3": "$75,000-$100,000",
            "4": "$100,000-$150,000",
            "5": "$150,000+"
        }
        profile["income"] = income_map.get(choice, "unspecified")
    else:
        profile["income"] = "unspecified"

    # Priorities
    print("\nWhat are your top priorities? (enter numbers separated by commas)")
    print("  1. Walkability / being close to things")
    print("  2. Good schools")
    print("  3. Nightlife and restaurants")
    print("  4. Quiet and suburban feel")
    print("  5. Short commute to downtown Raleigh or Durham")
    print("  6. Affordability")
    choices = input("Enter 1-6 (e.g. 1,3,6): ").strip().split(",")
    priority_map = {
        "1": "walkability",
        "2": "good schools",
        "3": "nightlife and restaurants",
        "4": "quiet suburban feel",
        "5": "short commute",
        "6": "affordability"
    }
    profile["priorities"] = [priority_map[c.strip()] for c in choices if c.strip() in priority_map]

    # How long they plan to stay
    print("\nHow long do you plan to stay in the area?")
    print("  1. Less than 2 years")
    print("  2. 2-5 years")
    print("  3. 5+ years")
    print("  4. Not sure")
    choice = input("Enter 1-4: ").strip()
    timeline_map = {
        "1": "less than 2 years",
        "2": "2-5 years",
        "3": "5+ years",
        "4": "not sure"
    }
    profile["timeline"] = timeline_map.get(choice, "not sure")

    return profile


def format_profile(profile):
    """Format user profile as a string to inject into prompts"""
    priorities = ", ".join(profile.get("priorities", [])) or "unspecified"
    return f"""
User Profile:
- Monthly housing budget: {profile.get('budget', 'unspecified')}
- Looking to: {profile.get('tenure', 'unspecified')}
- Annual household income: {profile.get('income', 'unspecified')}
- Priorities: {priorities}
- Planning to stay: {profile.get('timeline', 'unspecified')}
"""


# ---- RAG SETUP ----

# In-memory store for conversation history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def build_chain(user_profile_text):
    """Build the RAG chain with user profile and conversation memory"""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory=".chroma",
        embedding_function=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and knowledgeable Raleigh-Durham NC housing assistant.
Answer the question based only on the following context.
Respond in a warm, conversational tone as if you are a local who knows the area well.
Do not use bullet points, headers, or bold text. Write in natural flowing paragraphs.
When discussing finances, reference general guidelines only and remind the user
to consult a financial advisor for personalized advice.

User Profile:
{user_profile}

Context from knowledge base:
{context}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Condense follow-up questions into standalone questions
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the conversation history and a follow-up question, 
    rewrite the follow-up question to be a complete standalone question that 
    includes all necessary context from the conversation history.
    If the question is already standalone, return it unchanged.
    Only return the rewritten question, nothing else."""),
        ("placeholder", "{chat_history}"),
        ("human", "Follow-up question: {question}"),
    ])

    condense_chain = condense_prompt | llm | StrOutputParser()

# Cache for storing retrieved context
    context_cache = {}

    def get_context(x):
            history = get_session_history("user_session_1").messages
            if history:
                condensed = condense_chain.invoke({
                    "question": x["question"],
                    "chat_history": history
                })
                if "context" in context_cache:
                    return context_cache["context"]
                result = retriever.invoke(condensed)
                context_cache["context"] = result
                return result
            else:
                result = retriever.invoke(x["question"])
                context_cache["context"] = result
                return result

    chain = (
        {
            "context": get_context,
            "question": lambda x: x["question"],
            "user_profile": lambda x: x["user_profile"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_memory


# ---- MAIN ----

if __name__ == "__main__":
    print(DISCLAIMER)

    # Collect user profile
    profile = collect_user_profile()
    user_profile_text = format_profile(profile)

    print("\nGreat! I have everything I need. Ask me anything about housing in Raleigh-Durham.\n")

    # Build RAG chain with memory
    chain = build_chain(user_profile_text)
    session_id = "user_session_1"

    # Chat loop
    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit", "bye"]:
            print("Good luck with your housing search!")
            break
        if not question:
            continue

        answer = chain.invoke(
            {"question": question, "user_profile": user_profile_text},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\nAssistant: {answer}\n")