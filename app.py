import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# Override with Streamlit secrets if available (for cloud deployment)
if hasattr(st, "secrets"):
    for key in ["GOOGLE_API_KEY", "FRED_API_KEY", "HUD_API_TOKEN"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]

# ---- PAGE CONFIG ----

st.set_page_config(
    page_title="Raleigh-Durham Housing Assistant",
    page_icon="🏠",
    layout="centered"
)

# ---- DISCLAIMER ----

DISCLAIMER = """
This tool provides general housing information and financial guidelines for 
educational purposes only. It is **not** personalized financial advice. 
Always consult a licensed financial advisor before making major financial decisions.
"""

# ---- SESSION STATE INIT ----
# This must happen before anything else

if "profile_complete" not in st.session_state:
    st.session_state.profile_complete = False

if "profile" not in st.session_state:
    st.session_state.profile = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "context_cache" not in st.session_state:
    st.session_state.context_cache = {}

# ---- LOAD VECTORSTORE ----

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return Chroma(
        persist_directory=".chroma",
        embedding_function=embeddings
    )

# ---- BUILD CHAIN ----

def build_chain(user_profile_text):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

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

    def get_session_history(session_id: str):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = ChatMessageHistory()
        return st.session_state.chat_history

    def get_context(x):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = ChatMessageHistory()
        if "context_cache" not in st.session_state:
            st.session_state.context_cache = {}
        history = st.session_state.chat_history.messages
        if history:
            condensed = condense_chain.invoke({
                "question": x["question"],
                "chat_history": history
            })
            if "context" in st.session_state.context_cache:
                return st.session_state.context_cache["context"]
            result = retriever.invoke(condensed)
            st.session_state.context_cache["context"] = result
            return result
        else:
            result = retriever.invoke(x["question"])
            st.session_state.context_cache["context"] = result
            return result

    from langchain_core.runnables.history import RunnableWithMessageHistory

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

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_memory

# ---- PROFILE FORM ----

def show_profile_form():
    st.title("🏠 Raleigh-Durham Housing Assistant")
    st.warning(DISCLAIMER)
    st.markdown("### Before we get started, tell me a little about yourself.")
    st.markdown("This helps me give you more relevant information.")

    with st.form("profile_form"):
        budget = st.selectbox(
            "What is your monthly housing budget?",
            ["Under $1,000", "$1,000 - $1,500", "$1,500 - $2,000",
             "$2,000 - $2,500", "$2,500+"]
        )
        tenure = st.selectbox(
            "Are you looking to rent or buy?",
            ["Not sure yet", "Rent", "Buy"]
        )
        income = st.selectbox(
            "What is your approximate gross annual household income?",
            ["Prefer not to say", "Under $50,000", "$50,000 - $75,000",
             "$75,000 - $100,000", "$100,000 - $150,000", "$150,000+"]
        )
        priorities = st.multiselect(
            "What are your top priorities?",
            ["Walkability", "Good schools", "Nightlife and restaurants",
             "Quiet suburban feel", "Short commute", "Affordability"]
        )
        timeline = st.selectbox(
            "How long do you plan to stay in the area?",
            ["Not sure", "Less than 2 years", "2-5 years", "5+ years"]
        )

        submitted = st.form_submit_button("Let's get started →")

        if submitted:
            st.session_state.profile = {
                "budget": budget,
                "tenure": tenure,
                "income": income,
                "priorities": ", ".join(priorities) if priorities else "unspecified",
                "timeline": timeline
            }
            st.session_state.profile_complete = True
            st.rerun()

# ---- CHAT INTERFACE ----

def show_chat():
    st.title("🏠 Raleigh-Durham Housing Assistant")

    # Sidebar with profile summary
    with st.sidebar:
        st.markdown("### Your Profile")
        p = st.session_state.profile
        st.markdown(f"**Budget:** {p['budget']}".replace("$", r"\$"))
        st.markdown(f"**Looking to:** {p['tenure']}")
        st.markdown(f"**Income:** {p['income']}".replace("$", r"\$"))
        st.markdown(f"**Priorities:** {p['priorities']}")
        st.markdown(f"**Timeline:** {p['timeline']}")
        st.divider()
        st.caption(DISCLAIMER)
        if st.button("Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Format user profile for prompt
    p = st.session_state.profile
    user_profile_text = f"""
User Profile:
- Monthly housing budget: {p['budget']}
- Looking to: {p['tenure']}
- Annual household income: {p['income']}
- Priorities: {p['priorities']}
- Planning to stay: {p['timeline']}
"""

    # Build chain once
    if st.session_state.chain is None:
        with st.spinner("Loading housing data..."):
            st.session_state.chain = build_chain(user_profile_text)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask me anything about housing in Raleigh-Durham..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(
                    {"question": question, "user_profile": user_profile_text},
                    config={"configurable": {"session_id": "streamlit_session"}}
                )
            st.markdown(answer.replace("$", r"\$"))
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- MAIN ----

if st.session_state.profile_complete:
    show_chat()
else:
    show_profile_form()