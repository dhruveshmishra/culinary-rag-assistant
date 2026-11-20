import streamlit as st
from assistant import ask
import config

st.set_page_config(
    page_title="Le Grand Bistro Culinary Assistant",
    page_icon="🍳",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stAlert {
        border-radius: 8px;
    }
    .refusal-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 4px;
        color: #856404;
        font-weight: 500;
        margin: 10px 0;
    }
    .source-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍳 Le Grand Bistro Culinary Operations Assistant")
st.markdown(
    "Strict RAG Assistant for **recipes, ingredient quantities, allergens, food safety standards, cooking temperatures, and kitchen procedures** "
    "verified exclusively against the *Official Culinary Operations, Standardized Recipes & Allergen Manual*."
)

# Sidebar with Sample Questions
st.sidebar.header("📋 Sample Questions")
st.sidebar.markdown("Click any sample question to evaluate:")

sample_questions = [
    ("Q1: Truffle Ganache Quantities", "What exact quantities, in grams, of dark couverture chocolate and heavy cream are required for a standard 20-portion batch of Signature Truffle Ganache?"),
    ("Q2: Autumn Menu Allergens", "According to the Allergen Matrix, which dishes on the Autumn Tasting Menu contain tree nuts or sesame?"),
    ("Q3: Eiffel Tower Hours (Out-of-Scope)", "At what time does the Eiffel Tower in Paris close to visitors during the winter season?"),
    ("Q4: Sous-Vide Duck Pasteurization", "What is the mandatory minimum internal cooking temperature, in degrees Celsius, and the required holding time for sous-vide duck breast, as specified in the Food Safety section?"),
    ("Q5: Noma Michelin Rating (Out-of-Scope)", "What is the current Michelin-star rating of Noma in Copenhagen?")
]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for label, query_text in sample_questions:
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.pending_query = query_text

st.sidebar.divider()
st.sidebar.markdown("### 🛡️ Refusal Guard Status")
st.sidebar.markdown(f"- **Cosine Similarity Threshold:** `{config.SIM_THRESHOLD}`")
st.sidebar.markdown(f"- **BM25 Threshold:** `{config.BM25_THRESHOLD}`")
st.sidebar.markdown("- **Refusal Policy:** Strict offline grounding. Rejects out-of-scope or unverified queries.")

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.session_state.pending_query = None
    st.rerun()

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg.get("refused"):
                st.markdown(f'<div class="refusal-box">⛔ {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
                if msg.get("sources"):
                    src_str = ", ".join([f"Page {p}" for p in msg["sources"]])
                    st.markdown(f'<span class="source-tag">Verified Source: {src_str}</span>', unsafe_allow_html=True)

            if msg.get("retrieved"):
                with st.expander("🔍 Show retrieved context & scores"):
                    for c in msg["retrieved"]:
                        st.markdown(f"**Page {c.get('page')} | {c.get('section_title')}**")
                        st.caption(f"RRF Score: `{c.get('score')}` | Dense Similarity: `{c.get('sim_score')}` | BM25 Score: `{c.get('bm25_score')}`")
                        st.text(c.get("text"))
                        st.divider()

# Input handling
user_input = st.chat_input("Ask a question about recipes, allergens, or food safety...")
active_query = None

if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
elif user_input:
    active_query = user_input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual and verifying compliance..."):
            result = ask(active_query)
            ans = result["answer"]
            refused = result["refused"]
            sources = result["sources"]
            retrieved = result["retrieved"]

            if refused:
                st.markdown(f'<div class="refusal-box">⛔ {ans}</div>', unsafe_allow_html=True)
            else:
                st.markdown(ans)
                if sources:
                    src_str = ", ".join([f"Page {p}" for p in sources])
                    st.markdown(f'<span class="source-tag">Verified Source: {src_str}</span>', unsafe_allow_html=True)

            if retrieved:
                with st.expander("🔍 Show retrieved context & scores"):
                    for c in retrieved:
                        st.markdown(f"**Page {c.get('page')} | {c.get('section_title')}**")
                        st.caption(f"RRF Score: `{c.get('score')}` | Dense Similarity: `{c.get('sim_score')}` | BM25 Score: `{c.get('bm25_score')}`")
                        st.text(c.get("text"))
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": ans,
                "refused": refused,
                "sources": sources,
                "retrieved": retrieved
            })
