import streamlit as st
from rag_core import load_pdfs, retrieve, generate_answer

st.set_page_config(page_title="Исламский RAG", page_icon="🕌", layout="wide")
st.markdown("""
<style>
    .stApp {background-color: #0F2C2C; color: #E8D5A3;}
    .chat-message {background-color: #1A3C3C; border-radius: 15px; padding: 15px; margin: 10px 0;}
    h1 {color: #E8D5A3; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("🕌 Исламский RAG Чатбот")
st.caption("Вопросы по Корану и Сахих аль-Бухари (на русском в PDF) • Ответы на английском")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Задай вопрос по исламу..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        docs = retrieve(prompt)
        answer = generate_answer(prompt, docs)
        st.markdown(answer)
        st.caption("Цитаты из документов указаны в ответе")
    st.session_state.messages.append({"role": "assistant", "content": answer})