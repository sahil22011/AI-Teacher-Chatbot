import streamlit as st
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
template = PromptTemplate(
    input_variables=['history', 'input'],
    template="""
You are a helpful AI Teacher.
You understand and respond in the **same language** as the user.
Your answers must be educational — include definitions, explanations, and examples.
Make your responses clear, structured, and easy to understand.

Conversation so far:
{history}

User: {input}
AI:
"""
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
chain=ConversationChain(
    llm=model,
    memory=st.session_state.memory,
    prompt=template,
    verbose=False
)

st.title("AI Teacher Chatbot")
st.write("Supports **almost all languages**.")

if st.session_state.memory.chat_memory.messages:
    for i, msg in enumerate(st.session_state.memory.chat_memory.messages):
        if i % 2 == 0:
            st.markdown(f"**User:** {msg.content}")
        else:
            st.markdown(f"**AI:** {msg.content}")

user_question = st.text_input("Type your question:",key="user_question")
if st.button("Send"):
        
    if user_question:
        response = chain.invoke({"input": user_question})
        st.session_state.user_input = ""
        st.rerun()
