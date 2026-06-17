import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. Page Configuration & UI Layout
# ==========================================
st.set_page_config(page_title="AI Skeptic Assistant", page_icon="🔬", layout="centered")

st.title("🔬 The Scientific Skeptic")
st.write("Ask a factual question. The response will answer it, but also attempt to give scientific caveats.")

# Fetch the API key safely from the environment or secrets management
# (We'll look at how Streamlit stores this securely below)
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    pass
else:
    st.warning("Please configure your GEMINI_API_KEY to use this application.")

# ==========================================
# 2. Your LangChain Pipeline
# ==========================================
@st.cache_resource # This caches the model engine instantiation so it'll run fast
def load_langchain_pipeline():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    parser = StrOutputParser()

    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question accurately. Then, format your entire response into a concise, direct summary of the question and the answer."),
        ("user", "{question}")
    ])
    chain1 = prompt1 | model | parser

    prompt2 = ChatPromptTemplate.from_messages([
        ("system", (
            "First, analyze the provided text and determine the specific scientific field it relates to. "
            "Then, adopt the persona of a scientist in that field. Provide strong, scientifically grounded reasons "
            "to be skeptical of the provided statement."
        )),
        ("user", "Statement: {AI_answer01}\n\nGive me scientific reasons to be skeptical of this.")
    ])
    chain2 = prompt2 | model | parser

    formatting_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an editor rewriting technical data for a general audience. "
            "You must output the final response using EXACTLY this layout template. "
            "Do not include any greeting or extra text outside this format:\n\n"
            "In answer to your question, - this is the likely answer.\n"
            "[Insert the data from AI_answer01 here]\n\n"
            "However, you should bear in mind the following reasons why it is not clear cut.\n"
            "[Summarize the data from Scientist_reasons here. Word it smoothly for a non-scientist, "
            "ensuring the tone flows naturally from the first part of the response.]"
        )),
        ("user", "AI_answer01:\n{AI_answer01}\n\nScientist_reasons:\n{Scientist_reasons}")
    ])

    full_chain = (
        {"AI_answer01": chain1}
        | RunnablePassthrough.assign(Scientist_reasons=chain2)
        | formatting_prompt
        | model
        | parser
    )
    return full_chain

# ==========================================
# 3. Interactive Web Elements
# ==========================================
# Input box for your friends to type their question
user_question = st.text_input("Enter your question:", placeholder="e.g., Do birds use spiderwebs to hold their nests together?")

# Trigger execution on button click
if st.button("Analyze the Dixon Way", type="primary"):
    if user_question.strip() == "":
        st.error("Please enter a valid question first.")
    else:
        # Show a loading spinner while the chains run
        with st.spinner("Consulting the experts..."):
            try:
                pipeline = load_langchain_pipeline()
                final_response = pipeline.invoke({"question": user_question})
                
                # Render the clean, structured text output
                st.markdown("---")
                st.write(final_response)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

                
