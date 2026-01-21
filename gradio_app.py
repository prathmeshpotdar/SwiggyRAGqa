import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

VECTOR_DB_PATH = "vectorstore"


print("Loading vector database...")

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

prompt_template = """
You are a financial document assistant.

Answer ONLY using the provided Swiggy Annual Report context.

If numeric values such as revenue or income are mentioned, extract them exactly as written.

If the answer is not found in the context, say:
"I could not find this information in the Swiggy Annual Report."

Context:
{context}

Question:
{question}

Answer:
"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)


def ask_question(query):

    if not query.strip():
        return "Please enter a valid question.", ""

    response = qa_chain(query)

    answer = response["result"]

    sources_text = ""

    for i, doc in enumerate(response["source_documents"]):
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content[:400]

        sources_text += f"\nSource {i+1} (Page {page}):\n{content}\n"

    return answer, sources_text


interface = gr.Interface(

    fn=ask_question,

    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question about Swiggy Annual Report..."
    ),

    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Supporting Context")
    ],

    title="Swiggy Annual Report — RAG QA System",

    description="Ask questions and get answers grounded strictly in Swiggy's Annual Report using Retrieval-Augmented Generation.",

    theme="soft"
)

if __name__ == "__main__":
    interface.launch()
