from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = "vectorstore"


def main():

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
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
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\nSwiggy Annual Report RAG System")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        response = qa_chain(query)

        print("\nAnswer:\n")
        print(response["result"])

        print("\nSupporting Context:\n")

        for i, doc in enumerate(response["source_documents"]):
            print(f"Source {i+1}:")
            print(doc.page_content[:400])
            print("\n")

        print("-" * 50)


if __name__ == "__main__":
    main()
