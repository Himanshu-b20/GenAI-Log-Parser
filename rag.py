from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate


load_dotenv()
llm = None
vector_store=None
def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.9, max_tokens=1000)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        vector_store = Chroma(
            collection_name='logs_data',
            persist_directory='./vectorstore',
            embedding_function=ef
        )

def generate_answer(query):
    if not vector_store:
        raise RuntimeError('Vector Database is Not Initialize..')

    # 1. Define your system prompt
    system_prompt = """You are an expert log analyst AI assistant.
Your job is to analyze application logs and provide clear, concise answers.
When identifying errors, always suggest possible root causes and solutions (This is important).
If you don't find relevant information in the logs, say so explicitly and provide your thoughts and inputs around that.
"""

    # 2. Build the document prompt (how each retrieved chunk is formatted)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Source: {source}\nContent: {page_content}"
    )

    # 3. Build the main QA prompt with system message
    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            """Use the following context to answer the question and When identifying errors, always suggest possible root causes and solutions. From the context try to find the Class name and method name from where the exception occurred and point out in you response  (This is important).

{summaries}

Question: {question}
Answer:"""
        )
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # 4. Pass the custom prompt into the chain
    combine_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,               # <-- system prompt injected here
        document_prompt=document_prompt # <-- controls how each doc chunk is shown
    )

    chain = RetrievalQAWithSourcesChain(
        retriever=vector_store.as_retriever(),
        combine_documents_chain=combine_chain
    )

    result = chain.invoke({'question': query}, return_only_output=True)
    sources = result.get('sources', '')

    return result['answer'], sources

def process_data(log_data):
    yield "Initializing Component..."
    initialize_components()

    vector_store.reset_collection()

    loader = TextLoader(log_data)
    data = loader.load()

    yield "Splitting Data..."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        separators=["\n\n", "\n", ".", " "])


    docs =text_splitter.split_documents(data)

    id_list = [str(i+1) for i in range(len(docs))]

    yield "Adding Doc in Vector DB.."
    vector_store.add_documents(docs, ids=id_list)

    yield "Done Adding Doc to Vector DB.."

    # result = vector_store.similarity_search("Do you see any ConstraintViolationException for user=user_18 and for uri=/api/v1/logout and PaymentGateway?", k=2)
    # print("===============> ",result)

    # answer, source = generate_answer('Do you see any ConstraintViolationException for user=user_18 and for uri=/api/v1/logout and PaymentGateway?? if yes tell me the possible solution as well')
    # print("=================>",answer, source)
    # return answer
