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
    system_prompt = """You are an expert log analyst for a Java microservice system (AuthService, OrderService, InventoryService, PaymentGateway, UserService).

    Logs follow two formats:
    - Format 1: `TIMESTAMP LEVEL [Service] [user=X request_id=Y action=Z thread=T] - message` (+ optional Java stack trace)
    - Format 2: `TIMESTAMP LEVEL [Service] [user=X req_id=Y method=M uri=U status=S latency=Xms thread=T] - message`

    For every ERROR: quote the log line, identify the exception, give 2 likely root causes and a fix.
    For every WARN: assess severity and recommend action.
    For performance issues: flag latency >800ms and memory >80%.
    If info is missing, say so and suggest what data would help.
    Always end with prioritized next steps: Critical → High → Medium → Low."""

    # 2. Build the document prompt (how each retrieved chunk is formatted)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Source: {source}\nContent: {page_content}"
    )

    # 3. Build the main QA prompt with system message
    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            """Use the log chunks below to answer the question. For each ERROR include the exception, root cause, and fix. For each WARN assess risk. Correlate by request_id where possible. End with prioritized next steps.

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
