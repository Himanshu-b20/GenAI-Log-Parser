from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

load_dotenv()
llm = None
vector_store=None
def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

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
    combine_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    chain = RetrievalQAWithSourcesChain(
        retriever=vector_store.as_retriever(),
        combine_documents_chain=combine_chain
    )
    # chain = RetrievalQAWithSourcesChain(llm=llm, retriever=vector_store.as_retriever())
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