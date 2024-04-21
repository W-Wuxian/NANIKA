from pathlib import Path
import os

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain.indexes import VectorstoreIndexCreator

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

PDF_ROOT_DIR    = Path(os.environ['DOC'])
PDF_FOLDER_PATH = f'{PDF_ROOT_DIR}/HDR'
VDB_PATH        = Path("./chroma_vdb")

def create_vdb(datas, embedding, vdb_path):
    """Create a vector database from the documents"""
    if vdb_path.exists():
        if any(vdb_path.iterdir()):
            raise FileExistsError(
                f"Vector database directory {vdb_path} is not empty"
            )
    else:
        vdb_path.mkdir(parents=True)

    vectordb = Chroma.from_documents(
        documents=datas,
        embedding=embedding,
        persist_directory=str(vdb_path),  # Does not accept Path
        collection_name="local-HDR-rag"
    )
    vectordb.persist()  # Save database to use it later

    print(f"vector database created in {vdb_path}")
    return vectordb

#A = os.listdir(PDF_FOLDER_PATH)
#print(A)

# Load PDF files
loader = DirectoryLoader(
    PDF_FOLDER_PATH, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader
)
print(loader)
data = loader.load()
print(len(data))
# Split and chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=30,
    separators=["\n\n", "\n", r"(?<=\. )",  " ", "",]
)
splitted_data = splitter.split_documents(data)
# Embedding Using Ollama
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    show_progress=True
)
# Add to vector database
vectordb = create_vdb(splitted_data, ollama_embeddings, VDB_PATH)

# LLM from Ollama
local_model = "phi"
llm = ChatOllama(model=local_model, temperature=0)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate two
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
retriever = MultiQueryRetriever.from_llm(
    vectordb.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)
# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
print("*"*20)
prompt = ChatPromptTemplate.from_template(template)
print("*"*20)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("*"*20)
print("enter question")
response = chain.invoke(input(""))
print(response)
print("#"*20)
response = chain.invoke("How many types of nuclear reactions are there?")
print(response)

#query = "How many types of nuclear reactions are there?"
#docs = vectordb.similarity_search(query, k=6)
#for doc in docs:
#    print(doc.page_content)
#    print("#"*20)

vectordb.delete_collection()

#loader = [UnstructuredPDFLoader( os.path.join(PDF_FOLDER_PATH, fn)) for fn in A]
#print(len(loader))
#data = loader[0].load()
#print(data)
#print(data[0].page_content)

# Split and chunk 
#data = loader.load()
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#chunks = text_splitter.split_documents(data)

#index = VectorstoreIndexCreator().from_loaders(loader)
#print(index)

#index.query()
#index.query_with_sources()
