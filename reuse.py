from pathlib import Path
import getopt, sys, os, shutil

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

from langchain.chains import RetrievalQA

PDF_ROOT_DIR    = ""
PDF_FOLDER_PATH = ""
VDB_PATH        = "./default_chroma_vdb"
COLLECTION_NAME = "default_collection_name"
MODEL_NAME      = ""
EMBEDDING_NAME  = ""
REUSE_VDB       = False

argumentlist = sys.argv[1:]
options = "hm:e:p:v:c:r:"
long_options = ["help",
                 "model_name",
                 "embedding_name"
                 "pdf_path",
                 "vdb_path",
                 "collection_name",
                 "reuse"]

try:
    arguments, values = getopt.getopt(argumentlist, options, long_options)
    for currentArgument, currentValue in arguments:
        print(currentArgument, " ", currentValue)
        if currentArgument in ("-h", "--Help"):
            print ("Displaying Help:")
            print ("-h or --help to get this help msg")
            print ("-m or --model_name name of the model used, ex:phi3")
            print ("-e or --embedding_name name of the embedding model used, ex:nomic-embed-text")
            print ("-p or --pdf_path path to datas to be used for RAG")
            print ("-v or --vdb_path path to the directory used as a vector database")
            print ("-c or --collection_name name of the vector database collection")
            print ("-r or --reuse vdb_path")
            exit()
        elif currentArgument in ("-m", "--model_name"):
            MODEL_NAME = currentValue
        elif currentArgument in ("-e", "--embedding_name"):
            EMBEDDING_NAME = currentValue
        elif currentArgument in ("-p", "--pdf_path"):
            PDF_FOLDER_PATH = Path(currentValue)
        elif currentArgument in ("-v", "--vdb_path"):
            VDB_PATH = Path(currentValue)
        elif currentArgument in ("-c", "--collection_name"):
            COLLECTION_NAME = currentValue
        elif currentArgument in ("-r", "--reuse"):
            if currentValue.casefold() == "true":
                REUSE_VDB = True
            else:
                REUSE_VDB = False
    #exit()
except getopt.error as err:
    print (str(err))

print(MODEL_NAME)
print(EMBEDDING_NAME)
print(PDF_FOLDER_PATH)
print(VDB_PATH)
print(COLLECTION_NAME)
print(REUSE_VDB)
#exit()
def create_vdb(datas, embedding, vdb_path, collection_name, reuse_vdb):
    """Create a vector database from the documents"""
    Isvectordb = False
    if vdb_path.exists():
        if any(vdb_path.iterdir()):
            if reuse_vdb == True:
                vectordb = Chroma(persist_directory=str(vdb_path),
                                 embedding_function=embedding,
                                 collection_name=collection_name)
                print(vectordb._collection.count())
                Isvectordb = True
                print(f"vector database REUSED in {vdb_path}")
            else:
                shutil.rmtree(str(vdb_path))
                print(f"vector database REMOVED in {vdb_path}")
    else:
        vdb_path.mkdir(parents=True)
    
    if Isvectordb == False:
        vectordb = Chroma.from_documents(
            documents=datas,
            embedding=embedding,
            persist_directory=str(vdb_path),  # Does not accept Path
            collection_name=collection_name
        )
        print(f"vector database CREATED in {vdb_path}")

    return vectordb

# Embedding Using Ollama
ollama_embeddings = OllamaEmbeddings(
    model=EMBEDDING_NAME,
    show_progress=True
)
splitted_data=None
# Add to vector database
vectordb = create_vdb(splitted_data,
                     ollama_embeddings,
                     Path(VDB_PATH),
                     COLLECTION_NAME,
                     REUSE_VDB)

print("vectordb._collection.count() ", vectordb._collection.count())
# LLM from Ollama
local_model = MODEL_NAME
llm = ChatOllama(model=local_model, temperature=0)

template="""You are an AI language model assistant. Your task is to generate two
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Answer the question based ONLY on the following context:
    {context}
    Original question:
    {question}"""
prompt = ChatPromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectordb.as_retriever(), chain_type_kwargs={"prompt": prompt}
)

question = ""
while True:
    print("*"*20)
    print("Enter a QUESTION: (to exit enter q or quit)")
    question = input("")
    if question.casefold() == "q" or question.casefold() == "quit":
        print("End QA")
        break
    else:
        result = qa_chain({"context" : vectordb.as_retriever(),"query": question})
        print(result["result"])
