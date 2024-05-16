from pathlib import Path
import getopt, sys, os, shutil

from langchain_community.document_loaders import (
    DirectoryLoader, UnstructuredPDFLoader, TextLoader,
    PythonLoader, UnstructuredImageLoader,
    UnstructuredExcelLoader, UnstructuredWordDocumentLoader, UnstructuredXMLLoader,
    UnstructuredCSVLoader, UnstructuredPowerPointLoader, UnstructuredODTLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain.indexes import VectorstoreIndexCreator

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


def routerloader(obj):
    loader = []
    accumulator = []
    if os.path.isfile(obj):
        Fname = os.path.basename(obj)
        if Fname.endswith(".pdf"):
            loader = UnstructuredPDFLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".txt") or Fname.endswith(".py"):
            loader = TextLoader(obj, autodetect_encoding = True)
        if Fname.endswith(".py"):
            loader = PythonLoader(obj)
        if Fname.endswith(".png") or Fname.endswith(".jpg"):
            loader = UnstructuredImageLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".xlsx") or Fname.endswith(".xls"):
            loader = UnstructuredExcelLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".odt"):
            loader = UnstructuredODTLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".csv"):
            loader = UnstructuredCSVLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".md"):
            loader = UnstructuredMarkdownLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        if Fname.endswith(".org"):
            loader = UnstructuredOrgModeLoader(str(obj), mode="single", strategy="hi_res",
            show_progress=True, use_multithreading=True)
        accumulator.extend(loader.load())
    elif os.path.isdir(obj):
        if any(File.endswith(".pdf") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".txt") for File in os.listdir(obj)):
            abc={'autodetect_encoding': True}
            loader = DirectoryLoader(
                obj, glob="**/*.txt", loader_cls=TextLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".py") for File in os.listdir(obj)):
            loader = DirectoryLoader(
                obj, glob="**/*.py", loader_cls=PythonLoader,
                show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".png") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.png", loader_cls=UnstructuredImageLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".jpg") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.jpg", loader_cls=UnstructuredImageLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".xls") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.xls", loader_cls=UnstructuredExcelLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".xlsx") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".odt") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.odt", loader_cls=UnstructuredODTLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".csv") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.csv", loader_cls=UnstructuredCSVLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".pptx") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".md") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".org") for File in os.listdir(obj)):
            abc={'mode': "single", 'strategy': "hi_res"}
            loader = DirectoryLoader(
                obj, glob="**/*.org", loader_cls=UnstructuredOrgModeLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
    return accumulator

def loaddata(data_path):
    documents = []
    for data in data_path:
        documents.extend(routerloader(data))
    return documents

def remove_blankline(d):
    text = d.page_content.replace('\n\n','\n')
    d.page_content = text
    return d

def initfromcmdlineargs():
    PDF_ROOT_DIR     = ""
    IDOC_PATH        = []
    VDB_PATH         = "./default_chroma_vdb"
    COLLECTION_NAME  = "default_collection_name"
    MODEL_NAME       = ""
    EMBEDDING_NAME   = ""
    REUSE_VDB        = False
    DISPLAY_DOC      = False
    argumentlist = sys.argv[1:]
    options = "hm:e:i:v:c:r:d:"
    long_options = ["help",
                 "model_name=",
                 "embedding_name=",
                 "inputdocs_path=",
                 "vdb_path=",
                 "collection_name=",
                 "reuse=",
                 "display_doc="]
    try:
        arguments, values = getopt.getopt(argumentlist, options, long_options)
        for currentArgument, currentValue in arguments:
            print("currArg ", currentArgument, " currVal", currentValue)
            if currentArgument in ("-h", "--help"):
                print ("Displaying Help:")
                print ("-h or --help to get this help msg")
                print ("-m or --model_name name of the model used, ex:phi3")
                print ("-e or --embedding_name name of the embedding model used, ex:nomic-embed-text")
                print ("-i or --inputdocs_path path list between \" \" to folders or files to be used for RAG")
                print ("-v or --vdb_path path to the directory used as a vector database")
                print ("-c or --collection_name name of the vector database collection")
                print ("-r or --reuse str True or False reuse vector database")
                print("-d or --display_doc str Tur or False whether or not to display partially input documents")
                print("Command line arguments example:")
                print("python --model_name MyModelName --embedding_name MyEmbeddingModel \ ")
                print("--inputdocs_path \"My/Path/To/folder1 My/Path/To/folder2 My/Path/To/file1\" \ ")
                print("--collection_name MyCollectionName \ ")
                print("--reuse False --display-doc False")
                exit()
            elif currentArgument in ("-m", "--model_name"):
                MODEL_NAME = currentValue
            elif currentArgument in ("-e", "--embedding_name"):
                EMBEDDING_NAME = currentValue
            elif currentArgument in ("-i", "--inputdocs_path"):
                for i in currentValue.split(" "):
                    if (len(i) != 0):
                        if (os.path.isfile(i)) or ((os.path.isdir(i)) and (len(os.listdir(i)) != 0)):
                            IDOC_PATH.append(Path(i))
            elif currentArgument in ("-v", "--vdb_path"):
                VDB_PATH = Path(currentValue)
            elif currentArgument in ("-c", "--collection_name"):
                COLLECTION_NAME = currentValue
            elif currentArgument in ("-r", "--reuse"):
                if currentValue.casefold() == "true":
                    REUSE_VDB = True
                else:
                    REUSE_VDB = False
            elif currentArgument in ("-d", "--display_doc"):
                if currentValue.casefold() == "true":
                    DISPLAY_DOC = True
                else:
                    DISPLAY_DOC = False
        return MODEL_NAME, EMBEDDING_NAME, IDOC_PATH, VDB_PATH, COLLECTION_NAME, REUSE_VDB, DISPLAY_DOC
    except getopt.error as err:
        print (str(err))
        exit()

def create_vdb(splitted_data, embedding, vdb_path, collection_name, reuse_vdb):
    """Create a vector database from the documents"""
    Isvectordb = False
    if vdb_path.exists():
        if any(vdb_path.iterdir()):
            if reuse_vdb == True:
                vectordb = Chroma(persist_directory=str(vdb_path),
                                 embedding_function=embedding,
                                 collection_name=collection_name)
                print("vectordb._collection.count() ", vectordb._collection.count())
                Isvectordb = True
                print(f"vector database REUSED in {vdb_path}")
            else:
                shutil.rmtree(str(vdb_path))
                print(f"vector database REMOVED in {vdb_path}")
    else:
        vdb_path.mkdir(parents=True)
    
    if Isvectordb == False:
        vectordb = Chroma.from_documents(
            documents=splitted_data,
            embedding=embedding,
            persist_directory=str(vdb_path),  # Does not accept Path
            collection_name=collection_name
        )
        print(f"vector database CREATED in {vdb_path}")

    return vectordb

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
print("#-----------------------------------#")
print("#           INPUTS ARGS             #")
print("#-----------------------------------#")
MODEL_NAME, EMBEDDING_NAME, IDOC_PATH, VDB_PATH, COLLECTION_NAME, REUSE_VDB, DISPLAY_DOC = initfromcmdlineargs()
print("MODEL      NAME::", MODEL_NAME)
print("EMBEDDING  NAME::", EMBEDDING_NAME)
print("INPUT DOCS PATH::", IDOC_PATH)
print("VDB        PATH::", VDB_PATH)
print("COLLECTION NAME::", COLLECTION_NAME)
print("REUSE      VDB ::", REUSE_VDB)
print("DISPLAY    DOC ::", DISPLAY_DOC)
print("#-----------------------------------#")
print("#   STARTING DATA LOAD AND SPLIT    #")
print("#-----------------------------------#")

splitted_data = None
if REUSE_VDB is False:
    # Load datas
    documents = loaddata(IDOC_PATH)
    print("documents length::", len(documents))
    if DISPLAY_DOC is True:
        for i in range(len(documents)):
            print("Printing document ", i, " :")
            print(documents[i].page_content[0:300])
    
    documents = [remove_blankline(d) for d in documents]
    if DISPLAY_DOC is True:
        for i in range(len(documents)):
            print("Printing document after remove_blankline() ", i, " :")
            print(documents[i].page_content[0:300])
    # Split and chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", r"(?<=\. )",  " ", "",
        "\u200b","\uff0c","\u3001","\uff0e","\u3002",]
        )
    splitted_data = splitter.split_documents(documents)
    if DISPLAY_DOC is True:
        for i in range(len(splitted_data)):
            print("Printing splitted_data ", i, " :")
            print(splitted_data[i].page_content[0:300])
    # Split and chunk
print("#-----------------------------------#")
print("#     STARTING VECTOR DATABASE      #")
print("#-----------------------------------#")
# Embedding Using Ollama
ollama_embeddings = OllamaEmbeddings(
    model=EMBEDDING_NAME,
    show_progress=True
)
# Add to vector database
vectordb = create_vdb(splitted_data,
                     ollama_embeddings,
                     Path(VDB_PATH),
                     COLLECTION_NAME,
                     REUSE_VDB)
print("vectordb._collection.count() ", vectordb._collection.count())
print("#-----------------------------------#")
print("#     STARTING LLM AND PROMPT RAG   #")
print("#-----------------------------------#")
# LLM model
local_model = MODEL_NAME
llm = ChatOllama(model=local_model, temperature=0)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate four
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
question = ""
print("#-----------------------------------#")
print("#         STARTING  RAG Q&A         #")
print("#-----------------------------------#")
question = ""
while True:
    print("*"*20)
    print("Enter a QUESTION: (to exit enter q or quit)")
    question = input("")
    if question.casefold() == "q" or question.casefold() == "quit":
        print("End QA")
        break
    else:
        response = chain.invoke(question)
        print(response)
