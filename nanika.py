from pathlib import Path
import getopt, sys, os, shutil

from collections import Counter
import torch
import random
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)

from langchain_community.document_loaders import (
    DirectoryLoader, UnstructuredPDFLoader, TextLoader,
    PythonLoader, UnstructuredImageLoader,
    UnstructuredExcelLoader, UnstructuredWordDocumentLoader, UnstructuredXMLLoader,
    UnstructuredCSVLoader, UnstructuredPowerPointLoader, UnstructuredODTLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.embeddings import (
    OllamaEmbeddings,
    HuggingFaceEmbeddings
)
from langchain_community.llms import (
    HuggingFacePipeline,
    #Ollama
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain.indexes import VectorstoreIndexCreator

from langchain.prompts import (ChatPromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import (
    ChatOllama
)
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torchdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAXNEWTOKENS = 524

def rag_generation(query, tokenizer, model, vectordb, k=3, fetch_k=6, **gen_parameters):
    """Generate text from a prompt after rag and print it."""
    docs = vectordb.max_marginal_relevance_search(query, k, fetch_k)
    retrieved_infos = " ".join([doc.page_content for doc in docs])
    
    text_input = f"With the following informations: {retrieved_infos}\nAnswer this question: {query}"

    model_inp = tokenizer(text_input, return_tensors="pt").to(torchdevice)
    input_nb_tokens = model_inp['input_ids'].shape[1]
    print(torchdevice, input_nb_tokens)
    out = model.generate(input_ids=model_inp["input_ids"], **gen_parameters)
    #print(f"LLM input:\n{text_input}\n" + "#"*50)
    #print(f"LLM output:\n{tokenizer.decode(out[0][input_nb_tokens:])}")
    return tokenizer.decode(out[0][input_nb_tokens:])

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
        # BEGIN F90 C .h CPP As TextLoader
        if Fname.endswith(".f90") or Fname.endswith(".c") or Fname.endswith(".h") or Fname.endswith(".cpp"):
            loader = TextLoader(obj, autodetect_encoding = True)
        # END F90 C .h CPP As TextLoader
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
        # BEGIN F90 C .h CPP As TextLoader
        if any(File.endswith(".f90") for File in os.listdir(obj)):
            abc={'autodetect_encoding': True}
            loader = DirectoryLoader(
                obj, glob="**/*.f90", loader_cls=TextLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".c") for File in os.listdir(obj)):
            abc={'autodetect_encoding': True}
            loader = DirectoryLoader(
                obj, glob="**/*.c", loader_cls=TextLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".h") for File in os.listdir(obj)):
            abc={'autodetect_encoding': True}
            loader = DirectoryLoader(
                obj, glob="**/*.h", loader_cls=TextLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        if any(File.endswith(".cpp") for File in os.listdir(obj)):
            abc={'autodetect_encoding': True}
            loader = DirectoryLoader(
                obj, glob="**/*.cpp", loader_cls=TextLoader,
                loader_kwargs=abc, show_progress=True, use_multithreading=True
            )
            accumulator.extend(loader.load())
        # END F90 C .h CPP As TextLoader
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
    API_NAME         = "HFE"
    MODEL_NAME       = ""
    EMBEDDING_NAME   = ""
    PDF_ROOT_DIR     = ""
    IDOC_PATH        = []
    VDB_PATH         = "./default_chroma_vdb"
    COLLECTION_NAME  = "default_collection_name"
    BASE_URL         = "local" # local i.e local serve, no base_url, elif BASE_URL[0:6]=="hermes",
    #base_url='http://192.168.2.16:11434' elif BASE_URL[0:4]=="http", base_url=BASE_URL
    REUSE_VDB        = False
    DISPLAY_DOC      = False
    argumentlist = sys.argv[1:]
    options = "ha:m:e:i:v:c:b:r:d:"
    long_options = ["help",
                 "api_name=",
                 "model_name=",
                 "embedding_name=",
                 "inputdocs_path=",
                 "vdb_path=",
                 "collection_name=",
                 "base_url=",
                 "reuse=",
                 "display_doc="]
    try:
        arguments, values = getopt.getopt(argumentlist, options, long_options)
        for currentArgument, currentValue in arguments:
            print("currArg ", currentArgument, " currVal", currentValue)
            if currentArgument in ("-h", "--help"):
                print ("Displaying Help:")
                print ("-h or --help to get this help msg")
                print ("-a or --api_name name of the embedding API to be used:")
                print ("OLE for OllamaEmbeddings or HFE-PIP for HuggingfaceEmbeddings with pipeline")
                print ("-m or --model_name name of the model used, ex:phi3")
                print ("-e or --embedding_name name of the embedding model used, ex:nomic-embed-text")
                print ("-i or --inputdocs_path path list between \" \" to folders or files to be used for RAG")
                print ("-v or --vdb_path path to the directory used as a vector database")
                print ("-c or --collection_name name of the vector database collection")
                print ("Collection name Rules:")
                print ("(1) contains 3-63 characters")
                print ("(2) starts and ends with an alphanumeric character")
                print ("(3) otherwise contains only alphanumeric characters, underscores or hyphens (-)")
                print ("(4) contains no two consecutive periods (..) and")
                print ("(5) is not a valid IPv4 address")
                print ("-b or --base_url Base url the model is hosted under")
                print ("-r or --reuse str True or False reuse vector database")
                print ("-d or --display_doc str Tur or False whether or not to display partially input documents")
                print ("Command line arguments example:")
                print ("python --api_name OLE --model_name MyModelName --embedding_name MyEmbeddingModel \ ")
                print ("--inputdocs_path \"My/Path/To/folder1 My/Path/To/folder2 My/Path/To/file1\" \ ")
                print ("--collection_name MyCollectionName \ ")
                print ("--reuse False --display-doc False")
                exit()
            elif currentArgument in ("-a", "--api_name"):
                API_NAME = currentValue
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
            elif currentArgument in ("-b", "--base_url"):
                BASE_URL = currentValue
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
        return API_NAME, MODEL_NAME, EMBEDDING_NAME, IDOC_PATH, VDB_PATH, COLLECTION_NAME, BASE_URL, REUSE_VDB, DISPLAY_DOC
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
API_NAME, MODEL_NAME, EMBEDDING_NAME, IDOC_PATH, VDB_PATH, COLLECTION_NAME, BASE_URL, REUSE_VDB, DISPLAY_DOC = initfromcmdlineargs()
print("API        NAME::", API_NAME)
print("MODEL      NAME::", MODEL_NAME)
print("EMBEDDING  NAME::", EMBEDDING_NAME)
print("INPUT DOCS PATH::", IDOC_PATH)
print("VDB        PATH::", VDB_PATH)
print("COLLECTION NAME::", COLLECTION_NAME)
print("BASE       URL ::", BASE_URL)
print("REUSE      VDB ::", REUSE_VDB)
print("DISPLAY    DOC ::", DISPLAY_DOC)
print("#-----------------------------------#")
print("#   STARTING DATA LOAD AND SPLIT    #")
print("#-----------------------------------#")
# https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
splitted_data = None
if REUSE_VDB is False:
    # Load datas
    documents = loaddata(IDOC_PATH)
    print("Type of documents::", type(documents))
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
    print("Type of splitted_data::", type(splitted_data))
    if DISPLAY_DOC is True:
        for i in range(len(splitted_data)):
            print("Printing splitted_data ", i, " :")
            print(splitted_data[i].page_content[0:300])

print("#-----------------------------------#")
print("#     STARTING EMBEDDINGS           #")
print("#-----------------------------------#")
api_embeddings = None
if API_NAME == "OLE":
    # Embedding Using Ollama
    # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.ollama.OllamaEmbeddings.html#langchain-community-embeddings-ollama-ollamaembeddings
    if BASE_URL == "local":
        ollama_embeddings = OllamaEmbeddings(
            model=EMBEDDING_NAME,
            show_progress=True
        )
    elif BASE_URL == "hermes":
        ollama_embeddings = OllamaEmbeddings(
            base_url='http://192.168.2.16:11434',#base_url='http://192.168.2.16 OR 127.0.0.1 :11434' /api/embeddings
            model=EMBEDDING_NAME,
            show_progress=True
        )
    elif BASE_URL[0:4] == "http":
        ollama_embeddings = OllamaEmbeddings(
            base_url=BASE_URL,
            model=EMBEDDING_NAME,
            show_progress=True
        )
    api_embeddings = ollama_embeddings
elif API_NAME[0:3] == "HFE":
    #Embedding using HuggingFace
    #"Snowflake/snowflake-arctic-embed-l"
    #https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html
    #https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_NAME),
        multi_process=False,
        show_progress=True,
        model_kwargs={"device": torchdevice},
        encode_kwargs={"normalize_embeddings": False},  # Set `True` for cosine similarity
    )
    api_embeddings = huggingface_embeddings
print("#-----------------------------------#")
print("#     STARTING VECTOR DATABASE      #")
print("#-----------------------------------#")
# Add to vector database
vectordb = create_vdb(splitted_data,
                     api_embeddings,
                     Path(VDB_PATH),
                     COLLECTION_NAME,
                     REUSE_VDB)
print("vectordb._collection.count() ", vectordb._collection.count())
print("#-----------------------------------#")
print("#     STARTING LLM AND PROMPT RAG   #")
print("#-----------------------------------#")
# LLM model
local_model = MODEL_NAME
if API_NAME == "OLE":
    if BASE_URL == "local":
        llm = ChatOllama(
            model=local_model,
            num_ctx=MAXNEWTOKENS,
            temperature=0
        )
    elif BASE_URL == "hermes":
        llm = ChatOllama(
            base_url='http://192.168.2.16:11434',#base_url='http://192.168.2.16 OR 127.0.0.1 :11434' /api/chat
            model=local_model,
            num_ctx=MAXNEWTOKENS,
            temperature=0
        )
    elif BASE_URL[0:4] == "http":
        llm = ChatOllama(
            base_url=BASE_URL,
            model=local_model,
            num_ctx=MAXNEWTOKENS,
            temperature=0
        )
elif API_NAME[0:3] == "HFE":
    tokenizer = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(local_model,
    device_map=torchdevice,
    torch_dtype="auto", #torch.bfloat16 torch.int8 torch.uint8
    trust_remote_code=True,
    attn_implementation="flash_attention_2")#.to(torchdevice)
    if API_NAME == "HFE-PIP":
        hfpipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            #device=0,
            device_map=torchdevice,
            max_new_tokens=MAXNEWTOKENS,
            temperature=0,
            #do_sample=False,
            )
        llm = HuggingFacePipeline(pipeline=hfpipe)

QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate four
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

if API_NAME == "OLE" or API_NAME == "HFE-PIP":
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
        if API_NAME == "OLE" or API_NAME == "HFE-PIP":
            response = chain.invoke(question)
            print(response)
        elif API_NAME == "HFE-RGN":
            response = rag_generation(question, tokenizer, model, vectordb, k=4, fetch_k=8, max_new_tokens=MAXNEWTOKENS)
            print(response)