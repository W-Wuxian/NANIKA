# Testing [langchain rag](https://github.com/tonykipkemboi/ollama_pdf_rag/blob/main/local_ollama_rag.ipynb)
# Get branch ollama-rag from Nanika repo
The branch focus the use of ollama and langchain to RAG from your *PDF* documents.
```bash
cd $HOME
git clone nanika -b ollama --single-branch git@github.com:W-Wuxian/NANIKA.git
```

## Install [ollama](https://github.com/ollama/ollama?tab=readme-ov-file)
After a successful installation run:
```bash
ollama pull nomic-embed-text 
ollama pull phi3
ollama list
```
*nomic-embed-text* is mandatory but *phi3* can be replaced with any model name at
(ollama.com/library)[https://ollama.com/library]

## Activate langchain dependencies:
After installing ollama materials you need to do the following:
```bash
conda env create -f langchain_rag_env.yml
conda activate langchain_rag_env
pip install "unstructured[all-docs]"
pip install chromadb langchain-text-splitters
```
### Alternative using Python-venv
```bash
python -m venv langchain_rag_venv
pip install --upgrade unstructured langchain "unstructured[all-docs]"
pip install --upgrade chromadb langchain-text-splitters
```

## Running the code:
Once ollama and langchain stuff are done (see previous sections)
you can use RAG. Here is two python scripts *main.py* and *reuse.py* to do so.
### Creating a database and QA loop
The *main.py* script is used to create a database from your *PDF* documents as follow:
```bash
python main.py --help
options are:
-m model name
-e embedding name
-p path to folder containing pdf files
-v vector data base path
-c collection name
```
So for example using *phi3* llm model, with *nomic-embed-text* as an embedding model to create a database from my *PDF* documents at /path/to/my/folder/ one can use the following command:
```bash
python main.py -m phi3 -e nomic-embed-text -p /path/to/my/folder/
```
In order to run several database  we need to specify the databse storing location via --vdb_path and the collection name via --collection_name, as follow:
```bash
python main.py -m phi3 -e nomic-embed-text -p /path/to/my/folder1/ -v ./database1 -c collection1
python main.py -m phi3 -e nomic-embed-text -p /path/to/my/folder2/ -v ./database2 -c collection2
```
The main.py script will also ask you to enter questions (RAG), to end this phase enter *q* or *quit*.

### Reusing a database and QA loop
To reuse a database you need the corresponding *-v* and *-c* and run the *reuse.py* script with *-r True* as follow:
```bash
python reuse.py -m phi3 -e nomic-embed-text -v ./database1 -c collection1 -r True
python reuse.py -m phi3 -e nomic-embed-text -v ./database2 -c collection2 -r True
```