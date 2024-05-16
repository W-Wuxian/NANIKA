# Testing [langchain rag](https://github.com/tonykipkemboi/ollama_pdf_rag/blob/main/local_ollama_rag.ipynb)
# Get branch ollama-rag from Nanika repo
The branch focus the use of ollama and langchain to RAG from your documents. (see list of supported file extensions at the end)
```bash
cd $HOME
git clone -b ollama-rag --single-branch git@github.com:W-Wuxian/NANIKA.git
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
The *main.py* script is used to create a database from your documents as follow:
```bash
python main.py --help
options or long_options are:
-m or --model_name model name
-e or --embedding_name embedding name
-i or --inputdocs_path  path given between " " to folders or files to be used at RAG step
-v or --vdb_path vector data base path
-c or --collection_name collection name
-r or --reuse reuse previous vdb and collection
-d or --display-doc whether or not to display given documents
```
So for example using *phi3* llm model, with *nomic-embed-text* as an embedding model to create a database from my documents at /path/to/my/folder/ one can use the following command:
```bash
python main.py -m phi3 -e nomic-embed-text -i "/path/to/my/folder1 /path/to/my/folder2 /path/to/my/file1"
```
In order to run several database  we need to specify the database storing location via *-v* and the collection name via *-c*, as follow:
```bash
python main.py -m phi3 -e nomic-embed-text -i /path/to/my/folder1/ -v ./database1 -c collection1
python main.py -m phi3 -e nomic-embed-text -i /path/to/my/folder2/ -v ./database2 -c collection2
```
The main.py script will also ask you to enter questions (RAG), to end this phase enter *q* or *quit*.

### Reusing a database and QA loop
To reuse a database you need the corresponding *-v* and *-c* and run the *main.py* script with *-r True* as follow:
```bash
python reuse.py -m phi3 -e nomic-embed-text -v ./database1 -c collection1 -r True
python reuse.py -m phi3 -e nomic-embed-text -v ./database2 -c collection2 -r True
```

### File extension coverage

| file extension | Coverage           |
| -------------- | ------------------ |
| pdf            | :heavy_check_mark: |
| txt            | :heavy_check_mark: |
| py             | :heavy_check_mark: |
| png            | :heavy_check_mark: |
| jpg            | :heavy_check_mark: |
| xlsx           | :heavy_check_mark: |
| xls            | :heavy_check_mark: |
| odt            | :heavy_check_mark: |
| csv            | :heavy_check_mark: |
| pptx           | :heavy_check_mark: |
| md             | :heavy_check_mark: |
| org            | :heavy_check_mark: |