# Testing [langchain rag](https://github.com/tonykipkemboi/ollama_pdf_rag/blob/main/local_ollama_rag.ipynb)
# Get Nanika repo
See how to clone a project with Submodules at [Git-Tools-submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
```bash
cd $HOME
git clone nanika git@github.com:W-Wuxian/NANIKA.git
git submodule init
git submodule update
```
Or in One line:
```bash
git clone --recurse-submodules git@github.com:W-Wuxian/NANIKA.git
```
## Install [ollama](https://github.com/ollama/ollama?tab=readme-ov-file)
After a successful installation run:
```bash
ollama pull nomic-embed-text
ollama pull phi
ollama list
```

## Activate langchain dependencies:

```bash
conda create -f langchain_rag_env.yml
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
