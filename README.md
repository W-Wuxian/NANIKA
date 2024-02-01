# Testing [Mistral-src](https://github.com/mistralai/mistral-src/tree/main) 

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
## Activate Mistral dependencies:
[pytorch-gpu info](https://pytorch.org/)
```bash
conda activate nanika_env
```
### Alternative using Python-venv
```bash
python -m venv nanika_venv
pip install --upgrade fire sentencepiece torch>=2.1.0 xformers simple-parsing
python -m main demo /path/to/mistral-7B-v0.1/
# To give your own prompts
python -m main interactive /path/to/mistral-7B-v0.1/
```
## Download the model
Warning the arxiv is 14G!
```bash
mkdir $HOME/MISTRAL_DWL_NANIKA && cd $HOME/MISTRAL_DWL_NANIKA
wget https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```
## run the model
```bash
cd $HOME/NANIKA/mistral-src
python -m main demo $HOME/MISTRAL_DWL_NANIKA/mistral-7B-v0.1/
# To give your own prompts
python -m main interactive $HOME/MISTRAL_DWL_NANIKA/mistral-7B-v0.1/
```
And other things at [README Mistral-src](https://github.com/mistralai/mistral-src/tree/main)

# TODO
- [ ] Rerun the env file on my local machine
- [ ] on mistral-hermes add more memory
