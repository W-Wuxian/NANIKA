# Testing [Mistral-src](https://github.com/mistralai/mistral-src/tree/main) 

# Get Nanika repo
```bash
cd $HOME/Bureau
git clone nanika repo
```
## Activate Mistral dependencies:
```bash
conda activate nanika_env
```
## Download the model
```bash
mkdir $HOME/Bureau/MISTRAL_DWL_NANIKA && cd $HOME/Bureau/MISTRAL_DWL_NANIKA
wget https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```
## run the model
```bash
cd $HOME/Bureau/NANIKA
python -m main demo $HOME/Bureau/MISTRAL_DWL_NANIKA/mistral-7B-v0.1/
# To give your own prompts
python -m main interactive $HOME/Bureau/MISTRAL_DWL_NANIKA/mistral-7B-v0.1/
```
And other things at [README Mistral-src](https://github.com/mistralai/mistral-src/tree/main)
