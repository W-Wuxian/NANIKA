# Testing [Mistral-src](https://github.com/mistralai/mistral-src/tree/main) 

# Get Nanika repo
```bash
git clone nanika repo
```
## Activate Mistral dependencies:
```bash
conda activate nanika_env
```
## Download the model
```bash
wget https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```
## run the model
```bash
python -m main demo /path/to/mistral-7B-v0.1/
# To give your own prompts
python -m main interactive /path/to/mistral-7B-v0.1/
```
And the other things at [README Mistral-src](https://github.com/mistralai/mistral-src/tree/main)
