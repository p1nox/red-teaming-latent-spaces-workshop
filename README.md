# red-teaming-latent-spaces-workshop

> Workshop: Red Teaming Latent Spaces & Protecting LLM apps.

### Setup

*Requirements:*
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [uv](https://pypi.org/project/uv/).
- [Ollama](https://www.ollama.com/download/linux) or a [OpenAI key](https://platform.openai.com/api-keys).

```bash
# set env vars and load
mv template.env .env
source .env

# using conda
conda create -n red-teaming python=3.12
conda activate red-teaming
pip install -r requirements.txt

# using uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# setup guardrails
# https://hub.guardrailsai.com/keys
guardrails configure

# install guard rails
guardrails hub install hub://guardrails/detect_jailbreak
guardrails hub install hub://guardrails/unusual_prompt
guardrails hub install hub://guardrails/gibberish_text

jupyter notebook

# deactivate env
conda deactivate
deactivate
```

*Windows installation:*

```bash
# using powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "C:\Users\Duoc\.local\bin;$env:Path"
uv venv --python 3.12
.\.venv\Scripts\activate.ps1
uv pip install -r requirements.txt
```

*Ollama models used:*

```
ollama pull gemma3:4b
ollama pull gemma3:1b
ollama pull mistral:7b
ollama pull deepseek-r1:7b
ollama pull deepseek-r1:8b
```
