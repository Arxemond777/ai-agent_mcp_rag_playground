# Description
My little playground with AI agents, MCP and RAGs


## Requirements  
```bash
python -m pip install -r requirements.txt
```

### Create an .env dir with API_KEYS and after it set env vars there
* For launching the app.run you need to generate an GROQ_API_KEY here https://console.groq.com/keys  
* For app/some_experiments/get_models_and_deployments.py you need a DIAL key

The example and initialization of it below  
```bash
# how it should look like
cat .env
export GROQ_API_KEY="***"
export DIAL_API_KEY="***"

# initialization of it
source .env
```

Scans the ./kb folder, splits supported text files into chunks, generates embeddings with sentence-transformers/all-MiniLM-L6-v2, and stores them in a persistent Chroma vector database (./data/chroma) for later semantic search.  
```bash
python -m app.run index
```

Ask something  
```bash
python -m app.run ask_llm "what is about the README.md? gimme key items"

python -m app.run ask_llm "Who is the line manager of my line manager?"
```