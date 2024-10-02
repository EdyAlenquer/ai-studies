from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langserve import add_routes

# Create prompt template
system_template = """
    Translate the following text into {target_language} and 
    give examples of common phrases that use parts of the text
"""
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{input_text}')
])

# Setup llama
llm = ChatOllama(
    model='llama3.1:8b',
    temperature=0
)

# Create chain
chain = prompt_template | llm | StrOutputParser()


response = chain.invoke({
    'input_text': "I like to eat bread with butter.",
    'target_language': "Portuguese"
})


# Setup API
app = FastAPI(
    title="LangChain Server with Llama 3.1 (8b)",
    version="0.1",
    description="A simple API for the LangChain using Llama model."
)

# Add chain route
add_routes(app, chain, path='/translate')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

