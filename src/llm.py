from src.config import LLM_PROVIDER, GROQ_MODEL, OLLAMA_MODEL


def get_llm(temperature=0):

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        extra = {"reasoning_effort": "low"} if GROQ_MODEL.startswith("openai/gpt-oss") else {}
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, max_retries=8, **extra)

    if LLM_PROVIDER == "fake":
        from src.fake_llm import FakeAgentLLM
        return FakeAgentLLM()

    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, reasoning=False)
