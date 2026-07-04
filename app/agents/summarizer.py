from app.agents.base import BaseAgent


class SummarizerAgent(BaseAgent):
    name = "summarizer"
    system_prompt = (
        "Summarizer Agent\n"
        "You take everything produced by the other agents and condense it "
        "into a single, direct answer for the end user. No meta-commentary "
        "about the process, just the final answer."
    )

    def build_prompt(self, query: str, context: dict) -> str:
        research = context.get("researcher_output", "")
        code = context.get("coder_output", "")
        combined = "\n\n".join(filter(None, [research, code]))
        return f"Original query: {query}\n\nWork produced by the team:\n{combined}\n\nWrite the final answer."
