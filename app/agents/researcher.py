from app.agents.base import BaseAgent


class ResearcherAgent(BaseAgent):
    name = "researcher"
    system_prompt = (
        "Researcher Agent\n"
        "You gather and structure factual context for a downstream agent to use. "
        "Be concise, prefer bullet points, and explicitly flag anything you're "
        "uncertain about instead of guessing. Do not write code or final answers, "
        "only the background information needed to produce one."
    )

    def build_prompt(self, query: str, context: dict) -> str:
        prior = context.get("critic_feedback")
        if prior:
            return (
                f"Original query: {query}\n\n"
                f"The critic previously rejected the answer for this reason: {prior}\n"
                f"Re-research with that gap specifically in mind."
            )
        return f"Research the following request and list the key facts needed to answer it:\n{query}"
