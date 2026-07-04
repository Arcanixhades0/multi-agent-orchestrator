from app.agents.base import BaseAgent


class CoderAgent(BaseAgent):
    name = "coder"
    system_prompt = (
        "Coder Agent\n"
        "You write or fix code based on the query and any research context "
        "provided. Prefer small, correct, testable functions over clever ones. "
        "If the critic sent back feedback, treat it as a code review comment "
        "and address it directly rather than rewriting from scratch."
    )

    def build_prompt(self, query: str, context: dict) -> str:
        research = context.get("researcher_output", "")
        feedback = context.get("critic_feedback")
        parts = [f"Task: {query}"]
        if research:
            parts.append(f"Research context:\n{research}")
        if feedback:
            parts.append(f"Critic feedback to address:\n{feedback}")
        return "\n\n".join(parts)
