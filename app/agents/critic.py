from app.agents.base import BaseAgent


class CriticAgent(BaseAgent):
    name = "critic"
    system_prompt = (
        "Critic Agent\n"
        "You review the work produced by the researcher/coder against the "
        "original query. Respond starting with 'VERDICT: PASS' or "
        "'VERDICT: FAIL' followed by one or two sentences of reasoning. Be "
        "strict about correctness but do not nitpick style."
    )

    def build_prompt(self, query: str, context: dict) -> str:
        work = context.get("coder_output") or context.get("researcher_output") or ""
        return (
            f"Original query: {query}\n\n"
            f"Work to review:\n{work}\n\n"
            f"Does this fully and correctly address the query?"
        )

    @staticmethod
    def parse_verdict(critic_output: str) -> tuple[bool, str]:
        passed = "VERDICT: PASS" in critic_output.upper().replace("  ", " ")
        # keep everything after the verdict line as the feedback reason
        reason = critic_output.split(".", 1)[-1].strip() if "." in critic_output else critic_output
        return passed, reason
