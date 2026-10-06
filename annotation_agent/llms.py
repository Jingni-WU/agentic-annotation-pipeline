"""LLM setup: real Claude models, or MockLLM for free testing."""
import random
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from annotation_agent import config
from annotation_agent.schemas import Annotation, Guideline, Label, Taxonomy

load_dotenv(override=True)   # reads ANTHROPIC_API_KEY from .env

FAKE_LABELS = ["shipping", "return_refund", "product_question", "complaint", "praise"]


class MockLLM:
    """
    A fake LLM for testing the graph without calling any API.
    It supports the same methods our nodes use:
      - llm.invoke(prompt)                           -> returns an object with .content
      - llm.with_structured_output(Schema).invoke()  -> returns a Schema object
    """

    def __init__(self, name, seed=0, schema=None):
        self.name = name
        self.seed = seed
        self.schema = schema
        self.rng = random.Random(seed)   # fixed seed = same results every run

    def with_structured_output(self, schema):
        return MockLLM(self.name, self.seed, schema)

    def invoke(self, prompt):
        time.sleep(config.MOCK_DELAY)
        print(f"    [MOCK {self.name}] called")

        if self.schema is Taxonomy:
            return Taxonomy(
                labels=[Label(name=n, definition=f"Fake definition of {n}") for n in FAKE_LABELS]
            )

        if self.schema is Guideline:
            return Guideline(rules=[
                "Fake rule 1: pick the customer's main intent",
                "Fake rule 2: use complaint for negative service experiences",
            ])

        if self.schema is Annotation:
            r = self.rng.random()
            if r < 0.03:
                raise ValueError("Simulated parse error")
            if r < 0.06:
                return Annotation(label="made_up_label", reason="fake")
            default = FAKE_LABELS[len(prompt) % len(FAKE_LABELS)]
            label = default if self.rng.random() < 0.75 else self.rng.choice(FAKE_LABELS)
            return Annotation(label=label, reason=f"Fake reason from {self.name}")

        return AIMessage(content=f"Fake disagreement report from {self.name}")


if config.USE_MOCK:
    llm_generator = MockLLM("generator")
    llm_a = MockLLM("annotator_a", seed=1)
    llm_b = MockLLM("annotator_b", seed=2)
    llm_judge = MockLLM("judge", seed=3)
else:
    llm_generator = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.2)
    llm_a = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    llm_b = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperature=0)
    llm_judge = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
