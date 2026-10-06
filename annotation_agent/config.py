"""Settings you may want to change between runs."""

# ---------- LLMs ----------
USE_MOCK = True          # True: fake LLMs (free, fast). False: real Claude API.
MOCK_DELAY = 0.5         # seconds per fake call, so you can press Ctrl+C when testing
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# ---------- Iteration loop ----------
IAA_THRESHOLD = 0.7      # minimum acceptable Cohen's kappa
MAX_ROUNDS = 3           # maximum number of annotation rounds

# ---------- Files ----------
CHECKPOINT_DB = "checkpoints.db"
