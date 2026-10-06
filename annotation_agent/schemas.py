"""Pydantic schemas for structured LLM output."""
from pydantic import BaseModel, Field


class Label(BaseModel):
    name: str = Field(description="Short label name")
    definition: str = Field(description="Definition of the label")


class Taxonomy(BaseModel):
    labels: list[Label] = Field(description="List of labels; labels must be mutually exclusive")


class Guideline(BaseModel):
    rules: list[str] = Field(description="Annotation rules, including how to handle edge cases")


class Annotation(BaseModel):
    label: str = Field(description="Pick one label from the given list")
    reason: str = Field(description="One-sentence reason")
