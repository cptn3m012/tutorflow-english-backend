"""Lesson vocabulary and classroom question contracts, separate from image pixels."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QUESTION_KINDS = ("observation", "application", "discussion")


class VocabularyVisualLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str = Field(min_length=1, max_length=100)
    visual_cue: str = Field(min_length=5, max_length=300)


class PictureQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=5, max_length=300)
    answer: str = Field(min_length=1, max_length=400)
    evidence: str = Field(min_length=3, max_length=300)
    object_names: list[str] = Field(min_length=1, max_length=3)
    kind: Literal["observation", "application", "discussion"] = "observation"
    target_words: list[str] = Field(default_factory=list, max_length=16)


class PictureLearningActivity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[PictureQuestion] = Field(min_length=3, max_length=3)
    sentence_frames: list[str] = Field(min_length=2, max_length=3)


class LearningActivityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = Field(strict=True)
    issues: list[str] = Field(max_length=5)

    @model_validator(mode="after")
    def consistent_verdict(self):
        if self.approved and self.issues:
            raise ValueError("Approved activities must have no unresolved issues.")
        if not self.approved and (not self.issues or not all(issue.strip() for issue in self.issues)):
            raise ValueError("Rejected activities need specific repair feedback.")
        return self


def lesson_vocabulary(context: dict) -> list[str]:
    return list(dict.fromkeys(word.strip() for word in context.get("vocabulary", [])
                             if isinstance(word, str) and word.strip()))[:16]


def practice_words(questions: list[dict], vocabulary: list[str]) -> list[str]:
    used = {word for item in questions for word in item.get("target_words", [])}
    return [word for word in vocabulary if word in used]


def vocabulary_root(word: str) -> str:
    # Accept grammatical word families, rather than forcing noun phrases into
    # every answer (e.g. data is collected / data collection, monitor / monitoring).
    families = {
        "collection": "collect", "collected": "collect", "collecting": "collect", "collections": "collect",
        "privacy": "private", "security": "secure", "invasion": "invade", "invading": "invade", "invaded": "invade",
    }
    if word in families:
        return families[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def uses_vocabulary(word: str, text: str) -> bool:
    expected = [vocabulary_root(token) for token in re.findall(r"[a-z]+", word.casefold())]
    tokens = [vocabulary_root(token) for token in re.findall(r"[a-z]+", text.casefold())]
    if not expected:
        return False
    if len(expected) == 1:
        return expected[0] in tokens
    for start in range(len(tokens)):
        window = tokens[start:start + len(expected) + 3]
        if all(window.count(token) >= expected.count(token) for token in expected):
            return True
    return False


def validate_learning_activity(activity: PictureLearningActivity, vocabulary: list[str], names: list[str]):
    if tuple(question.kind for question in activity.questions) != QUESTION_KINDS:
        raise ValueError("Use exactly one observation, application and discussion question, in that order.")
    seen = set()
    for item in activity.questions:
        key = " ".join(item.question.split()).casefold()
        if key in seen:
            raise ValueError("Picture questions must be distinct.")
        seen.add(key)
        if not set(item.object_names).issubset(names):
            raise ValueError("Every question needs a basis in observed objects.")
        if not set(item.target_words).issubset(vocabulary) or len(set(item.target_words)) != len(item.target_words):
            raise ValueError("Question vocabulary must use distinct original lesson words.")
        if vocabulary and item.kind != "observation" and not item.target_words:
            raise ValueError("Application and discussion questions must practice lesson vocabulary.")
        text = " ".join((item.question + " " + item.answer).split())
        for word in item.target_words:
            if not uses_vocabulary(word, text):
                raise ValueError(f"Use the declared vocabulary '{word}' in the question or model answer.")
    covered = practice_words([item.model_dump() for item in activity.questions], vocabulary)
    if len(covered) < min(4, len(vocabulary)):
        unused = [word for word in vocabulary if word not in covered]
        raise ValueError("Practice at least four different lesson words when available. Currently used: "
                         + ", ".join(covered) + ". Add relevant words from: " + ", ".join(unused)
                         + " to a question or answer, not only to evidence or tags.")
    frames = [" ".join(frame.split()) for frame in activity.sentence_frames]
    if any("____" not in frame for frame in frames) or len(set(frames)) != len(frames):
        raise ValueError("Provide distinct sentence starters with ____ for the student's own ideas.")
