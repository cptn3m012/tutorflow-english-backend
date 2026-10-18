from app.quiz_schemas import LessonExitQuiz
from app.quiz_service import quiz_source_token


def quiz_draft():
    return {"title": "Cafe quick check", "instruction": "Choose one answer for questions 1 and 2. Write one request for question 3.", "questions": [
        {"kind": "multiple_choice", "focus": "vocabulary", "target": "bill",
         "question": "Which word means a list of what you need to pay?", "options": ["bill", "cup", "waiter"],
         "answer": "bill", "explanation": "A bill shows how much the customer must pay."},
        {"kind": "multiple_choice", "focus": "grammar", "target": "Question words",
         "question": "Complete: ____ is your favourite cafe?", "options": ["Where", "Who", "When"],
         "answer": "Where", "explanation": "Where asks about a place."},
        {"kind": "short_answer", "focus": "communication", "target": "Order food and drinks politely.",
         "question": "Ask a waiter politely for a cup of tea.", "options": [],
         "answer": "Could I have a cup of tea, please?", "explanation": "Accept another polite request for tea. Exact wording is not required."},
    ]}


def quiz_for(content, level="A2"):
    return LessonExitQuiz(**quiz_draft(), level=level, source_token=quiz_source_token(content, level))
