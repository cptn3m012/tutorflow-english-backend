from app.schemas import LessonVariant


READING = (
    "Anna visits a small cafe near her office on Monday morning. "
    "She reads the menu and asks the waiter for tea and a sandwich. "
    "The waiter brings a clean cup and a glass of water to her table. "
    "Anna likes the quiet room because she can read her book there. "
    "After breakfast, she asks for the bill and pays with her card. "
    "She thanks the waiter and walks to work. "
    "She plans to visit the cafe again with her friend on Friday."
)

COMPREHENSION = {"items": [
    {"question": "Where is the cafe?", "answer": "Near Anna's office.",
     "evidence": "Anna visits a small cafe near her office on Monday morning."},
    {"question": "What does Anna order?", "answer": "Tea and a sandwich.",
     "evidence": "She reads the menu and asks the waiter for tea and a sandwich."},
    {"question": "How does Anna pay?", "answer": "With her card.",
     "evidence": "After breakfast, she asks for the bill and pays with her card."},
]}


def make_lesson(**updates) -> LessonVariant:
    data = {
        "title": "At the Cafe",
        "lesson_goal": "Order food and drinks politely.",
        "target_vocabulary": ["menu", "tea", "sandwich", "cup", "water", "table", "bill", "waiter"],
        "reading_text": READING,
        "reading_questions": ["Where does Anna go?", "What does she order?", "When does she return?"],
        "speaking_questions": [
            "What do you order in a cafe?", "Do you prefer tea or coffee?",
            "Where is your favourite cafe?", "Who do you go with?", "How do you ask for the bill?",
        ],
        "pair_work_task": "Student A orders breakfast. Student B is the waiter.",
        "role_play_scenario": "Order a drink and a sandwich, then ask for the bill.",
        "sentence_frames": ["Can I have __, please?", "I would like __.", "How much is it?"],
        "cefr_focus": ["Making requests (e.g. at a restaurant)", "Food and drinks", "Question words"],
    }
    data.update(updates)
    return LessonVariant.model_validate(data)
