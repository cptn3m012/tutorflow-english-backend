from datetime import datetime


def resolve_theme(topic: str | None, lesson_date: str | None) -> tuple[str, str]:
    if topic and topic.strip():
        return topic.strip(), "manual"

    if not lesson_date:
        return "general conversation", "fallback"

    date_obj = datetime.strptime(lesson_date, "%Y-%m-%d")
    month = date_obj.month
    day = date_obj.day

    if month == 12:
        return "Christmas traditions and winter holidays", "calendar"
    if month == 10:
        return "Halloween and autumn activities", "calendar"
    if month in [3, 4]:
        return "spring, Easter, and family traditions", "calendar"
    if month == 2 and day <= 20:
        return "Valentine's Day, friendship, and food", "calendar"
    if month in [6, 7, 8]:
        return "summer holidays and travel", "calendar"

    return "daily life and conversation topics", "calendar"