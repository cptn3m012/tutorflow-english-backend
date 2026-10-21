"""Shared teaching context for topic selection, generation and repairs."""
def audience_guidance(english_type: str, age_group: str) -> str:
    course = ("Business English: practise purposeful professional communication, workplace vocabulary and polite register."
              if english_type == "business" else
              "General English: practise everyday communication and broadly useful vocabulary.")
    audience = ("Learners: children. Use age-appropriate interests, concrete situations, playful activities and short clear instructions. "
                "Do not assume employment, adult responsibilities or adult life experience. Keep language at the requested CEFR level; age is not proficiency."
                if age_group == "children" else
                "Learners: adults. Use adult interests, realistic situations and respectful, non-infantilising activities at the requested CEFR level.")
    if age_group == "teens":
        audience = (
            "Learners: teenagers. Use age-appropriate interests such as friendships, school, hobbies, "
            "technology and future plans, with respectful, non-childish activities. "
            "Do not assume adult life experience or full-time employment. "
            "Keep language at the requested CEFR level; age is not proficiency."
        )
        if english_type == "business":
            course += " Use student projects, youth entrepreneurship, career exploration and first-job or internship scenarios."
    if english_type == "business" and age_group == "children":
        course += " Adapt business communication to pretend shops, school fairs and team projects; avoid corporate jargon and adult employment scenarios."
    return course + "\n" + audience
