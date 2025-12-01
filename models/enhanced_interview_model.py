# models/enhanced_interview_model.py
import json
import logging
import os
import random

class EnhancedInterviewModel:
    def __init__(self):
        self.setup_logging()
        self.questions = self.load_questions()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            h.setFormatter(fmt)
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    def load_questions(self):
        """Load Format C: ques + answer"""
        path = "database.jsonl"
        questions = []

        if not os.path.exists(path):
            self.logger.error("❌ database.jsonl not found.")
            return []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)

                    # Format C: convert keys
                    question_text = item.get("question") or item.get("ques") or ""
                    ideal = item.get("ideal_answer") or item.get("answer") or ""

                    if not question_text.strip():
                        continue

                    questions.append({
                        "question": question_text.strip(),
                        "ideal_answer": ideal.strip()
                    })

                except Exception as e:
                    self.logger.warning(f"Skipping invalid line: {e}")

        self.logger.info(f"Loaded {len(questions)} questions")
        return questions

    def get_question(self, used_questions=None):
        """Return EXACT DB reference"""
        if used_questions is None:
            used_questions = []

        available = [q for q in self.questions if q["question"] not in used_questions]

        if not available:
            available = self.questions.copy()

        return random.choice(available)

    def get_actual_correct_answer(self, q):
        """Return correct answer directly from DB entry"""
        return q.get("ideal_answer", "").strip()
