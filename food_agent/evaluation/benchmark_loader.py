"""HD-EPIC VQA benchmark loader."""

import json
from pathlib import Path
from typing import Dict, List, Optional


# HD-EPIC VQA categories
VQA_CATEGORIES = [
    "recipe",
    "ingredient",
    "nutrition",
    "fine_grained_action",
    "3d_perception",
    "object_motion",
    "gaze",
]


class BenchmarkLoader:
    """Load and query the HD-EPIC VQA benchmark (26K questions)."""

    def __init__(self, benchmark_path: str):
        self.path = Path(benchmark_path)
        self._questions: List[Dict] = []
        self._by_category: Dict[str, List[Dict]] = {}
        self._load()

    def _load(self):
        """Load benchmark questions from file."""
        if self.path.is_dir():
            # Load from directory of JSON/JSONL files
            for f in sorted(self.path.glob("*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self._questions.extend(data)
                elif isinstance(data, dict) and "questions" in data:
                    self._questions.extend(data["questions"])
            for f in sorted(self.path.glob("*.jsonl")):
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            self._questions.append(json.loads(line))
        elif self.path.suffix == ".json":
            with open(self.path) as f:
                data = json.load(f)
            if isinstance(data, list):
                self._questions = data
            elif isinstance(data, dict):
                self._questions = data.get("questions", data.get("samples", []))
        elif self.path.suffix == ".jsonl":
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._questions.append(json.loads(line))

        # Index by category
        for q in self._questions:
            cat = q.get("category", q.get("type", "general"))
            self._by_category.setdefault(cat, []).append(q)

    def get_questions(self, category: Optional[str] = None) -> List[Dict]:
        """Get questions, optionally filtered by category."""
        if category:
            return self._by_category.get(category, [])
        return list(self._questions)

    def get_question_detail(self, question_id: str) -> Optional[Dict]:
        """Get a specific question by ID."""
        for q in self._questions:
            if str(q.get("id", q.get("question_id", ""))) == str(question_id):
                return q
        return None

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self._by_category.keys())

    @property
    def total_questions(self) -> int:
        return len(self._questions)
