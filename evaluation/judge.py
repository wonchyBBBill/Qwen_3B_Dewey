import json
import os
from pathlib import Path

from openai import OpenAI


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QUESTIONS_FILE = Path("evaluation/questions.json")
ANSWER_A_FILE = Path("evaluation/answers_base.json")
ANSWER_B_FILE = Path("evaluation/answers_dewey.json")

OUTPUT_FILE = Path("evaluation/judge_results.json")

MODEL = "gpt-5.6-luna"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


questions = load_json(QUESTIONS_FILE)
answers_a = load_json(ANSWER_A_FILE)
answers_b = load_json(ANSWER_B_FILE)


# If questions.json contains {"questions": [...]}
if isinstance(questions, dict):
    questions = questions["questions"]


# ---------------------------------------------------------
# Organize answers by question ID
# ---------------------------------------------------------

def index_answers(data):

    if isinstance(data, dict):
        data = data.get("answers", data)

    return {
        item["id"]: item
        for item in data
    }


answers_a = index_answers(answers_a)
answers_b = index_answers(answers_b)


# ---------------------------------------------------------
# Judge schema
# ---------------------------------------------------------

judge_schema = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string"
                    },

                    "model_a": {
                        "type": "object",
                        "properties": {
                            "conceptual_accuracy": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "argument_coherence": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "relational_reasoning": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "deweyan_style": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "textual_grounding": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "justification": {
                                "type": "object",
                                "properties": {
                                    "conceptual_accuracy": {
                                        "type": "string"
                                    },
                                    "argument_coherence": {
                                        "type": "string"
                                    },
                                    "relational_reasoning": {
                                        "type": "string"
                                    },
                                    "deweyan_style": {
                                        "type": "string"
                                    },
                                    "textual_grounding": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "conceptual_accuracy",
                                    "argument_coherence",
                                    "relational_reasoning",
                                    "deweyan_style",
                                    "textual_grounding"
                                ],
                                "additionalProperties": False
                            },
                            "strength": {
                                "type": "string"
                            },
                            "weakness": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "conceptual_accuracy",
                            "argument_coherence",
                            "relational_reasoning",
                            "deweyan_style",
                            "textual_grounding",
                            "justification",
                            "strength",
                            "weakness"
                        ],
                        "additionalProperties": False
                    },

                    "model_b": {
                        "type": "object",
                        "properties": {
                            "conceptual_accuracy": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "argument_coherence": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "relational_reasoning": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "deweyan_style": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "textual_grounding": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "justification": {
                                "type": "object",
                                "properties": {
                                    "conceptual_accuracy": {
                                        "type": "string"
                                    },
                                    "argument_coherence": {
                                        "type": "string"
                                    },
                                    "relational_reasoning": {
                                        "type": "string"
                                    },
                                    "deweyan_style": {
                                        "type": "string"
                                    },
                                    "textual_grounding": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "conceptual_accuracy",
                                    "argument_coherence",
                                    "relational_reasoning",
                                    "deweyan_style",
                                    "textual_grounding"
                                ],
                                "additionalProperties": False
                            },
                            "strength": {
                                "type": "string"
                            },
                            "weakness": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "conceptual_accuracy",
                            "argument_coherence",
                            "relational_reasoning",
                            "deweyan_style",
                            "textual_grounding",
                            "justification",
                            "strength",
                            "weakness"
                        ],
                        "additionalProperties": False
                    }
                },
                "required": [
                    "question_id",
                    "model_a",
                    "model_b"
                ],
                "additionalProperties": False
            }
        },

        "summary": {
            "type": "object",
            "properties": {
                "model_a": {
                    "type": "object",
                    "properties": {
                        "conceptual_accuracy_mean": {
                            "type": "number"
                        },
                        "argument_coherence_mean": {
                            "type": "number"
                        },
                        "relational_reasoning_mean": {
                            "type": "number"
                        },
                        "deweyan_style_mean": {
                            "type": "number"
                        },
                        "textual_grounding_mean": {
                            "type": "number"
                        }
                    },
                    "required": [
                        "conceptual_accuracy_mean",
                        "argument_coherence_mean",
                        "relational_reasoning_mean",
                        "deweyan_style_mean",
                        "textual_grounding_mean"
                    ],
                    "additionalProperties": False
                },

                "model_b": {
                    "type": "object",
                    "properties": {
                        "conceptual_accuracy_mean": {
                            "type": "number"
                        },
                        "argument_coherence_mean": {
                            "type": "number"
                        },
                        "relational_reasoning_mean": {
                            "type": "number"
                        },
                        "deweyan_style_mean": {
                            "type": "number"
                        },
                        "textual_grounding_mean": {
                            "type": "number"
                        }
                    },
                    "required": [
                        "conceptual_accuracy_mean",
                        "argument_coherence_mean",
                        "relational_reasoning_mean",
                        "deweyan_style_mean",
                        "textual_grounding_mean"
                    ],
                    "additionalProperties": False
                },

                "conceptual_questions": {
                    "type": "object"
                },

                "transfer_questions": {
                    "type": "object"
                },

                "wins": {
                    "type": "object",
                    "properties": {
                        "model_a": {
                            "type": "integer"
                        },
                        "model_b": {
                            "type": "integer"
                        },
                        "ties": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "model_a",
                        "model_b",
                        "ties"
                    ],
                    "additionalProperties": False
                },

                "overall_assessment": {
                    "type": "string"
                }
            },
            "required": [
                "model_a",
                "model_b",
                "conceptual_questions",
                "transfer_questions",
                "wins",
                "overall_assessment"
            ],
            "additionalProperties": False
        }
    },

    "required": [
        "evaluations",
        "summary"
    ],

    "additionalProperties": False
}


# ---------------------------------------------------------
# Construct evaluation prompt
# ---------------------------------------------------------

question_data = []

for q in questions:

    qid = q["id"]

    question_data.append({
        "id": qid,
        "type": q["type"],
        "question": q["question"],
        "rubric": q["rubric"],

        "model_a_answer":
            answers_a[qid]["answer"],

        "model_b_answer":
            answers_b[qid]["answer"]
    })


prompt = f"""
You are evaluating two LLMs that answered questions about
Chapter II of John Dewey's Art as Experience.

The complete evaluation dataset is below.

{json.dumps(question_data, ensure_ascii=False, indent=2)}

Evaluate every answer according to the rubric.

Remember:

- Chapter II was withheld from both models during fine-tuning.
- Do not reward Deweyan vocabulary by itself.
- "Deweyan style" means similarity in argumentative movement,
  conceptual integration, progressive development, and treatment
  of relationships among concepts.
- A response that merely lists terms such as "experience",
  "organism", "environment", or "consummation" should NOT receive
  a high relational-reasoning or Deweyan-style score.
- For transfer questions, evaluate whether the model genuinely
  applies the framework to the new situation.
- Evaluate Model A and Model B independently.
- Do not let one model's answer influence the score of the other.
- After the independent evaluations, compare the models.
- Higher scores are always better on every dimension.

Return ONLY the requested structured JSON.
"""


# ---------------------------------------------------------
# Call API
# ---------------------------------------------------------

response = client.responses.create(
    model=MODEL,

    input=[
        {
            "role": "system",
            "content": (
                "You are a rigorous evaluator of philosophical "
                "reasoning and writing. Follow the supplied rubric "
                "exactly."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ],

    text={
        "format": {
            "type": "json_schema",
            "name": "dewey_evaluation",
            "strict": True,
            "schema": judge_schema
        }
    }
)


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

result = json.loads(response.output_text)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )


print(f"Evaluation saved to: {OUTPUT_FILE}")