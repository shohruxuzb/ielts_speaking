from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import tempfile
import uvicorn
import json
import os
import io
import random
import re

from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict

from api.routes_emotion import router as emotion_router
app.include_router(emotion_router)

# Init app
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

load_dotenv()



# Define allowed origins (use wildcard during local testing, tighten up in production)
origins = [
    "*",  # open to all origins for development; replace with specific domains in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 Set your Groq API Key in env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def transcribe_audio(audio_path: str) -> str:
    """Transcribe speech using Whisper on Groq"""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f
        )
    return transcript.text if transcript else ""


def evaluate_ielts_with_improvements(
        questions: List[str],
        answers: List[str],
        emotion_data: dict = None  # NEW parameter
) -> dict:
    """
    Evaluate IELTS answers across one or multiple questions,
    and provide improved versions of each answer.
    """
    qa_pairs = "\n".join(
        [f"{i + 1}. Q: \"{q}\" \n   A: \"{a}\"" for i, (q, a) in enumerate(zip(questions, answers))]
    )

    # Build emotion context block
    emotion_context = ""
    if emotion_data and "summary" in emotion_data:
        s = emotion_data["summary"]
        timeline = emotion_data.get("emotion_timeline", [])

        # Find peak nervous moments
        nervous_moments = [
            f"second {e['second']}"
            for e in timeline
            if e["emotion"] in ("fear", "sad") and e["confidence"] > 0.5
        ]
        nervous_str = ", ".join(nervous_moments[:5]) if nervous_moments else "none detected"

        emotion_context = f"""
Candidate Emotion Analysis (from facial recognition during the answer):
- Dominant emotion throughout: {s.get('dominant_emotion', 'unknown')}
- Confidence score: {s.get('confidence_score', 'N/A')}/100
- Nervous/anxious: {s.get('nervous_percentage', 0)}% of speaking time
- Happy/positive: {s.get('happy_percentage', 0)}% of speaking time
- Emotion breakdown: {s.get('emotion_breakdown', {})}
- Peak nervous moments at: {nervous_str}

Instructions for using emotion data:
- If nervous > 40%: acknowledge anxiety may have affected fluency, encourage candidate
- If confident > 70%: mention strong confident delivery as a positive
- Factor delivery confidence into your Fluency & Coherence score
- Add 1 emotion-specific tip in weaknesses or strengths accordingly
"""

    prompt = f"""
You are a certified IELTS examiner and English tutor. 
Evaluate the candidate's speaking performance strictly based on IELTS Band Descriptors.

Assessment Criteria:
- Fluency and Coherence
- Lexical Resource
- Grammatical Range and Accuracy
- Pronunciation

Tasks:
1. Give realistic sub-scores (0–9, steps of 0.5).
2. Give overall band score.
3. Mention strengths and weaknesses.
4. For each answer, provide an improved version (same meaning, but more natural and fluent).
{f"5. Use the emotion analysis below to enrich your feedback." if emotion_context else ""}

Candidate's responses:
{qa_pairs}
{emotion_context}

Return ONLY valid JSON with this structure:
{{
  "overall_band": float,
  "fluency": float,
  "vocabulary": float,
  "grammar": float,
  "pronunciation": float,
  "strengths": [list of strings],
  "weaknesses": [list of strings],
  "improved_answers": [list of improved answers, same length as input],
  "emotion_feedback": "one sentence tip based on candidate emotion during test"
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=900,  # slightly increased for emotion_feedback field
        )
        text_output = response.choices[0].message.content
        result_json = json.loads(text_output[text_output.index("{"): text_output.rindex("}") + 1])
        return result_json
    except Exception as e:
        return {"error": str(e)}
@app.post("/evaluate")
async def evaluate(
    questions: str = Form(...),   # JSON string of questions
    answers: str = Form(...),     # JSON string of answers (manual text or transcripts)
    audios: List[UploadFile] = File(None)  # Optional audio list
):
    """
    Evaluate IELTS speaking performance.
    - Supports Part 1 & 3 (3 Q/As) and Part 2 (1 Q/A).
    - Returns scores + improved versions of answers.
    """

    questions = json.loads(questions) if isinstance(questions, str) else questions
    answers = json.loads(answers) if isinstance(answers, str) else answers

    # If audio is uploaded, transcribe each and replace answers
    if audios:
        transcripts = []
        for audio in audios:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(await audio.read())
                tmp_path = tmp.name
            try:
                transcripts.append(transcribe_audio(tmp_path))
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        answers = transcripts

    # Safety check
    if not questions or not answers or len(questions) != len(answers):
        return {"error": "Questions and answers must be same length and non-empty"}

    evaluation = evaluate_ielts_with_improvements(questions, answers)
    return {"answers": answers, "evaluation": evaluation}

# Keep a pool of asked questions in memory
asked_questions = set()

def normalize_question(q: str) -> str:
    """Normalize a question to detect duplicates by meaning (basic)."""
    q = q.lower().strip()
    q = re.sub(
        r'^(can you|could you|do you think|describe|talk about|tell me about|would you rather|if you could)\s+',
        '',
        q,
    )
    q = re.sub(r'[?.!,]', '', q)  # remove punctuation
    return q.strip()


def _generate_unique_question(prompt: str) -> dict:
    """Generic helper that ensures uniqueness of generated questions."""
    try:
        for _ in range(5):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=1,
                presence_penalty=1.2,
                frequency_penalty=0.9,
                max_tokens=150,
            )

            question = response.choices[0].message.content.strip()
            normalized = normalize_question(question)

            if normalized not in asked_questions:
                asked_questions.add(normalized)

                # reset after 200 unique questions
                if len(asked_questions) > 200:
                    asked_questions.clear()
                    asked_questions.add(normalized)

                return {"question": question}

        return {"question": question, "note": "⚠️ Might be semantically similar"}
    except Exception as e:
        return {"error": str(e)}


# ----------- IELTS PART-SPECIFIC GENERATORS -----------

def generate_part1_questions() -> dict:
    """Generate 3 IELTS Part 1 questions (short, everyday)."""
    prompt = """
    Generate ONE IELTS Speaking Part 1 question.

    Requirements:
    - Short, direct, and about familiar everyday topics 
      (hometown, hobbies, food, studies, work, friends, family, daily routine, etc.)
    - Not too abstract.
    - Suitable for a 20–30 second answer.
    - Return ONLY the question text.
    """
    return {"questions": [_generate_unique_question(prompt)["question"] for _ in range(3)]}


def generate_part2_question() -> dict:
    """Generate 1 IELTS Part 2 cue card question."""
    prompt = """
    Generate ONE IELTS Speaking Part 2 "Cue Card" style question.

    Requirements:
    - Start with "Describe ..." or "Talk about ..."
    - Include 3–4 bullet points (using dashes) that guide the candidate.
    - Topic should be about people, experiences, places, or objects.
    - Suitable for a 1–2 minute long answer.
    - Return ONLY the question text with bullet points.
    """
    return _generate_unique_question(prompt)


def generate_part3_questions() -> dict:
    """Generate 3 IELTS Part 3 abstract, discussion questions."""
    prompt = """
    Generate ONE IELTS Speaking Part 3 discussion question.

    Requirements:
    - Abstract, opinion-based, and analytical.
    - Related to society, culture, education, technology, future, or global issues.
    - Suitable for a 30–40 second thoughtful answer.
    - Return ONLY the question text.
    """
    return {"questions": [_generate_unique_question(prompt)["question"] for _ in range(3)]}


# ----------- API ROUTES -----------

@app.get("/generate-part1")
async def generate_part1():
    return generate_part1_questions()

@app.get("/generate-part2")
async def generate_part2():
    return generate_part2_question()

@app.get("/generate-part3")
async def generate_part3():
    return generate_part3_questions()


def aggregate_evaluations(evaluations: list[dict]) -> dict:
    """
    Combine multiple evaluations (from Part 1, 2, 3) into one overall IELTS result.
    Weighted scoring:
        Part 1 -> 25%
        Part 2 -> 40%
        Part 3 -> 35%
    """
    if not evaluations or len(evaluations) != 3:
        return {"error": "Expected exactly 3 evaluations (Part 1, Part 2, Part 3)"}

    weights = [0.25, 0.40, 0.35]  # Part 1, Part 2, Part 3
    criteria = ["fluency", "vocabulary", "grammar", "pronunciation", "overall_band"]

    weighted_scores = {c: 0.0 for c in criteria}
    strengths, weaknesses = [], []

    for i, ev in enumerate(evaluations):
        if "error" in ev:
            continue

        try:
            for c in criteria:
                weighted_scores[c] += float(ev[c]) * weights[i]

            strengths.extend(ev.get("strengths", []))
            weaknesses.extend(ev.get("weaknesses", []))
        except Exception:
            continue

    overall_result = {
        "overall_band": round(weighted_scores["overall_band"], 1),
        "fluency": round(weighted_scores["fluency"], 1),
        "vocabulary": round(weighted_scores["vocabulary"], 1),
        "grammar": round(weighted_scores["grammar"], 1),
        "pronunciation": round(weighted_scores["pronunciation"], 1),
        "strengths": list(set(strengths)),   # remove duplicates
        "weaknesses": list(set(weaknesses))  # remove duplicates
    }

    return overall_result


class EvaluationsRequest(BaseModel):
    evaluations: List[Dict]

@app.post("/aggregate-results")
async def aggregate_results(req: EvaluationsRequest):
    """
    Takes a list of JSON evaluations for Part 1, Part 2, and Part 3
    and returns a weighted overall IELTS band report.
    """
    return aggregate_evaluations(req.evaluations)


def synthesize_speech(text: str) -> bytes:
    """
    Convert text into speech (mp3) using Groq TTS.
    """
    try:
        response = client.audio.speech.create(
            model="whisper-tts",   # or "whisper-tts" depending on Groq support
            voice="alloy",            # choose available voice
            input=text,
        )
        return response.audio  # raw audio bytes
    except Exception as e:
        print("TTS error:", e)
        return b""


@app.post("/mock-test-voice")
async def mock_test_voice(part1: list[str], part2: str, part3: list[str]):
    """
    Generate a full mock IELTS test audio:
    - Greeting
    - Part 1 questions
    - Part 2 cue card
    - Part 3 questions
    - Goodbye
    """

    script = []

    # Greeting
    script.append("Hello, welcome to your IELTS speaking mock test. Let's begin.")

    # Part 1
    script.append("Part one. I will ask you some questions about yourself and everyday topics.")
    for q in part1:
        script.append(q)

    # Part 2
    script.append("Now let's move on to part two. You will have a cue card.")
    script.append(part2)

    # Part 3
    script.append("Now we continue with part three. I will ask you some more discussion questions.")
    for q in part3:
        script.append(q)

    # Goodbye
    script.append("That concludes the IELTS speaking mock test. Thank you, and goodbye.")

    # Concatenate into one long script
    full_text = " ".join(script)

    # Convert to speech
    audio_bytes = synthesize_speech(full_text)

    if not audio_bytes:
        return {"error": "TTS generation failed"}

    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)