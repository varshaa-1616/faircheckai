from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


def explain_bias(metrics_text: str) -> str:
    """Uses Gemini to explain fairness metrics in simple language."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""You are an AI fairness expert.

Explain the following bias metrics in clear, simple language for a non-technical audience.

Metrics:
{metrics_text}

Structure your response as:
1. **Simple Explanation** – What do these numbers actually mean?
2. **Possible Causes of Bias** – Why might this bias exist?
3. **Suggestions to Reduce Bias** – Concrete steps to improve fairness.

Keep it concise, actionable, and plain-English.""",
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"


def generate_fairness_report(summary: str) -> str:
    """Generate a structured fairness audit report."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""Generate a professional AI Fairness Audit Report based on the following dataset summary.

Dataset Summary:
{summary}

Structure the report as follows:

# AI Fairness Audit Report

## 1. Executive Summary
## 2. Fairness Summary
## 3. Key Bias Risks Identified
## 4. Possible Root Causes
## 5. Mitigation Strategies
## 6. Deployment Recommendation

Be specific, professional, and actionable.""",
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"


def suggest_mitigation(metrics_text: str, context: str = "") -> str:
    """Suggest targeted mitigation strategies for detected bias."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""You are an AI ethics and fairness expert.

Given these fairness metrics:
{metrics_text}

{"Additional context: " + context if context else ""}

Provide 3-5 specific, actionable mitigation strategies. For each strategy:
- Name the technique
- Explain how it works in 1-2 sentences
- State when to use it

Focus on practical steps a data scientist or ML engineer can implement.""",
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"