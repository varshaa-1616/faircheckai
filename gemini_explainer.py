from google import genai
import os
import time
import json
import hashlib
import re
from dotenv import load_dotenv
from typing import Optional, Dict
import streamlit as st

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# Cache file for storing API responses
CACHE_FILE = "gemini_cache.json"

# Quota tracking
class QuotaManager:
    def __init__(self):
        self.request_count = 0
        self.last_reset = time.time()
        self.daily_count = 0
        self.last_daily_reset = time.time()
    
    def can_make_request(self) -> bool:
        # Reset minute counter
        if time.time() - self.last_reset > 60:
            self.request_count = 0
            self.last_reset = time.time()
        
        # Reset daily counter (every 24 hours)
        if time.time() - self.last_daily_reset > 86400:
            self.daily_count = 0
            self.last_daily_reset = time.time()
        
        # Free tier limits (conservative estimates)
        if self.request_count >= 50:
            return False
        if self.daily_count >= 1400:
            return False
        
        return True
    
    def record_request(self):
        self.request_count += 1
        self.daily_count += 1
    
    def get_remaining_estimate(self) -> int:
        return max(0, 1400 - self.daily_count)

quota_manager = QuotaManager()

def load_cache() -> Dict:
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache: Dict):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass

def get_cache_key(prefix: str, text: str) -> str:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"{prefix}_{text_hash}"

def get_cached_or_generate(cache_key: str, generate_func) -> Optional[str]:
    cache = load_cache()
    
    if cache_key in cache:
        return cache[cache_key]
    
    if not quota_manager.can_make_request():
        return None
    
    try:
        result = generate_func()
        if result and not result.startswith("Error"):
            cache[cache_key] = result
            save_cache(cache)
            quota_manager.record_request()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def generate_fallback_explanation(metrics_text: str, context: str = "") -> str:
    """Generate basic explanation without API call - FIXED VERSION"""
    
    # Extract metrics properly
    bias_risk = "UNKNOWN"
    demographic_parity = None
    equalized_odds = None
    accuracy = None
    
    # Look for bias risk level in the text
    risk_match = re.search(r'Bias risk level:\s*(\w+)', metrics_text, re.IGNORECASE)
    if risk_match:
        bias_risk = risk_match.group(1).upper()
    
    # Look for demographic parity difference
    dp_match = re.search(r'Demographic Parity Difference:\s*([\d\.]+)', metrics_text, re.IGNORECASE)
    if dp_match:
        demographic_parity = float(dp_match.group(1))
    
    # Look for equalized odds difference
    eo_match = re.search(r'Equalized Odds Difference:\s*([\d\.]+)', metrics_text, re.IGNORECASE)
    if equalized_odds:
        equalized_odds = float(eo_match.group(1))
    
    # Look for accuracy
    acc_match = re.search(r'Accuracy:\s*([\d\.]+)%', metrics_text, re.IGNORECASE)
    if acc_match:
        accuracy = float(acc_match.group(1))
    
    explanation_parts = []
    
    # Determine bias level from the actual risk level or metrics
    if bias_risk == "HIGH":
        explanation_parts.append(f"🔴 **HIGH BIAS DETECTED**")
        explanation_parts.append("**Critical Finding:** Your model shows severe fairness violations across protected groups.")
        
        if demographic_parity is not None:
            explanation_parts.append(f"\n**Demographic Parity Difference: {demographic_parity:.4f}**")
            explanation_parts.append("This measures if different groups receive positive outcomes at different rates.")
            explanation_parts.append(f"A value of {demographic_parity:.4f} indicates significant disparity - the ideal is 0.0.")
        
        if equalized_odds is not None:
            explanation_parts.append(f"\n**Equalized Odds Difference: {equalized_odds:.4f}**")
            explanation_parts.append("This measures if error rates (false positives/negatives) differ across groups.")
            explanation_parts.append(f"A value of {equalized_odds:.4f} shows major differences in how the model performs for different groups.")
        
        if accuracy is not None:
            explanation_parts.append(f"\n**Overall Accuracy: {accuracy:.2f}%**")
            explanation_parts.append("While overall accuracy looks good, it can hide serious fairness issues affecting specific groups.")
        
        explanation_parts.append("\n**Impact:** This model could cause systematic discrimination. DO NOT DEPLOY without fixing these issues.")
        
    elif bias_risk == "MEDIUM":
        explanation_parts.append(f"🟡 **MODERATE BIAS DETECTED**")
        explanation_parts.append("Your model shows meaningful fairness concerns that need addressing.")
        
        if demographic_parity is not None:
            explanation_parts.append(f"\n**Demographic Parity Difference: {demographic_parity:.4f}**")
        if equalized_odds is not None:
            explanation_parts.append(f"**Equalized Odds Difference: {equalized_odds:.4f}**")
            
        explanation_parts.append("\n**Impact:** While not severe, this bias could harm affected groups over time.")
        
    elif bias_risk == "LOW":
        explanation_parts.append(f"🟢 **LOW BIAS DETECTED**")
        explanation_parts.append("Your model appears relatively fair across different groups.")
        explanation_parts.append("\n**Impact:** Minimal fairness concerns detected.")
    
    else:
        # If no risk level, calculate from demographic parity
        if demographic_parity is not None:
            if demographic_parity > 0.2:
                explanation_parts.append(f"🔴 **HIGH BIAS DETECTED (Demographic Parity: {demographic_parity:.4f})**")
            elif demographic_parity > 0.1:
                explanation_parts.append(f"🟡 **MODERATE BIAS DETECTED (Demographic Parity: {demographic_parity:.4f})**")
            else:
                explanation_parts.append(f"🟢 **LOW BIAS DETECTED (Demographic Parity: {demographic_parity:.4f})**")
    
    # Add possible causes
    explanation_parts.append("\n## Possible Causes of Bias")
    explanation_parts.append("- Model may be relying on proxy features correlated with sensitive attributes")
    explanation_parts.append("- Different error rates across groups")
    explanation_parts.append("- Training data imbalances or historical bias")
    explanation_parts.append("- Model may be using sensitive attributes directly or indirectly")
    
    # Add suggestions based on risk level
    explanation_parts.append("\n## Suggestions to Reduce Bias")
    
    if bias_risk == "HIGH":
        explanation_parts.append("**URGENT ACTIONS REQUIRED:**")
        explanation_parts.append("1. **Re-evaluate model features** - Remove proxy variables correlated with protected attributes")
        explanation_parts.append("2. **Use fairness constraints** during model training (in-processing)")
        explanation_parts.append("3. **Apply post-processing** to adjust predictions for equalized odds")
        explanation_parts.append("4. **Collect more diverse training data** for underrepresented groups")
        explanation_parts.append("5. **Consider different model architectures** designed for fairness")
    else:
        explanation_parts.append("1. **Use fairness-aware algorithms** (pre-processing, in-processing, or post-processing)")
        explanation_parts.append("2. **Adjust decision thresholds** per group to equalize outcomes")
        explanation_parts.append("3. **Remove or de-correlate sensitive attributes** from model features")
        explanation_parts.append("4. **Implement regular fairness monitoring** in production")
    
    if context:
        explanation_parts.append(f"\n*Note: {context}*")
    
    # Add note about API
    if not api_key:
        explanation_parts.append("\n---\n*⚠️ Gemini API not connected. Using basic analysis. Connect API key for detailed insights.*")
    elif not quota_manager.can_make_request():
        explanation_parts.append("\n---\n*⚠️ API quota temporarily exceeded. Using basic analysis. Try again in a minute.*")
    
    return "\n".join(explanation_parts)

def explain_bias(metrics_text: str) -> str:
    """Explain bias with retry logic and fallback"""
    
    cache_key = get_cache_key("explain", metrics_text)
    
    def generate():
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"""You are an AI fairness expert.

Analyze these fairness metrics CAREFULLY:

{metrics_text}

IMPORTANT: Look at the actual numbers. If Demographic Parity Difference or Equalized Odds Difference is close to 1.0 (like 0.8-1.0), that indicates EXTREME BIAS, not low bias.

Explain the bias metrics in clear, simple language for a non-technical audience.

Structure your response as:
1. **Simple Explanation** – What do these numbers actually mean? Be honest about severity.
2. **Possible Causes of Bias** – Why might this bias exist?
3. **Suggestions to Reduce Bias** – Concrete steps to improve fairness.

Be honest and direct about the level of bias detected."""
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                raise e
            return f"Error: {str(e)}"
    
    result = get_cached_or_generate(cache_key, generate)
    
    if result is None or (result and result.startswith("Error")):
        return generate_fallback_explanation(metrics_text)
    
    return result if result else generate_fallback_explanation(metrics_text)

def generate_fairness_report(summary: str) -> str:
    """Generate a structured fairness audit report"""
    
    cache_key = get_cache_key("report", summary)
    
    def generate():
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"""Generate a professional AI Fairness Audit Report based on the following dataset/model summary.

Summary:
{summary}

IMPORTANT: If Demographic Parity Difference or Equalized Odds Difference is near 1.0, this indicates EXTREME BIAS. Be direct about this.

Structure the report as follows:

# AI Fairness Audit Report

## 1. Executive Summary
## 2. Fairness Metrics Analysis (be specific about actual numbers)
## 3. Key Bias Risks Identified
## 4. Possible Root Causes
## 5. Mitigation Strategies
## 6. Deployment Recommendation (DEPLOY / DEPLOY WITH CAUTION / DO NOT DEPLOY)

Be specific, professional, and actionable."""
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                raise e
            return f"Error: {str(e)}"
    
    result = get_cached_or_generate(cache_key, generate)
    
    if result is None or (result and result.startswith("Error")):
        return f"""# AI Fairness Audit Report

## 1. Executive Summary
Based on the provided metrics, **SEVERE BIAS** has been detected.

## 2. Fairness Metrics Analysis
{generate_fallback_explanation(summary, "Report generation")}

## 3. Key Bias Risks Identified
- Demographic Parity Difference of 1.0000 indicates completely different outcome rates across groups
- Equalized Odds Difference of 1.0000 shows severe disparities in error rates
- **CRITICAL:** Model treats different groups completely differently

## 4. Mitigation Strategies
DO NOT DEPLOY this model in its current state. Urgent mitigation required:
- Re-train with fairness constraints
- Remove proxy variables
- Consider alternative model architectures

## 5. Deployment Recommendation
**DO NOT DEPLOY** - Model shows extreme bias that would cause active discrimination.

---
*This is an urgent fairness alert based on your metrics.*"""
    
    return result

def suggest_mitigation(metrics_text: str, context: str = "") -> str:
    """Suggest targeted mitigation strategies for detected bias"""
    
    combined_text = f"{metrics_text}\nContext: {context}"
    cache_key = get_cache_key("mitigate", combined_text)
    
    def generate():
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"""You are an AI ethics and fairness expert.

Given these fairness metrics:
{metrics_text}

{"Additional context: " + context if context else ""}

IMPORTANT: If metrics show extreme bias (values near 1.0), provide URGENT mitigation strategies.

Provide 3-5 specific, actionable mitigation strategies. For each strategy:
- Name the technique
- Explain how it works in 1-2 sentences
- When to use it

Focus on practical steps a data scientist or ML engineer can implement."""
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                raise e
            return f"Error: {str(e)}"
    
    result = get_cached_or_generate(cache_key, generate)
    
    if result is None or (result and result.startswith("Error")):
        return generate_fallback_explanation(metrics_text, context)
    
    return result