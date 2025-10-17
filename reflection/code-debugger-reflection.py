"""
AI Code Debugger with Self-Reflection
Student-friendly tool that finds bugs, reflects on its analysis, and teaches debugging

Reflection Pattern:
1. Analyze Code → Find potential bugs
2. Reflect → "Am I sure about these bugs?"
3. Verify → Test fixes mentally
4. Explain → Teach student what went wrong

Perfect for: CS students learning to debug
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== AZURE OPENAI CONFIGURATION ====================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_GPT_4O_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT_4O_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_GPT_4O_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_GPT_4O_API_KEY"),
    temperature=0.3,  # Lower for more consistent debugging
)

print("=" * 100)
print("🐛 AI CODE DEBUGGER WITH SELF-REFLECTION 🐛")
print("=" * 100)
print("Student-friendly debugging assistant that teaches while it fixes!\n")

# ==================== SAMPLE BUGGY CODE ====================
buggy_code = """
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    average = total / len(numbers)
    return average

# Test
scores = [85, 90, 78, 92, 88]
print("Average score:", calculate_average(scores))

# Edge case - empty list
empty_scores = []
print("Empty average:", calculate_average(empty_scores))
"""

print("📝 Buggy Code to Debug:")
print("-" * 100)
print(buggy_code)
print("-" * 100)

# ==================== STEP 1: INITIAL BUG ANALYSIS ====================
print("\n🔍 STEP 1: Initial Bug Analysis")
print("=" * 100)

initial_analysis_prompt = ChatPromptTemplate.from_template("""
You are a code debugger. Analyze this code and identify all bugs and potential issues.

Code:
{code}

Identify:
1. Syntax errors
2. Logic errors
3. Runtime errors (edge cases)
4. Code smells or inefficiencies

For each bug, provide:
- Line number (if applicable)
- Bug description
- Why it's a problem
- Severity (critical, major, minor)

Be thorough but concise.
""")

initial_chain = initial_analysis_prompt | llm | StrOutputParser()

initial_analysis = initial_chain.invoke({"code": buggy_code})

print(initial_analysis)

# ==================== STEP 2: SELF-REFLECTION ====================
print("\n🤔 STEP 2: Self-Reflection & Verification")
print("=" * 100)

reflection_prompt = ChatPromptTemplate.from_template("""
You are a senior code reviewer checking another debugger's analysis.

ORIGINAL CODE:
{code}

INITIAL BUG ANALYSIS:
{initial_analysis}

Reflect critically on this analysis:
1. Are all bugs correctly identified?
2. Are there any false positives (things marked as bugs that aren't)?
3. Did we miss any bugs?
4. Are the severity ratings accurate?
5. Could the explanations be clearer?

Provide:
- What the initial analysis got RIGHT
- What it got WRONG or missed
- Confidence level (1-10) in the final analysis
- Any additional insights

Be honest and critical - it's okay to find mistakes!
""")

reflection_chain = reflection_prompt | llm | StrOutputParser()

reflection_result = reflection_chain.invoke({
    "code": buggy_code,
    "initial_analysis": initial_analysis
})

print(reflection_result)

# ==================== STEP 3: FINAL BUG REPORT ====================
print("\n✅ STEP 3: Final Bug Report (After Reflection)")
print("=" * 100)

final_report_prompt = ChatPromptTemplate.from_template("""
Based on the initial analysis and self-reflection, create a final bug report.

ORIGINAL CODE:
{code}

INITIAL ANALYSIS:
{initial_analysis}

REFLECTION:
{reflection}

Create a clear, student-friendly bug report with:
1. List of confirmed bugs (prioritized by severity)
2. Detailed explanation of each bug
3. Example of what happens when the bug occurs
4. How to fix each bug (but don't give full solution)
5. Learning tips to avoid this bug in the future

Format for readability. Use emojis to highlight severity:
🔴 Critical
🟡 Major
🟢 Minor
""")

final_chain = final_report_prompt | llm | StrOutputParser()

final_report = final_chain.invoke({
    "code": buggy_code,
    "initial_analysis": initial_analysis,
    "reflection": reflection_result
})

print(final_report)

# ==================== STEP 4: GENERATE FIXED CODE ====================
print("\n🔧 STEP 4: Proposed Fix (With Explanation)")
print("=" * 100)

fix_prompt = ChatPromptTemplate.from_template("""
Generate the corrected version of this code based on our bug analysis.

ORIGINAL CODE:
{code}

BUG REPORT:
{final_report}

Provide:
1. Fixed code (fully working)
2. Inline comments explaining each fix
3. Test cases to verify the fix works
4. Brief explanation of key changes

Make it educational - explain WHY each change was made.
""")

fix_chain = fix_prompt | llm | StrOutputParser()

fixed_code = fix_chain.invoke({
    "code": buggy_code,
    "final_report": final_report
})

print(fixed_code)