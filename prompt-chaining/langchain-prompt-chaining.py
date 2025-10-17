"""Quiz generation"""
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import os 
from datetime import datetime

load_dotenv()

class QuizOption(BaseModel):
    """Model for quiz options"""
    A: str = Field(description="Option A text")
    B: str = Field(description="Option B text")
    C: str = Field(description="Option C text")
    D: str = Field(description="Option D text")

class QuizQuestion(BaseModel):
    """Model for a single quiz question"""
    question_num: int = Field(description="Question Number")
    question: str = Field(description="Question text")
    options: QuizOption = Field(description="The four answer options")
    correct_answer: str = Field(description="The correct answer letter (A, B, C, or D)")
    explanation: str = Field(description="Explanation of why this is correct")
    learning_objective: str = Field(description="The learning objective this tests")

class QuizSet(BaseModel):
    """Model for complete quiz"""
    questions: List[QuizQuestion] = Field(description="List of quiz questions")

class QuizResult(BaseModel):
    """Model for quiz results"""
    question_num: int
    user_answer: str
    correct_answer: str
    is_correct: bool
    question: str
    explanation: str


# ==================== AZURE OPENAI CONFIGURATION ====================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_GPT_4O_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT_4O_DEPLOYMENT"),  # Your deployment name (e.g., "gpt-4o")
    api_version=os.getenv("AZURE_OPENAI_GPT_4O_API_VERSION", "2024-12-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_GPT_4O_API_KEY"),
    temperature=0.7,
)

quiz_topic = "Python Programming - Lists and Dictionaries"
difficulty_level = "intermediate"
num_questions = 5
student_name = "Alex"

quiz_prompt = ChatPromptTemplate.from_template("""
You are an expert educational content creator. Generate {num_questions} multiple-choice 
quiz questions on: {topic}

Difficulty Level: {difficulty}

Requirements:
- Each question should test understanding, not just memorization
- Options should be plausible but only one clearly correct
- Explanations should be educational and clear
- Cover different aspects of the topic
- Make questions progressively challenging

Generate exactly {num_questions} questions.
""")


quiz_generation_chain = (
    quiz_prompt 
    | llm.with_structured_output(QuizSet, method="function_calling")
)

print("⏳ Calling Azure OpenAI API...")
quiz_data = quiz_generation_chain.invoke({
    "num_questions": num_questions,
    "topic": quiz_topic,
    "difficulty": difficulty_level
})

print(f"✓ Generated {len(quiz_data.questions)} structured questions")
print(f"✓ Using Pydantic models for type safety")

# Display generated quiz
for i, q in enumerate(quiz_data.questions, 1):
    print(f"\n📝 Question {i}: {q.question}")
    print(f"   Learning Objective: {q.learning_objective}")

user_answer = []

simulated_responses = ["A", "B", "C", "A", "D"]

for i, question in enumerate(quiz_data.questions):
    print(f"\n📝 QUESTION {i+1}/{len(quiz_data.questions)}")
    print("-" * 100)
    print(f"\n{question.question}\n")
    
    print(f"  A) {question.options.A}")
    print(f"  B) {question.options.B}")
    print(f"  C) {question.options.C}")
    print(f"  D) {question.options.D}")
    
    # For real interaction: user_answer = input("\n🎯 Your Answer (A/B/C/D): ").strip().upper()
    user_answers = simulated_responses[i] if i < len(simulated_responses) else "A"
    print(f"\n🎯 Your Answer: {user_answer}")

    results = []
correct_count = 0

for i, (question, user_answer) in enumerate(zip(quiz_data.questions, user_answers)):
    is_correct = user_answer.upper() == question.correct_answer.upper()
    if is_correct:
        correct_count += 1
    
    results.append(QuizResult(
        question_num=i + 1,
        user_answer=user_answer,
        correct_answer=question.correct_answer,
        is_correct=is_correct,
        question=question.question,
        explanation=question.explanation
    ))

score_percentage = (correct_count / len(quiz_data.questions)) * 100

results_summary = "\n".join([
    f"Q{r.question_num}: {'✓ CORRECT' if r.is_correct else '✗ INCORRECT'} "
    f"(Student: {r.user_answer}, Correct: {r.correct_answer})"
    for r in results
])

incorrect_details = "\n".join([
    f"Q{r.question_num}: {r.question}\n  Correct: {r.correct_answer}\n  Explanation: {r.explanation}"
    for r in results if not r.is_correct
]) or "All answers correct!"

# Define feedback prompt
feedback_prompt = ChatPromptTemplate.from_template("""
You are an encouraging tutor. Provide personalized feedback:

TOPIC: {topic}
SCORE: {score}%
RESULTS: {results}

Provide:
1. Encouraging opening
2. Strengths identified
3. Areas needing improvement
4. Specific study tips
5. Motivational closing

Keep it concise but impactful.
""")

# Define remediation prompt
remediation_prompt = ChatPromptTemplate.from_template("""
Create a personalized learning plan:

TOPIC: {topic}
SCORE: {score}%
INCORRECT QUESTIONS: {incorrect}

Provide:
1. Performance assessment
2. Recommended difficulty adjustment
3. Topics to review
4. Study resources
5. Practice recommendations
6. Timeline for improvement

Be specific and actionable.
""")

# Define progress report prompt
report_prompt = ChatPromptTemplate.from_template("""
Create a concise progress report:

STUDENT: {student}
TOPIC: {topic}
DATE: {date}
SCORE: {score}%

Provide:
1. Executive summary
2. Skill level assessment
3. Next steps
4. Estimated study time needed

Format professionally.
""")

# Modern LCEL: Parallel chain execution using RunnableParallel with Azure OpenAI
parallel_chains = RunnableParallel(
    feedback=feedback_prompt | llm | StrOutputParser(),
    remediation=remediation_prompt | llm | StrOutputParser(),
    report=report_prompt | llm | StrOutputParser()
)

# Execute all chains in parallel
print("⚡ Running 3 Azure OpenAI chains in parallel...")

parallel_results = parallel_chains.invoke({
    "topic": quiz_topic,
    "score": score_percentage,
    "results": results_summary,
    "incorrect": incorrect_details,
    "student": student_name,
    "date": datetime.now().strftime('%B %d, %Y')
})

print("✓ Parallel execution complete!")

print("\n" + "=" * 100)
print("📊 DETAILED FEEDBACK:")
print("-" * 100)
print(parallel_results["feedback"])

print("\n" + "=" * 100)
print("🎯 REMEDIATION PLAN:")
print("-" * 100)
print(parallel_results["remediation"])

print("\n" + "=" * 100)
print("📈 PROGRESS REPORT:")
print("-" * 100)
print(parallel_results["report"])