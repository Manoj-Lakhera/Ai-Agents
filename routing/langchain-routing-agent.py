"""
Programming Language Router - Intelligent Code Assistant
Routes programming questions to specialized language experts

Features:
- Automatic language detection
- Route to specialized agents (Python, JavaScript, SQL, etc.)
- Semantic routing based on question content
- Azure OpenAI powered
- Modern LangChain LCEL
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Dict
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ==================== PYDANTIC MODELS ====================
class RouteDecision(BaseModel):
    """Model for routing decision"""
    language: Literal["python", "javascript", "sql", "devops", "general"] = Field(
        description="The programming language or domain to route to"
    )
    reasoning: str = Field(description="Explanation of why this route was chosen")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)

# ==================== AZURE OPENAI CONFIGURATION ====================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_GPT_4O_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT_4O_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_GPT_4O_API_VERSION", "2024-12-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_GPT_4O_API_KEY"),
    temperature=0.3,  # Lower for more consistent routing
)

print("=" * 100)
print("🔀 PROGRAMMING LANGUAGE ROUTER - INTELLIGENT CODE ASSISTANT 🔀")
print("=" * 100)
print(f"📅 Date: {datetime.now().strftime('%B %d, %Y at %H:%M %p')}")
print(f"☁️  Using: Azure OpenAI (GPT-4o)")
print(f"🎯 Pattern: Routing Agent")
print("=" * 100)

# ==================== DEFINE SPECIALIZED AGENTS ====================
python_expert = """
You are a Python programming expert specializing in:
- Python syntax and best practices
- Data structures (lists, dicts, sets, tuples)
- Object-oriented programming
- Popular libraries (pandas, numpy, requests)
- Pythonic code patterns
- Debugging Python code

Provide clear, working Python code examples with explanations.
"""

javascript_expert = """
You are a JavaScript/TypeScript expert specializing in:
- Modern JavaScript (ES6+)
- Node.js and npm ecosystem
- React, Vue, Angular frameworks
- Async programming (Promises, async/await)
- DOM manipulation
- TypeScript types and interfaces

Provide clean, modern JavaScript code with best practices.
"""

sql_expert = """
You are a SQL database expert specializing in:
- SQL queries (SELECT, JOIN, subqueries)
- Database design and normalization
- Query optimization
- PostgreSQL, MySQL, SQL Server
- Indexes and performance tuning
- Stored procedures and functions

Provide optimized SQL queries with explanations.
"""

devops_expert = """
You are a DevOps and Cloud expert specializing in:
- Docker and containerization
- Kubernetes orchestration
- CI/CD pipelines
- Cloud platforms (Azure, AWS, GCP)
- Infrastructure as Code (Terraform, ARM)
- Shell scripting (bash, PowerShell)

Provide practical DevOps solutions and configurations.
"""

general_expert = """
You are a general programming consultant specializing in:
- Algorithm design and data structures
- Software architecture and design patterns
- Code review and best practices
- Multiple programming paradigms
- System design
- Problem-solving strategies

Provide language-agnostic advice and guidance.
"""

# Create expert map
EXPERTS: Dict[str, str] = {
    "python": python_expert,
    "javascript": javascript_expert,
    "sql": sql_expert,
    "devops": devops_expert,
    "general": general_expert,
}

# ==================== ROUTING CHAIN ====================
routing_prompt = ChatPromptTemplate.from_template("""
You are an intelligent routing system for programming questions.
Analyze the user's question and determine which specialized expert should handle it.

Available Experts:
- python: Python programming (syntax, libraries, debugging)
- javascript: JavaScript/TypeScript (Node.js, React, async)
- sql: Database queries and optimization
- devops: Docker, Kubernetes, CI/CD, Cloud
- general: Algorithm design, architecture, multi-language advice

User Question: {question}

Analyze the question and decide which expert is best suited.
Consider keywords, context, and specific technologies mentioned.
Return your routing decision with confidence score (0-1).
""")

routing_chain = (
    routing_prompt 
    | llm.with_structured_output(RouteDecision, method="function_calling")
)

# ==================== EXPERT ANSWER CHAIN ====================
def create_expert_chain(expert_system_prompt: str):
    """Create a chain for a specific expert"""
    expert_prompt = ChatPromptTemplate.from_messages([
        ("system", expert_system_prompt),
        ("human", "{question}")
    ])
    return expert_prompt | llm | StrOutputParser()

# ==================== TEST QUESTIONS ====================
test_questions = [
    "How do I create a list comprehension in Python that filters even numbers?",
    "What's the difference between let, const, and var in JavaScript?",
    "How can I optimize this SQL query with multiple JOINs?",
    "How do I create a Docker container for my Node.js app?",
    "What's the time complexity of quicksort vs mergesort?",
    "How do I use async/await in JavaScript?",
    "Write a Python function to reverse a string",
    "How do I set up a CI/CD pipeline in Azure DevOps?",
]

# ==================== PROCESS QUESTIONS ====================
print("\n" + "=" * 100)
print("🚀 ROUTING QUESTIONS TO SPECIALIZED EXPERTS")
print("=" * 100)

for idx, question in enumerate(test_questions, 1):
    print(f"\n{'=' * 100}")
    print(f"❓ QUESTION {idx}/{len(test_questions)}")
    print(f"{'=' * 100}")
    print(f"\n📝 User Question:\n{question}")
    
    # Step 1: Route the question
    print(f"\n🔀 Step 1: Routing Decision...")
    print("-" * 100)
    
    route_decision = routing_chain.invoke({"question": question})
    
    print(f"✓ Routed to: {route_decision.language.upper()} Expert")
    print(f"✓ Confidence: {route_decision.confidence * 100:.1f}%")
    print(f"✓ Reasoning: {route_decision.reasoning}")
    
    # Step 2: Get expert answer
    print(f"\n💡 Step 2: Getting Answer from {route_decision.language.upper()} Expert...")
    print("-" * 100)
    
    expert_system_prompt = EXPERTS[route_decision.language]
    expert_chain = create_expert_chain(expert_system_prompt)
    
    expert_answer = expert_chain.invoke({"question": question})
    
    print(f"\n🤖 {route_decision.language.upper()} Expert Response:")
    print(expert_answer)
    
    print(f"\n{'=' * 100}")

# ==================== INTERACTIVE MODE ====================
print("\n" + "=" * 100)
print("💬 INTERACTIVE MODE - ASK YOUR OWN QUESTIONS")
print("=" * 100)
print("Type 'exit' to quit\n")

while True:
    user_question = input("❓ Your Question: ").strip()
    
    if user_question.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Thanks for using the Programming Language Router!")
        break
    
    if not user_question:
        print("⚠️  Please enter a question.")
        continue
    
    print("\n🔀 Routing your question...")
    
    # Route the question
    route_decision = routing_chain.invoke({"question": user_question})
    
    print(f"\n✓ Routed to: {route_decision.language.upper()} Expert")
    print(f"✓ Confidence: {route_decision.confidence * 100:.1f}%")
    print(f"✓ Reasoning: {route_decision.reasoning}")
    
    # Get expert answer
    print(f"\n💡 {route_decision.language.upper()} Expert is thinking...")
    
    expert_system_prompt = EXPERTS[route_decision.language]
    expert_chain = create_expert_chain(expert_system_prompt)
    expert_answer = expert_chain.invoke({"question": user_question})
    
    print(f"\n🤖 Answer:\n")
    print(expert_answer)
    print("\n" + "-" * 100 + "\n")

# ==================== STATS & SUMMARY ====================
print("\n" + "=" * 100)
print("📊 KEY FEATURES DEMONSTRATED")
print("=" * 100)
print("""
✅ Intelligent Routing:
   - Automatic language/domain detection
   - Confidence scoring
   - Reasoning transparency

✅ Specialized Experts:
   - Python Expert (syntax, libraries, Pythonic code)
   - JavaScript Expert (ES6+, React, async)
   - SQL Expert (queries, optimization, design)
   - DevOps Expert (Docker, K8s, CI/CD)
   - General Expert (algorithms, architecture)

✅ Modern LangChain Features:
   - LCEL chain composition
   - Structured output (RouteDecision)
   - Function calling for reliable routing
   - Reusable expert chains

✅ Azure OpenAI Integration:
   - GPT-4o for intelligent routing
   - Low temperature for consistent decisions
   - Enterprise-grade security

✅ Production-Ready Patterns:
   - Clear routing logic
   - Confidence scores
   - Extensible (easy to add new experts)
   - Interactive mode for testing
""")
print("=" * 100)

print("\n🎯 ROUTING PATTERNS USED:")
print("=" * 100)
print("""
1. SEMANTIC ROUTING (Most Common):
   - Analyzes question content and context
   - Routes based on meaning, not just keywords
   - Example: "list comprehension" → Python Expert

2. KEYWORD-BASED ROUTING:
   - Detects specific technologies/keywords
   - Example: "Docker" → DevOps Expert
   - Example: "SELECT * FROM" → SQL Expert

3. CONFIDENCE-BASED ROUTING:
   - High confidence (>0.8) → Direct routing
   - Medium confidence (0.5-0.8) → Route with caution
   - Low confidence (<0.5) → Route to General Expert

4. MULTI-EXPERT ROUTING (Future Enhancement):
   - Route to multiple experts for complex questions
   - Combine answers from multiple specialists
   - Example: "Deploy Python app to Azure" → Python + DevOps
""")
print("=" * 100)

print("\n🔧 EXTENDING THIS ROUTER:")
print("=" * 100)
print("""
To add new experts:

1. Add to RouteDecision Literal:
   Literal["python", "javascript", "sql", "devops", "java", "rust"]

2. Define expert system prompt:
   java_expert = '''You are a Java expert...'''

3. Add to EXPERTS dictionary:
   EXPERTS["java"] = java_expert

4. Update routing prompt description:
   - java: Java programming, Spring Boot, Maven

That's it! The router will automatically handle the new expert.
""")
print("=" * 100)

print("\n🚀 ROUTING AGENT COMPLETE! 🎓\n")