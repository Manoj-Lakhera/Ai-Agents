"""
YouTube Chapter Marker Creator - Parallel Processing
Analyzes video transcript and generates chapter markers using parallel agents

Parallel Processing:
1. Topic Segmentation Agent → Identifies major topic changes
2. Keyword Extraction Agent → Finds key terms per segment
3. Engagement Analysis Agent → Identifies high-value moments
4. Summary Agent → Creates chapter titles

Features:
- Parallel processing for faster results
- Multiple analysis dimensions
- YouTube-optimized chapter format
- Timestamp generation
- Azure OpenAI powered
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from typing import List, Dict
import os
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()

# ==================== PYDANTIC MODELS ====================
class TimeSegment(BaseModel):
    """Model for a time segment"""
    start_time: str = Field(description="Start timestamp in MM:SS or HH:MM:SS format")
    end_time: str = Field(description="End timestamp in MM:SS or HH:MM:SS format")
    content: str = Field(description="Content in this time segment")

class Chapter(BaseModel):
    """Model for a chapter"""
    timestamp: str = Field(description="Chapter timestamp in MM:SS or HH:MM:SS format")
    title: str = Field(description="Chapter title")
    description: str = Field(description="Brief chapter description")
    keywords: List[str] = Field(description="Key topics covered")

# ==================== AZURE OPENAI CONFIGURATION ====================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_GPT_4O_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT_4O_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_GPT_4O_API_VERSION", "2024-12-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_GPT_4O_API_KEY"),
    temperature=0.7,
)

# ==================== SAMPLE TRANSCRIPT ====================
# This is a sample transcript with timestamps
# In production, you'd read from .srt, .vtt, or .txt files
sample_transcript = """
[00:00] Hey everyone, welcome back to the channel! Today we're going to talk about building AI agents with LangChain.
[00:15] Before we dive in, make sure to like and subscribe. We'll be covering a lot of ground today.
[00:30] First, let's understand what AI agents are. An AI agent is an autonomous system that can perceive its environment, make decisions, and take actions to achieve specific goals.
[01:00] There are several types of agents. We have reactive agents, deliberative agents, and hybrid agents. Each has its own use case.
[01:45] Now, let's talk about LangChain. LangChain is a framework for building applications powered by large language models.
[02:15] The key concept in LangChain is chains. Chains allow you to link multiple components together to create complex workflows.
[02:45] Let me show you a practical example. We'll build a simple question-answering agent that can access external tools.
[03:30] First, we need to install the necessary packages. Open your terminal and run pip install langchain langchain-openai.
[04:00] Now let's write some code. We'll start by importing the required modules and setting up our API keys.
[04:45] Here's how you create a simple chain. We define a prompt template, connect it to the LLM, and add an output parser.
[05:30] Next, let's add some tools. Tools allow our agent to interact with external systems like databases, APIs, or search engines.
[06:15] We'll create a tool for web search. This will let our agent search the internet for real-time information.
[07:00] Now we combine everything into an agent. The agent will decide when to use tools based on the user's question.
[07:45] Let's test it out. I'll ask it a question that requires web search.
[08:30] Awesome! The agent correctly identified it needed to search the web and used our search tool.
[09:00] Now let's talk about some advanced patterns. We have prompt chaining, routing, and parallelization.
[09:45] Prompt chaining is when you connect multiple prompts in sequence. The output of one becomes the input of the next.
[10:30] Routing is about directing queries to different specialized agents based on the question type.
[11:15] Parallelization allows you to run multiple operations simultaneously for better performance.
[12:00] Let me show you a real-world example using all these patterns together.
[12:45] We'll build a content creation assistant that can research, write, and optimize all in parallel.
[13:30] Here's the architecture. We have separate agents for research, writing, and SEO optimization running in parallel.
[14:15] The results are then combined by a coordinator agent that ensures everything flows well together.
[15:00] Let's run this and see how it performs. Notice how much faster it is compared to sequential processing.
[15:45] The parallel approach cut our processing time in half while maintaining quality.
[16:30] Now let's discuss best practices. Always handle errors gracefully, implement retry logic, and monitor your API usage.
[17:15] Testing is crucial. Write unit tests for your chains and integration tests for the full agent.
[18:00] For production, you'll want to add logging, monitoring, and potentially caching to reduce costs.
[18:45] Let's wrap up with some resources. I'll put links to the documentation and example code in the description.
[19:15] If you found this helpful, smash that like button and subscribe for more AI content.
[19:30] Thanks for watching! See you in the next video!
"""

print("\n📝 Sample Transcript Loaded:")
print("-" * 100)
print(sample_transcript[:500] + "...\n")

# ==================== PARALLEL ANALYSIS AGENTS ====================

# Agent 1: Topic Segmentation
topic_segmentation_prompt = ChatPromptTemplate.from_template("""
You are a video content analyzer. Analyze this transcript and identify major topic transitions.

Transcript:
{transcript}

Identify 5-8 major topic segments. For each segment:
- Note when the topic changes significantly
- Identify the timestamp of the change
- Describe what the new topic/section is about

Return your analysis as clear segments with timestamps and descriptions.
""")

# Agent 2: Keyword Extraction
keyword_extraction_prompt = ChatPromptTemplate.from_template("""
You are a keyword extraction specialist. Analyze this transcript and extract the most important keywords and concepts.

Transcript:
{transcript}

Extract:
- Main technical terms
- Important concepts
- Key topics discussed
- Framework/library names
- Actionable items

Return a structured list of keywords organized by importance.
""")

# Agent 3: Engagement Analysis
engagement_analysis_prompt = ChatPromptTemplate.from_template("""
You are a YouTube engagement analyst. Identify high-value moments in this transcript.

Transcript:
{transcript}

Identify moments that viewers would want to jump to:
- Tutorial/code examples
- Key explanations
- Demos/results
- Important tips
- Actionable advice

Return timestamps and descriptions of these high-value moments.
""")

# Agent 4: Structure Analysis
structure_analysis_prompt = ChatPromptTemplate.from_template("""
You are a content structure expert. Analyze the overall flow and structure of this video.

Transcript:
{transcript}

Analyze:
- Introduction length
- Main content sections
- Practical examples
- Conclusion/wrap-up
- Overall pacing

Suggest optimal chapter breaks based on content flow and pacing.
""")

# Create parallel chains
topic_chain = topic_segmentation_prompt | llm | StrOutputParser()
keyword_chain = keyword_extraction_prompt | llm | StrOutputParser()
engagement_chain = engagement_analysis_prompt | llm | StrOutputParser()
structure_chain = structure_analysis_prompt | llm | StrOutputParser()

# ==================== EXECUTE PARALLEL ANALYSIS ====================
print("🚀 Running Parallel Analysis...")
print("=" * 100)
print("\n⚡ Executing 4 agents in parallel:")
print("   1. Topic Segmentation Agent")
print("   2. Keyword Extraction Agent")
print("   3. Engagement Analysis Agent")
print("   4. Structure Analysis Agent")
print("\n⏳ Processing...\n")

parallel_chains = RunnableParallel(
    topic_segments=topic_chain,
    keywords=keyword_chain,
    engagement_moments=engagement_chain,
    structure_analysis=structure_chain
)

parallel_results = parallel_chains.invoke({"transcript": sample_transcript})

print("✓ Parallel analysis complete!")
print("=" * 100)

# Display results
print("\n" + "=" * 100)
print("📊 AGENT 1: TOPIC SEGMENTATION")
print("=" * 100)
print(parallel_results["topic_segments"])

print("\n" + "=" * 100)
print("🔑 AGENT 2: KEYWORD EXTRACTION")
print("=" * 100)
print(parallel_results["keywords"])

print("\n" + "=" * 100)
print("🎯 AGENT 3: ENGAGEMENT ANALYSIS")
print("=" * 100)
print(parallel_results["engagement_moments"])

print("\n" + "=" * 100)
print("📐 AGENT 4: STRUCTURE ANALYSIS")
print("=" * 100)
print(parallel_results["structure_analysis"])

# ==================== SYNTHESIZE CHAPTERS ====================
print("\n" + "=" * 100)
print("🎬 SYNTHESIZING FINAL CHAPTERS...")
print("=" * 100)

synthesis_prompt = ChatPromptTemplate.from_template("""
You are a YouTube chapter creator. Based on all the analysis below, create optimal chapter markers.

TOPIC SEGMENTS:
{topic_segments}

KEYWORDS:
{keywords}

ENGAGEMENT MOMENTS:
{engagement_moments}

STRUCTURE ANALYSIS:
{structure_analysis}

ORIGINAL TRANSCRIPT:
{transcript}

Create 8-12 chapter markers that:
1. Start with 00:00 (required by YouTube)
2. Use clear, engaging titles (max 80 characters)
3. Place chapters at natural breaks
4. Highlight key content and value moments
5. Use proper timestamp format (MM:SS or HH:MM:SS)

Format EXACTLY as:
00:00 - Introduction & Overview
01:45 - What are AI Agents?
03:30 - Setting Up LangChain
...

Return ONLY the chapter list, nothing else.
""")

synthesis_chain = synthesis_prompt | llm | StrOutputParser()

final_chapters = synthesis_chain.invoke({
    "topic_segments": parallel_results["topic_segments"],
    "keywords": parallel_results["keywords"],
    "engagement_moments": parallel_results["engagement_moments"],
    "structure_analysis": parallel_results["structure_analysis"],
    "transcript": sample_transcript
})

print("\n🎉 FINAL YOUTUBE CHAPTERS:")
print("=" * 100)
print(final_chapters)

# ==================== SAVE CHAPTERS ====================
output_filename = f"youtube_chapters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write("YOUTUBE CHAPTER MARKERS\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M %p')}\n")
    f.write(f"Using: Parallel Processing with Azure OpenAI\n\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("FINAL CHAPTERS (Copy to YouTube Description):\n")
    f.write("-" * 100 + "\n")
    f.write(final_chapters)
    f.write("\n\n" + "=" * 100 + "\n\n")
    
    f.write("DETAILED ANALYSIS:\n\n")
    
    f.write("TOPIC SEGMENTATION:\n")
    f.write("-" * 100 + "\n")
    f.write(parallel_results["topic_segments"])
    f.write("\n\n")
    
    f.write("KEYWORDS:\n")
    f.write("-" * 100 + "\n")
    f.write(parallel_results["keywords"])
    f.write("\n\n")
    
    f.write("ENGAGEMENT MOMENTS:\n")
    f.write("-" * 100 + "\n")
    f.write(parallel_results["engagement_moments"])
    f.write("\n\n")
    
    f.write("STRUCTURE ANALYSIS:\n")
    f.write("-" * 100 + "\n")
    f.write(parallel_results["structure_analysis"])
    f.write("\n\n")

print(f"\n💾 Chapters saved to: {output_filename}")

# ==================== COPY TO CLIPBOARD (BONUS) ====================
print("\n📋 COPY THIS TO YOUR YOUTUBE VIDEO DESCRIPTION:")
print("=" * 100)
print(final_chapters)
print("=" * 100)