from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew
from flask_cors import CORS  # Import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import Tool
import json
import requests

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Initialize the LLM and tools
serper = "7ee17327c8ac81766519b8ef3de5fd236f84ce66"
llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key="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")


def google_search(search_keyword):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_keyword})
    headers = {
        'X-API-KEY': serper,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()


google_tool = Tool(
    name="Google Search",
    func=google_search,
    description="Searches Google for articles based on the provided keyword."
)

# Define agents and tasks
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article about the topic: {topic} in 'https://medium.com/'.",
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[google_tool]
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory="You're working on a writing a new opinion piece about the topic: {topic} in 'https://medium.com/'.",
    allow_delegation=False,
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization 'https://medium.com/'.",
    backstory="You are an editor who receives a blog post from the Content Writer.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}, {keywords}, {bullet_points}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling blog post on {topic}, {keywords}, {bullet_points}, {length}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named in an engaging manner.\n"
        "4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post in markdown format, ready for publication.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for grammatical errors and alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, ready for publication.",
    agent=editor
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)


# Flask API routes
@app.route('/generate_blog_post', methods=['POST'])
def generate_blog_post():
    data = request.json

    # Extract inputs from the POST request
    topic = data.get("topic")
    keywords = data.get("keywords", "").split(",")
    bullet_points = data.get("bullet_points", "").split("\n")
    length = data.get("length", 500)

    inputs = {
        "topic": topic,
        "keywords": keywords,
        "bullet_points": bullet_points,
        "length": length
    }

    # Kick off the crew with user inputs
    result = crew.kickoff(inputs=inputs)

    # Assuming edit.output contains the final blog post text or similar
    edited_content = edit.output.content if hasattr(edit.output, 'content') else str(edit.output)

    # Return the generated content as JSON
    return jsonify({
        "edit": edited_content
    })

# Run the Flask app
if __name__ == '__main__':
    app.run()
