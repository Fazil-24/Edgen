import os
import requests  # Assuming you need this for the CrewAI complexity assessment
from crewai import Agent, Task, Crew, LLM
import ollama
from openai import OpenAI
import streamlit as st

# Initialize Cloud Model API Client (LLaMA 70B)
client = OpenAI(
    api_key="e0853735c1a244d388de6a568e2e11b8",  # Insert your actual API key here
    base_url="https://api.aimlapi.com",
)

# Title for the Streamlit app
#logo = st.image("logo.jpg", width=200)
st.title("EdGen: Smart Sustainable Solution to Assist Farmers")
st.write("Welcome to EdGen! A smart AI-powered assistant to help farmers with crop management, plant care, and farming advice. (Note: Download the llama 3.2 3B model locally)")

# Get user input for the query
query = st.text_input("Ask a question related to farming or crop management:")

# Initialize the LLM for CrewAI
llm = LLM(model="gemini/gemini-1.5-pro", temperature=0.3, verbose=True, api_key="AIzaSyCxmxt1nhFZ5sD8GMU8iR1HJgvsp3Hurpk")

# Function for AI agent to assess complexity
def assess_complexity(query):
    complexity_agent = Agent(
        role='Query Complexity Evaluator',
        goal=f"Understand and assess the complexity of the following query: {query}. If the query is more than 8 words, consider it is a complex question. Provide only score value between 0 (simple) and 1 (complex).",
        backstory="You are an agent responsible for evaluating query complexity based on the user's question.",
        llm=llm,
    )

    # Create a task for complexity evaluation
    complexity_task = Task(
        description=f"Evaluate the complexity of the query: {query}.",
        agent=complexity_agent,
        expected_output="A complexity score between 0 and 1."
    )

    # Initialize Crew for complexity assessment
    crew_complexity = Crew(
        agents=[complexity_agent],
        tasks=[complexity_task],
        verbose=False
    )
    
    # Execute the Crew task and get the complexity score
    complexity_output = crew_complexity.kickoff()
    complexity_score = float(complexity_output.tasks_output[0].raw)

    return complexity_score

# Local LLaMA model handler (for simpler queries)
def handle_local_llama(query):
    prompt = f'''
    You are an AI assistant for answering general questions about plant care and farming for everyday use. You help farmers with basic, practical advice about watering schedules, soil conditions, sunlight requirements, and common plant issues. Keep your responses short, clear, and easy to understand.

    {query}
    '''
    response = ollama.generate(model='llama3.2', prompt=prompt)
    return response['response']

# Cloud LLaMA model handler (for complex queries)
def handle_cloud_llama(query):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a farmer assistant. Based on the user question, provide clear detailed answers.",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )
    return response.choices[0].message.content

# Process the query when the user submits it
if query:
    complexity_score = assess_complexity(query)
    st.write(f"Complexity Score: {complexity_score:.2f}")

    # Determine which model to use based on complexity
    if complexity_score < 0.5:
        st.write("This is a simple question. Using the local LLaMA model...")
        response = handle_local_llama(query)
    else:
        st.write("This is a complex question. Using the cloud LLaMA model...")
        response = handle_cloud_llama(query)
    
    # Display the AI response
    st.write("AI Response:")
    st.write(response)
