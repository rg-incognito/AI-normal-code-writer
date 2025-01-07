from crewai import Agent, Task, Crew, Process
from langchain_ollama import ChatOllama

# Initialize the language model
llm = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Define the Agent (Web Developer)
rohit = Agent(
    role='Web Developer',
    goal='Build a complete website including HTML, CSS, and JS',
    backstory='20 years of experience in Tech, former SDE at Microsoft',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Task 1: Create the HTML
html_task = Task(
    description='Create the HTML structure for a furniture website.',
    agent=rohit,
    expected_output='<html>...</html>'
)

# Task 2: Create the CSS based on the HTML structure
css_task = Task(
    description='Create the CSS styling for the furniture website using the provided HTML structure.',
    agent=rohit,
    dependencies=[html_task],  # Ensure CSS is built after HTML
    expected_output='body { ... }'
)

# Task 3: Create the JavaScript for interactivity based on HTML and CSS
js_task = Task(
    description='Create the JavaScript functionality for the furniture website using the provided HTML and CSS.',
    agent=rohit,
    dependencies=[html_task, css_task],  # Ensure JS is built after HTML and CSS
    expected_output='function init() { ... }'
)

# Define the Crew (Group of tasks)
crew = Crew(
    agents=[rohit],
    tasks=[html_task, css_task, js_task],
    verbose=2,
    process=Process.sequential  # Ensure tasks run sequentially
)

# Start the tasks
result = crew.kickoff()

print('#######################################')
print(result)
