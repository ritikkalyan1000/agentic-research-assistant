# Agentic Research Assistant

An agentic research assistant built with Google ADK, Streamlit, Tavily Search, SiliconFlow image generation, and MCP humanizer support.

This project generates research reports by using a multi-agent workflow. The system can plan a research task, search for trusted sources, write a draft, review citations, check content quality, revise through a loop, generate useful image placeholders, create images, and compose the final report with visuals.

<img width="975" height="485" alt="image" src="https://github.com/user-attachments/assets/79956b65-a12a-4b1e-b5f6-9b7ac83b19c5" />

## Features

* Research planning agent
* Trusted source search using Tavily
* Draft writing agent
* Citation verification agent
* Content quality review agent
* Review aggregation and revision loop
* Image placeholder planning
* Image generation using SiliconFlow FLUX
* Final report composer with Markdown images
* MCP humanizer tool support
* Simple Streamlit frontend

## Project Structure

```text
agentic-research-assistant/
│
├── agent.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## How It Works

The project uses a multi-agent pipeline:

1. The planner agent understands the user’s research request and creates a research plan.
2. The source search agent finds trusted sources from the web.
3. The draft writer agent writes the first version of the report.
4. The citation verifier checks whether claims are supported by sources.
5. The content quality agent reviews the structure, clarity, and completeness of the draft.
6. The review aggregator combines feedback and decides if revision is needed.
7. The loop agent repeats the draft and review process until the report is ready or the maximum loop count is reached.
8. The image agent decides where visuals should be added and generates images.
9. The final composer inserts the generated images into the final report.

## Requirements

* Python 3.10+
* Node.js 16+
* OpenAI API key
* Tavily API key
* SiliconFlow API key

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/agentic-research-assistant.git
cd agentic-research-assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
```

Do not upload your real `.env` file to GitHub.

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL in your browser.

## Example Prompts

```text
Write a research report on depression among university students in Canada.
```

```text
Create an academic report about the digital divide in education with references and visuals.
```

```text
Explain how AI is changing healthcare using trusted sources.
```

## Notes

This project is for learning and demonstration purposes. The generated research output should be reviewed before academic or professional use. Citations and sources should always be checked manually for accuracy.

## Author

Ritik Kalyan
