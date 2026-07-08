from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool
from langchain_community.tools import TavilySearchResults

from typing import Optional, Literal, List, AsyncGenerator
from pydantic import BaseModel, Field


from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


import os
import requests

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# planing of rresearch agent loop for write the research papers then we review that
# root agent will have search tool so it can find the relevent information and then build steps
# for how the research paper will look like and what should be the things be there in the research
# at the end we will ask human to to see the research paper and if he want he can also humanize it (implement HITL)
# implement guardrails so user dont say anything toxic or absusive

# root agent will find the trusted source/ whitelisted sources from the internet related to user query
# and also write the steps on how the research should be going on and what are the things to be included in that

# loop agent1 -> follow the instruction given to it by the initial_root_agent and then write the things as per that
# parallel agents ->  p1 will search/ investigate whether the sources defined as useful and if useful then what ever is written by user by that source
# is that  really in these links or not
#  p2 -> it will look at the content by the loop agent 1 and give feedback on that
# accumulate both parallel rsult we have another agent will combine both the result and decide wheather to move forward or loop back
# for move forward we use custom agent and in that check whether the status is completed is there or not if not loop back if yes then finish the loop

# then we have image_agent it look at the research paper and decide places where we need images and put placeholder over there
# whcih can be filled by the next image generation agent
# finally final response shown to user along with whether do he want to humanize it or not
# if yes then humanize it otherwise not

app_name = "research_agent"
user_id = "Ritik_kalyan"
session_id = "session_1"


class output_schema_planner_agent(BaseModel):
    user_goal: str = Field(
        description="Explain what the user wants to research or create. Make the goal specific and clear."
    )

    user_intent: str = Field(
        description="Identify whether the user wants a summary, report, essay, annotated bibliography, assignment answer, comparison, explanation, presentation content, or another type of research output."
    )

    Target_audience: str = Field(
        description="Identify the likely audience, such as high school students, university students, general readers, technical experts, or professional readers. If the user does not mention an audience, choose the most reasonable audience based on the request."
    )

    Sources_requirement: str = Field(
        description="Define what types of sources should be used. Prefer trusted and high-quality sources such as: peer-reviewed journal articles,government websites, university sources, international organizations, official reports, reputable technical documentation, recognized industry sources"
    )

    avoided_sources: str = Field(
        description="Explain which sources should be avoided or used carefully, such as: random blogs, unsupported opinion articles, AI-generated content farms, outdated sources, sources with no author, date, or organization, Wikipedia as a primary source"
    )

    Citation_requirement: str = Field(
        description="State what citation style should be used if the user asks for one. If the user does not specify a citation style, recommend a suitable style based on the task. For engineering or technical writing, prefer IEEE style."
    )

    writing_requirement: str = Field(
        description="Give clear writing instructions for the Draft Writer Agent, including tone, level of detail, clarity, and whether the answer should be academic, simple, professional, or technical."
    )

    suggested_sections: list[str] = Field(
        description="list of section that as per the topic be included in the report"
    )


initial_root_agent = LlmAgent(
    name="initial_root_agent",
    model=LiteLlm(model="openai/gpt-5.4"),
    instruction="""
    You are the Planner Agent for an agentic research assistant system.
    Your job is to understand the user’s research request and create a clear research plan for the next agents. You do not write the final research answer. You do not generate images. You do not rewrite in a human/student tone. Your main responsibility is planning, source rules, structure, and quality requirements.
    For every user request, analyze the topic carefully and produce a structured plan that can be followed by the source-search, writing, review, verification, and final-response agents.
    """,
    output_schema=output_schema_planner_agent,
    output_key="initial_root_agent_output_key",
)


# squential flow with root agent
def tavily_online_Search(query: str) -> list:
    tavily_created = TavilySearchResults(max_results=7)
    result = tavily_created.invoke({"query": query})
    return result


source_search_agent = LlmAgent(
    name="source_search_agent",
    model=LiteLlm(model="openai/gpt-5.4-nano"),
    instruction="you are helpful whitelisted source search agent and your job is to Search the web. Find trusted/whitelisted sources.",
    tools=[tavily_online_Search],
    output_key="source_search_agent_output_key",
)

sequential_root_source = SequentialAgent(
    name="sequential_root_source", sub_agents=[initial_root_agent, source_search_agent]
)
# loop flow
# draft writter agent
draft_writter_agent = LlmAgent(
    name="draft_writter_agent",
    model=LiteLlm(model="openai/gpt-5.5"),
    instruction="""
    You are the Draft Writer Agent for an agentic research assistant system.

Your job is to write the first complete draft of the research output by following the plan created by the Planner Agent {initial_root_agent_output_key} and using only the sources approved or provided by the Source Search Agent Use these approved sources: {source_search_agent_output_key}. You do not decide the research plan. You do not create source rules. You do not verify citations. You do not generate images. Your main responsibility is to turn the approved research plan and approved evidence into a clear, well-structured draft.

You must follow the Planner Agent’s instructions carefully.

For every task, review the Planner Agent’s output before writing. Use the research goal, user intent, target audience, required structure, key research questions, source requirements, citation requirements, writing requirements, and completion criteria as your guide.

Your draft must include:

Clear Structure
Follow the section structure recommended by the Planner Agent. Use clear headings and organize the content logically.
Relevant Introduction
Begin with an introduction that explains the topic, gives context, and shows why the topic matters. The introduction should match the user’s requested level and audience.
Main Body Content
Develop the main points required by the Planner Agent. Explain ideas clearly and connect them to the research goal. Each section should answer part of the user’s research question.
Evidence-Based Support
Use only evidence from approved sources. Support important claims with source-based information such as definitions, statistics, findings, examples, case studies, or expert arguments.
Accurate Citations
Add citations wherever source-based information is used. Use the citation style required by the Planner Agent. If the citation style is not specified, use the style recommended by the Planner Agent.
Balanced Explanation
Include different dimensions of the topic where appropriate. Do not present only one side if the topic requires comparison, debate, causes, effects, solutions, limitations, or multiple viewpoints.
Clear Transitions
Connect paragraphs and sections smoothly so the draft reads like one complete piece, not a list of disconnected notes.
Limitations or Gaps
If the Planner Agent asks for limitations, challenges, research gaps, or weaknesses, include them clearly.
Conclusion
End with a conclusion that summarizes the main findings and connects back to the research goal. Do not introduce major new evidence in the conclusion.
Reference List
Include a reference list if required by the Planner Agent. Only include sources that were actually used in the draft.

Important writing rules:

Do not invent sources, authors, statistics, page numbers, dates, URLs, or citations.
Do not use sources that were not approved or provided by the Source Search Agent.
Do not make claims that are not supported by the available evidence.
Do not overstate what a source says.
Do not copy large passages from sources.
Do not write in an overly advanced or robotic style.
Match the tone requested by the Planner Agent, such as academic, simple, professional, technical, or student-friendly.
Keep the writing clear, natural, and easy to understand.
If evidence is missing for a section, mention the gap clearly instead of inventing information.
If two sources disagree, explain the difference instead of choosing one without reason.
If the task is academic, avoid casual language, slang, and unsupported opinions.
If the user requested a student-style response, write in a natural student tone while still being accurate and professional.
    
    """,
    output_key="draft_agent_output_key",
)


# parallel flow in loop flow with the draft writter agent
class cited_source(BaseModel):

    Claim_from_draft: str = Field(
        description="The exact claim, fact, statistic, definition, quotation, or argument from the draft that needs citation verification."
    )

    Cited_source: str = Field(
        description="The source cited for the claim. Include the citation label, author, title, URL, DOI, or reference details if available."
    )

    Verification_result: Literal[
        "Supported", "Partially_supported", "Not_supported"
    ] = Field(
        description="Whether the cited source fully supports, partially supports, or does not support the claim from the draft."
    )

    Problem: str = Field(
        description="Explain the issue found during verification. If the claim is fully supported, state that no major problem was found. If partially or not supported, explain what does not match, such as wrong statistic, exaggerated wording, missing context, incorrect date, or unsupported interpretation."
    )

    Suggested_fix: str = Field(
        description="Suggest how to correct the issue, such as revising the claim, adding context, replacing the citation, using a stronger source, or removing the unsupported claim."
    )


class output_schema_citation_agent(BaseModel):
    overall_status: Literal["Pass", "Fail", "Need_Revision"] = Field(
        description="Overall result of the citation verification. Use Pass if citations are mostly correct, Need_Revision if some citations need correction, and Fail if many claims are unsupported or sources are unreliable."
    )

    main_issue_found: List[cited_source] = Field(
        description="A list of citation verification issues found in the draft. Each item should include the claim, cited source, verification result, problem, and suggested fix."
    )


class output_schema_context_agent(BaseModel):
    overall_quality_status: Literal["Pass", "Fail", "Need_Revision"] = Field(
        description="Overall result of the citation verification. Use Pass if citations are mostly correct, Need_Revision if some citations need correction, and Fail if many claims are unsupported or sources are unreliable."
    )
    summary_feedback: List[str] = Field(
        description="A short summary of the overall quality of the draft."
    )
    strengths: List[str] = Field(description="List what the draft does well.")
    major_issues: List[str] = Field(
        description="List serious problems that must be fixed before the draft can be considered complete. If there are no major issues, write an empty list."
    )
    missing_content: List[str] = Field(
        description="List any important content, sections, explanations, or perspectives that are missing. if there is no missing content then write empty list"
    )
    clarity_and_structure_feedback: str = Field(
        description="Explain whether the structure, flow, headings, and explanation quality are effective."
    )

    revision_instructions_for_draft_writer: str = Field(
        description="Give clear instructions that the Draft Writer Agent can follow to improve the draft."
    )
    status: Literal["content_quality_needs_revision", "content_quality_passed"] = Field(
        description="content_quality_passed if the draft is strong enough to continue.content_quality_needs_revision if the draft should return to the Draft Writer Agent."
    )


Citation_Verifier_Agent = LlmAgent(
    name="Citation_Verifier_Agent",
    model=LiteLlm(
        model="openai/gpt-5.4-nano"
    ),  # we will use some perplexity model becuas they are better in reading from the web things
    instruction="""
    You are a Citation Verification Agent.

    Your job is to check whether the claims, facts, statistics, definitions,
    quotations, and arguments in the draft {draft_agent_output_key} are actually supported by the cited sources.

    Be strict and factual. Do not invent evidence. Do not approve a citation
    unless the source clearly supports the claim.

    Return the result using the required structured output schema.
    
    """,
    tools=[tavily_online_Search],
    output_schema=output_schema_citation_agent,
    output_key="Citation_Verifier_Agent_output_key",
)


# parallel 2
Content_Quality_Agent = LlmAgent(
    name="Content_Quality_Agent",
    model=LiteLlm(
        model="openai/gpt-5.4"
    ),  # we will use some perplexity model becuas they are better in reading from the web things
    instruction="""
    You are the Content Quality Agent for an agentic research assistant system.

Your job is to review the draft created by the Draft Writer Agent {draft_agent_output_key} and judge the quality of the writing, structure, explanation, completeness, and usefulness. You do not rewrite the full draft. You do not verify every citation in detail. You do not search for new sources. You do not generate images. Your main responsibility is to identify content-level problems and give clear feedback for improvement.

You must review the draft by comparing it with the Planner Agent’s research plan and the user’s original request.

Your review must check:

Alignment With User Request
Check whether the draft answers what the user actually asked for. Identify if the draft goes off-topic, misses the main goal, or includes unnecessary content.
Alignment With Planner Agent
Check whether the draft follows the research goal, target audience, required structure, key research questions, writing requirements, and completion criteria created by the Planner Agent.
Structure and Organization
Check whether the draft has a clear introduction, body sections, conclusion, and references if required. Identify missing sections, weak headings, poor ordering, or confusing flow.
Clarity of Explanation
Check whether the ideas are explained clearly and naturally. Identify sentences or sections that are vague, confusing, repetitive, too short, too long, or difficult to understand.
Depth and Completeness
Check whether the draft gives enough detail for the requested task. Identify missing causes, effects, examples, solutions, limitations, comparisons, definitions, or background information where needed.
Academic and Professional Quality
Check whether the tone matches the task. For academic tasks, the draft should sound formal, balanced, and evidence-based. For student-style tasks, it should still be clear, natural, and not overly casual.
Logical Flow
Check whether paragraphs connect well to each other. Identify places where transitions are weak or where the argument jumps suddenly from one idea to another.
Evidence Use
Check whether the draft uses evidence in a meaningful way. You do not need to fully verify citations, but you should identify places where claims seem unsupported, overgeneralized, too broad, or lacking examples.
Balance and Fairness
Check whether the draft presents the topic fairly. Identify if it ignores important perspectives, presents only one side of a debate, or makes claims that sound biased.
Redundancy and Repetition
Check whether the same idea is repeated too many times. Suggest where content can be combined, shortened, or made more focused.
Missing Limitations or Gaps
Check whether the draft includes limitations, challenges, or research gaps if the Planner Agent requested them.
Conclusion Quality
Check whether the conclusion summarizes the main points clearly and connects back to the research goal. Identify if the conclusion is too short, too vague, or introduces new unsupported ideas.

Important rules:

Do not rewrite the entire draft.
Do not approve weak content just because it has citations.
Do not focus only on grammar; focus on overall content quality.
Do not invent new facts or sources.
Do not perform detailed citation verification; leave that to the Citation Verifier Agent.
Do not generate images or visual placeholders.
Give specific feedback, not vague comments like “make it better.”
If the draft is good, still mention minor improvements if any.
If the draft has serious problems, clearly mark it as needing revision.

Your output must include:

overall_quality_status
Use one of the following:
"pass"
"needs_minor_revision"
"needs_major_revision"
summary_feedback
A short summary of the overall quality of the draft.
strengths
List what the draft does well.
major_issues
List serious problems that must be fixed before the draft can be considered complete. If there are no major issues, write an empty list.
missing_content
List any important content, sections, explanations, or perspectives that are missing.
clarity_and_structure_feedback
Explain whether the structure, flow, headings, and explanation quality are effective.
revision_instructions_for_draft_writer
Give clear instructions that the Draft Writer Agent can follow to improve the draft.
status
Use one of the following:
"content_quality_passed" if the draft is strong enough to continue.
"content_quality_needs_revision" if the draft should return to the Draft Writer Agent.
    
    """,
    output_schema=output_schema_context_agent,
    output_key="Content_Quality_Agent_output_key",
)

parallel_citation_content = ParallelAgent(
    name="parallel_citation_content",
    sub_agents=[Citation_Verifier_Agent, Content_Quality_Agent],
)


# merge parallel

Review_Aggregator_Agent = LlmAgent(
    name="Review_Aggregator_Agent",
    model=LiteLlm(model="openai/gpt-5.4-nano"),
    instruction="""
    You are the Review Aggregator Agent in an agentic research-writing system.

    Your job is to read and combine feedback from multiple reviewer agents, such as:
    - Citation Verification Agent {Citation_Verifier_Agent_output_key}
    - Content Quality Agent {Content_Quality_Agent_output_key}
    - Structure Review Agent
    - Any other review agent in the workflow

    Your main responsibility is to merge all reviewer feedback into one clear, non-duplicated revision plan for the Draft Writer Agent.

    Follow these rules:

    1. Read all reviewer feedback carefully.
    2. Identify repeated or overlapping feedback and combine it into one clear point.
    3. Separate major issues from minor issues.
    4. Keep only useful, actionable feedback that can help improve the draft.
    5. Remove vague, unnecessary, or duplicate comments.
    6. Do not rewrite the full draft yourself.
    7. Do not add new research claims unless they are clearly supported by reviewer feedback.
    8. Do not invent citation problems or content problems.
    9. If reviewers disagree, explain the disagreement clearly and choose the safer academic recommendation.
    10. Decide whether the draft needs another revision cycle.

    If the draft is strong enough and there are no important citation, accuracy, structure, or content issues left, return exactly:

    DONE

    If the draft needs revision, return a clear combined feedback report for the Draft Writer Agent
    
    """,
    output_key="check_up",
)


# basemodel just for check the review agent output so we can stop it early  but as i see the google adk they use the tool to do this kind of work but what i did is also good
class checkEvaluation(BaseAgent):

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        status = ctx.session.state.get("check_up", "there is some feedback")
        if status.lower() == "done":
            yield (Event(author=self.name, actions=EventActions(escalate=True)))
        else:
            yield Event(author=self.name)


check_status = checkEvaluation(name="name_is_checking")


looping = LoopAgent(
    name="looping",
    sub_agents=[
        draft_writter_agent,
        parallel_citation_content,
        Review_Aggregator_Agent,
        check_status,
    ],
    max_iterations=4,
)


make_it_final_before_images_agent = SequentialAgent(
    name="make_it_final_before_images_agent",
    sub_agents=[sequential_root_source, looping],
)


class output_schema_for_image_agent_planner(BaseModel):
    placeholder_id: int = Field(
        description="A unique numeric ID for the image placeholder in the research paper."
    )

    placeholder_name: str = Field(
        description="A short, clear snake_case name for the image placeholder, such as 'digital_divide_access_gap_chart'."
    )

    image_description: str = Field(
        description="A detailed description of the image that should be created. Include what the image should show, the main concept, important labels, and how it supports the surrounding text."
    )

    position_in_report: str = Field(
        description="Exact location in the report where the image should be inserted, such as section name, paragraph number, or after a specific sentence."
    )

    image_prompt_used: str = Field(
        description="The final prompt sent to the image generation tool."
    )

    image_generation_status: str = Field(
        description="Status of image generation. Use 'success' if the image was generated, otherwise use 'failed'."
    )

    generated_image_url: Optional[str] = Field(
        default=None,
        description="The generated image URL returned by the image generation tool. If image generation failed, this should be null.",
    )

    image_generation_error: Optional[str] = Field(
        default=None,
        description="If image generation failed, explain the reason. If successful, this should be null.",
    )


class placeholder_schema(BaseModel):
    image_placeholders: List[output_schema_for_image_agent_planner] = Field(
        description="List of all image placeholders, including their positions, descriptions, prompts, and generated image URLs."
    )


def generate_flux_image(prompt: str) -> dict:
    """
    Generate an image using SiliconFlow FLUX.2 Pro.
    Args:
        prompt: The image generation prompt.

    Returns:
        image URL
    """

    url = "https://api.siliconflow.com/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "black-forest-labs/FLUX.2-pro",
        "prompt": prompt,
        "image_size": "1024x1024",
        "batch_size": 1,
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return {
            "status": "error",
            "message": response.text,
        }

    data = response.json()

    return {
        "status": "success",
        "image_response": data,
    }


# agentic_image_tool = AgentTool(agent=image_generation_agent)

placeholder_and_image_generation_agent = LlmAgent(
    name="placeholder_and_image_generation_agent",
    model=LiteLlm(model="openai/gpt-5.4"),
    instruction="""
You are the Placeholder and Image Generation Agent in an agentic research-writing system.

You will receive a complete report or research paper draft from the Draft Writer Agent using the input key:
{draft_agent_output_key}

Your job has two parts:

PART 1: IMAGE PLACEHOLDER PLANNING
Read the full draft carefully and decide where images would improve the reader's understanding, readability, academic presentation, or visual clarity.

Only add image plans where a visual clearly supports the nearby text.

A useful image may:
- explain a complex concept,
- compare two or more ideas,
- show a process or framework,
- summarize key causes or impacts,
- present data visually,
- improve understanding of policy solutions or interventions.

Avoid adding images when:
- the paragraph is already simple and clear,
- the image would only be decorative,
- the same idea has already been covered by another image,
- the image would distract from the academic writing.

PART 2: IMAGE GENERATION
For every image placeholder you create, you must call the available image generation tool.

The image generation tool is available to you as a tool. Use it to generate the image for each placeholder.

When calling the image generation tool:
- create a clear and safe image prompt,
- avoid copyrighted brand names, real private people, or unsafe content,
- describe the image in a generic academic/educational style,
- include labels if the image is a diagram, comparison, framework, or process,
- make the image match the surrounding report content.

For each generated image, collect the image URL returned by the tool.

IMPORTANT:
You must not only plan placeholders.
You must also generate the images using the attached image generation tool.
You must include the generated image URL in the structured output.

For each placeholder, return:
- placeholder_id
- placeholder_name
- position_in_report
- image_description
- image_prompt_used
- image_generation_status
- generated_image_url
- image_generation_error

The placeholder_id must be unique.
The placeholder_name must be short, clear, descriptive, and written in snake_case.
The position_in_report must explain exactly where the image should appear in the draft.
The image_description must explain what the image shows and how it supports the nearby text.
The image_prompt_used must be the exact prompt you sent to the image generation tool.
The generated_image_url must be the URL returned by the image generation tool if successful.
If image generation fails, set image_generation_status to 'failed', generated_image_url to null, and explain the issue in image_generation_error.

Return your final answer using the required structured output schema only.
""",
    output_schema=placeholder_schema,
    output_key="placeholder_and_image_generation_agent_output_key",
    tools=[generate_flux_image],
)


final_pipeline_agent = LlmAgent(
    name="final_pipeline_agent",
    model=LiteLlm(model="openai/gpt-5.4-nano"),
    instruction="""
    
    You are the Final Report Composer Agent in an agentic research-writing system.

You will receive two inputs:

1. The complete draft report from the Draft Writer Agent using this input key:
{draft_agent_output_key}

2. The image placeholder and generated image data from the Placeholder and Image Generation Agent using this input key:
{placeholder_and_image_generation_agent_output_key}

Your job is to create the final report by inserting the generated images into the correct places in the draft.

The placeholder and image data contains a list of image placeholders. Each placeholder may include:
- placeholder_id
- placeholder_name
- image_description
- position_in_report
- image_prompt_used
- image_generation_status
- generated_image_url
- image_generation_error

For each placeholder:
1. Read the position_in_report carefully.
2. Find the correct place in the draft where the image should be inserted.
3. Insert the image using the generated_image_url.
4. Add a short academic caption under the image.
5. Make sure the image supports the nearby paragraph.
6. Do not place images randomly.
7. Do not remove important content from the original draft.

When inserting an image, use Markdown image format:

![short descriptive alt text](generated_image_url)

After the image, add a caption in this format:

Figure X. Short academic caption explaining what the image shows and how it connects to the report.

Rules:
- Preserve the original draft as much as possible.
- Only make small formatting changes if needed.
- Do not rewrite the full report unnecessarily.
- Do not invent new image URLs.
- Use only the generated_image_url provided by the placeholder_and_image_generation_agent.
- If image_generation_status is "failed" or generated_image_url is missing, do not insert a broken image.
- Instead, insert a clear placeholder note such as:

[Image could not be generated for Figure X: reason]

- Number figures in order based on where they appear in the final report.
- Use clear alt text based on placeholder_name or image_description.
- Make sure every inserted image appears near the section or paragraph described in position_in_report.
- Return only the final completed report with images inserted.
""",
    output_key="final_report_agent_output_key",
)

ai_humanizer_mcp_tool = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "ai-humanizer-mcp-server"],
        )
    )
)


research_agent = SequentialAgent(
    name="research_agent",
    sub_agents=[
        make_it_final_before_images_agent,
        placeholder_and_image_generation_agent,
        final_pipeline_agent,
    ],
    description="This is a research agent for a given topic it will generate high grade research report containing references, visuals, covering all the aspects regarding to the user query",
)


root_agent = LlmAgent(
    name="root_agent_finals",
    model=LiteLlm(model="openai/gpt-5.4-nano"),
    instruction="you are a helpful assistant for day to day questions and also work for given query you will deligate work either to research_agent or you also have tools to humanize the ai generated work so use tool for that only when user says to humanize the text",
    tools=[ai_humanizer_mcp_tool],
    sub_agents=[research_agent],
)
