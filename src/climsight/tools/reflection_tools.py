# src/tools/reflection_tools.py
import base64
import os
import logging
try:
    import streamlit as st  # Import streamlit to access session state
except ImportError:
    st = None
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from openai import OpenAI
try:
    from ..config import API_KEY as _API_KEY
except ImportError:
    from config import API_KEY as _API_KEY

# Define the function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def reflect_on_image(image_path: str) -> str:
    """
    Analyzes an image and provides feedback. Automatically resolves sandbox paths.
    """
    # --- NEW SANDBOX PATH RESOLUTION LOGIC ---
    final_image_path = image_path

    # If the path is relative, resolve it against the current session's sandbox
    if not os.path.isabs(image_path):
        thread_id = None
        if st is not None and hasattr(st, "session_state"):
            thread_id = st.session_state.get("thread_id")
        if not thread_id:
            thread_id = os.environ.get("CLIMSIGHT_THREAD_ID")

        if thread_id:
            sandbox_dir = os.path.join("tmp", "sandbox", thread_id)
            potential_path = os.path.join(sandbox_dir, image_path)
            if os.path.exists(potential_path):
                final_image_path = potential_path
                logging.info(f"Resolved relative path '{image_path}' to sandbox path '{final_image_path}'")
            else:
                logging.warning(f"Could not resolve relative path '{image_path}' in sandbox '{sandbox_dir}'")
        else:
            logging.warning("Could not resolve relative path: No thread_id available.")
    # --- END OF NEW LOGIC ---

    if not os.path.exists(final_image_path):
        # Provide a more informative error message
        return (f"Error: The file '{final_image_path}' (resolved from '{image_path}') does not exist. "
                f"This often happens if the file was not saved correctly or if there is a path mismatch between the agent's environment and the tool's environment.")

    base64_image = encode_image(final_image_path)

    prompt = (
        "You are a constructive reviewer of scientific climate visualizations.\n"
        "Evaluate the image and provide actionable, code-level feedback.\n\n"
        "## EVALUATION CRITERIA\n\n"
        "**Readability (35%)**\n"
        "- Axis labels: fully visible (not clipped), correctly sized, include units (°C, mm, m/s)\n"
        "- Tick labels: no overlap, readable formatting\n"
        "- Legend: present when needed, placed in non-overlapping position\n"
        "- Font sizes: title ≥14pt, axis labels ≥12pt, tick labels ≥10pt, legend ≥10pt\n\n"
        "**Data Representation (25%)**\n"
        "- Appropriate chart type (bar for categories, line for time series, etc.)\n"
        "- Data points clearly visible and distinguishable\n"
        "- Color scheme appropriate and distinguishable\n"
        "- Multiple series visually distinct (different colors, markers, or line styles)\n\n"
        "**Scientific Conventions (25%)**\n"
        "- ERA5 observations as black solid line with markers ('k-o') when alongside model data\n"
        "- Temperature: blue=cold, red=warm\n"
        "- Precipitation: green/blue palettes; units in mm/month or mm/day\n"
        "- Units consistent (don't mix °C and K, or mm and m)\n"
        "- Depth on Y-axis inverted (0 at top) if applicable\n\n"
        "**Layout & Polish (15%)**\n"
        "- Appropriate figsize, not cramped\n"
        "- No wasted whitespace, no clipped labels\n"
        "- No pixelation or rendering artifacts\n\n"
        "## SCORING GUIDELINES (be fair and proportional)\n\n"
        "- **9-10**: Excellent — publication-ready, no issues\n"
        "- **8**: Good — very minor cosmetic issues that don't affect readability\n"
        "- **7**: Acceptable — small issues (e.g., slightly small fonts, legend could be better placed)\n"
        "- **5-6**: Needs improvement — readability or convention issues that should be fixed\n"
        "- **3-4**: Poor — significant problems (overlapping text, missing labels, wrong units)\n"
        "- **1-2**: Unusable — fundamentally broken (wrong axis, scientifically misleading, illegible)\n\n"
        "IMPORTANT SCORING RULES:\n"
        "- A plot with readable text, correct units, and proper colors should score AT LEAST 7/10\n"
        "- Do NOT give 5/10 just because the legend is in a slightly suboptimal position\n"
        "- Do NOT give 5/10 for minor font size differences — only penalize if truly hard to read\n"
        "- Reserve scores below 6 for ACTUAL readability failures (text genuinely overlapping data,\n"
        "  labels cut off and unreadable, missing axis titles, wrong units)\n\n"
        "## RESPONSE FORMAT\n\n"
        "1. **Issues found** — bullet list of specific problems (if any)\n"
        "2. **Fix suggestions** — concrete matplotlib code for each issue, e.g.:\n"
        "   - `plt.legend(fontsize=10, loc='upper left', framealpha=0.9)`\n"
        "   - `plt.xticks(fontsize=10, rotation=0)`\n"
        "   - `plt.ylabel('Temperature (°C)', fontsize=12)`\n"
        "   - `plt.tight_layout()`\n"
        "3. **Score: X/10** — with one-line justification\n"
    )
    if not _API_KEY:
        return "Error: OPENAI_API_KEY not configured for reflect_on_image."

    openai_client = OpenAI(api_key=_API_KEY)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=1000
    )

    return response.choices[0].message.content

# Define the args schema for reflect_on_image
class ReflectOnImageArgs(BaseModel):
    image_path: str = Field(description="The path to the image to reflect on.")

# Define the reflect_on_image tool
reflect_tool = StructuredTool.from_function(
    func=reflect_on_image,
    name="reflect_on_image",
    description="A tool to reflect on an image and provide feedback for improvements.",
    args_schema=ReflectOnImageArgs
)
