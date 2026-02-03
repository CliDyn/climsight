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

    prompt = """You are a STRICT professional reviewer of scientific images. Your task is to provide critical feedback to the visual creator agent so they can improve their visualization. Be particularly harsh on basic readability issues. Evaluate the provided image using the following criteria:

**CRITICAL FAILURES (Each results in automatic score reduction of at least 5 points):**
- Any overlapping text or labels
- Illegible or cut-off axis labels
- Missing axis titles or units
- Text that is too small to read clearly
- Labels that obscure data points

1. **Axis and Font Quality** (CRITICAL): Evaluate the visibility of axes and appropriateness of font size and style. ANY of the following issues should result in a score of 3/10 or lower:
   - Axis labels that are cut off, truncated, or partially visible
   - Font size that is too small to read comfortably
   - Missing axis titles or units
   - Poorly formatted tick labels (overlapping, rotated at bad angles, etc.)

2. **Label Clarity** (CRITICAL): This is ABSOLUTELY ESSENTIAL. If ANY text overlaps with other text, data points, or visual elements, the maximum possible score is 2/10. Check for:
   - Text overlapping with other text
   - Labels overlapping with data points or lines
   - Legend text that overlaps or is cut off
   - Annotations that clash with other elements
   
3. Color Scheme: Analyze the color choices. Is the color scheme appropriate for the data presented? Are the colors distinguishable and not causing visual confusion?

4. Data Representation: Evaluate how well the data is represented. Are data points clearly visible? Is the chosen chart or graph type appropriate for the data?

5. **Legend and Scale** (Important): Check the presence and clarity of legends and scales. If the legend overlaps with the plot area or has overlapping text, reduce score by at least 4 points.

6. Overall Layout: Assess the overall layout and use of space. Poor spacing that causes any text overlap should be heavily penalized.

7. Technical Issues: Identify any technical problems such as pixelation, blurriness, or artifacts that might affect the image quality.

8. Scientific Accuracy: To the best of your ability, comment on whether the image appears scientifically accurate and free from obvious errors or misrepresentations.

9. **Convention Adherence**: Verify that the figure follows scientific conventions. For example, when depicting variables like 'Depth of water' or other vertical dimensions, these should appear on the Y-axis with minimum values at the top and maximum depth at the bottom. This is a critically important scientific convention - if depth/vertical dimensions are incorrectly presented on the horizontal X-axis, assign a score of 1/10.

**SCORING GUIDELINES:**
- 7-10: Professional quality with no text/label issues
- 5-6: Minor issues but all text is readable
- 3-4: Significant problems including some text overlap or readability issues
- 1-2: Major failures with overlapping text, illegible labels, or missing critical elements

BE CRITICAL of any text overlap or readability issues. A visualization with overlapping text is fundamentally flawed and should receive a very low score regardless of other qualities.

Please provide a structured review addressing each of these points. Conclude with an overall assessment of the image quality, highlighting any significant issues or exemplary aspects. Finally, give the image a score out of 10."""
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
