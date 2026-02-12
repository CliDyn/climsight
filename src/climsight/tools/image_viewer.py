"""
Image viewer tool for scientific analysis of climate-related visualizations.
"""

import base64
import os
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def view_and_analyze_image(image_path: str, openai_api_key: str, model_name: str) -> str:
    """
    View and analyze a climate-related scientific image to extract insights.
    
    Args:
        image_path: Path to the image file
        openai_api_key: OpenAI API key for vision model
        model_name: Model name to use for analysis
        
    Returns:
        Scientific analysis and insights from the image
    """
    if not os.path.exists(image_path):
        return f"Error: The file {image_path} does not exist."
    
    try:
        from openai import OpenAI
        
        base64_image = encode_image(image_path)
        
        prompt = (
            "You are a climate scientist analyzing a climate-related visualization.\n"
            "Extract scientific insights and describe the data patterns shown.\n\n"
            "Provide analysis covering:\n\n"
            "1. **Data Description**: chart type (line chart, bar chart, heatmap, scatter, etc.),\n"
            "   variables on each axis with their units, data source if identifiable,\n"
            "   number of data series and what they represent\n\n"
            "2. **Key Values**: extract specific numbers visible in the image:\n"
            "   - Minimum and maximum values with their units\n"
            "   - Mean or typical values if discernible\n"
            "   - Any labeled thresholds, reference lines, or benchmarks\n"
            "   - Notable outliers or extreme values\n\n"
            "3. **Patterns & Trends**:\n"
            "   - Temporal trends: warming/cooling, drying/wetting, acceleration/deceleration\n"
            "   - Seasonal cycles: which months are peaks/troughs, amplitude of seasonal variation\n"
            "   - Anomalies: departures from expected patterns, sudden shifts\n"
            "   - Multi-series comparison: which series diverge and when\n\n"
            "4. **Climate Insights**:\n"
            "   - If ERA5 observations are shown (typically black line or markers):\n"
            "     note the observed baseline values for each variable\n"
            "   - If model projections are shown (typically colored lines, one per decade or scenario):\n"
            "     note the climate change signal — how do future values differ from the baseline?\n"
            "   - Identify critical months: months with largest projected changes,\n"
            "     threshold exceedances (e.g., >35°C, <30mm precipitation), or highest risk\n"
            "   - If multiple scenarios are shown: compare the spread and uncertainty\n\n"
            "5. **Key Findings**: 3-5 bullet summary of the most important scientific insights.\n"
            "   Each bullet should include at least one specific number with units.\n\n"
            "6. **Data Quality**: note any unreadable text, clipped labels, overlapping elements,\n"
            "   or unclear features. State what you CAN read clearly and flag what you cannot.\n"
            "   If the plot has significant readability issues, note them but still extract\n"
            "   whatever information is discernible.\n\n"
            "Be concise but quantitative. Always include units. "
            "Focus on what is actionable for climate impact assessment.\n"
        )

        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model=model_name,
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
            max_completion_tokens=5000
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "Error: OpenAI library not available. Image analysis requires OpenAI API."
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return f"Error analyzing image: {str(e)}"


class ImageViewerArgs(BaseModel):
    """Arguments for image viewer tool."""
    image_path: str = Field(
        description="Path to the climate visualization image to analyze. This should be the full path to a saved plot or figure."
    )


def create_image_viewer_tool(openai_api_key: str, model_name: str, sandbox_path: Optional[str] = None):
    """
    Create the image viewer tool with the API key and model bound.
    
    Args:
        openai_api_key: OpenAI API key
        model_name: Model name to use
        sandbox_path: Optional sandbox directory path for resolving relative paths
        
    Returns:
        StructuredTool configured for image analysis
    """
    def view_image_wrapper(image_path: str) -> str:
        resolved_path = image_path
        # If path is relative and sandbox_path is provided, resolve it
        if sandbox_path and not os.path.isabs(image_path):
            resolved_path = os.path.join(sandbox_path, image_path)
        # Fallback: if still not found and sandbox_path exists, try it anyway
        if not os.path.exists(resolved_path) and sandbox_path:
            alt_path = os.path.join(sandbox_path, image_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
        return view_and_analyze_image(resolved_path, openai_api_key, model_name)
    
    return StructuredTool.from_function(
        func=view_image_wrapper,
        name="image_viewer",
        description=(
            "Analyze climate-related visualizations to extract scientific insights. "
            "Use this tool after generating plots with python_repl to understand "
            "the patterns and trends shown in the visualization. "
            "Provide the path to the saved image file (can be relative to the sandbox)."
        ),
        args_schema=ImageViewerArgs
    )