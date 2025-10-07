# """
# Image viewer tool for scientific analysis of climate-related visualizations.
# """

# import base64
# import os
# import logging
# from typing import Optional
# from pydantic import BaseModel, Field
# from langchain_core.tools import StructuredTool

# logger = logging.getLogger(__name__)


# def encode_image(image_path: str) -> str:
#     """Encode image to base64 string."""
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')


# def view_and_analyze_image(image_path: str, openai_api_key: str, model_name: str) -> str:
#     """
#     View and analyze a climate-related scientific image to extract insights.
#
#     Args:
#         image_path: Path to the image file
#         openai_api_key: OpenAI API key for vision model
#         model_name: Model name to use for analysis
#
#     Returns:
#         Scientific analysis and insights from the image
#     """
#     if not os.path.exists(image_path):
#         return f"Error: The file {image_path} does not exist."
#
#     try:
#         from openai import OpenAI
#
#         base64_image = encode_image(image_path)
#
#         prompt = """You are a climate scientist analyzing a climate-related visualization.
# Your task is to extract scientific insights and describe the data patterns shown in the image.

# Please provide a detailed analysis covering:

# 1. **Data Description**: What variables and data are being displayed? Identify the type of visualization (line chart, bar chart, heatmap, etc.) and the variables on each axis.

# 2. **Temporal Patterns**: If time-series data is shown, describe any trends, cycles, or changes over time. Note specific periods of interest.

# 3. **Quantitative Observations**: Extract specific values, ranges, or magnitudes visible in the image. Include units when identifiable.

# 4. **Climate Insights**: What climate-related patterns or phenomena are evident? This could include:
#    - Temperature changes or anomalies
#    - Precipitation patterns
#    - Seasonal variations
#    - Extreme events
#    - Future projections vs historical data

# 5. **Key Findings**: Summarize the most important scientific insights that can be drawn from this visualization.

# 6. **Implications**: Briefly discuss what these patterns might mean for climate impacts or decision-making at the location in question.

# Note: If the image quality makes any aspect difficult to read, mention this briefly and suggest regenerating the plot for better clarity.

# Provide a concise but thorough scientific analysis focused on extracting actionable insights from the data visualization."""

#         client = OpenAI(api_key=openai_api_key)
#
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/png;base64,{base64_image}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             max_tokens=5000
#         )
#
#         return response.choices[0].message.content
#
#     except ImportError:
#         return "Error: OpenAI library not available. Image analysis requires OpenAI API."
#     except Exception as e:
#         logger.error(f"Error analyzing image: {str(e)}")
#         return f"Error analyzing image: {str(e)}"


# class ImageViewerArgs(BaseModel):
#     """Arguments for image viewer tool."""
#     image_path: str = Field(
#         description="Path to the climate visualization image to analyze. This should be the full path to a saved plot or figure."
#     )


# def create_image_viewer_tool(openai_api_key: str, model_name: str):
#     """
#     Create the image viewer tool with the API key and model bound.
#
#     Args:
#         openai_api_key: OpenAI API key
#         model_name: Model name to use
#
#     Returns:
#         StructuredTool configured for image analysis
#     """
#     def view_image_wrapper(image_path: str) -> str:
#         return view_and_analyze_image(image_path, openai_api_key, model_name)
#
#     return StructuredTool.from_function(
#         func=view_image_wrapper,
#         name="image_viewer",
#         description=(
#             "Analyze climate-related visualizations to extract scientific insights. "
#             "Use this tool after generating plots with python_repl to understand "
#             "the patterns and trends shown in the visualization. "
#             "Provide the full path to the saved image file."
#         ),
#         args_schema=ImageViewerArgs
#     )