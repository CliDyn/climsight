"""
Question Generator for ClimSight

This module generates climate-related questions for different categories and validates their locations.
It uses LLM to generate questions and checks if the locations are on land.
"""

import os
import json
import random
import logging
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import geo functions
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'climsight'))
from geo_functions import is_point_onland

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

model_name = "gpt-5"#"gpt-4.1"
llm_temperature = 0.3

# Categories and their counts
CATEGORIES = [
    "Crop Yield and Productivity",
    "Food Production",
    "Agriculture",
    "Renewable Energy and Infrastructure",
    "Urban and Community Planning",
    "Climate Adaptation in Cropping Systems",
    "Long-Term Sustainability",
    "Public Health and Climate",
    "Tourism and Recreation",
    "Financial and Insurance Risk"
]



QUESTION_TYPES = ["general", "specific", "natural"]
QUESTIONS_PER_TYPE = 10 

ALL_AREAS=[
    "Whole World - Includes all major land regions on Earth, spanning every continent from North and South America to Europe, Africa, Asia, Australia, and Antarctica.",
	"North America - Includes Canada, the United States, Greenland, and Mexico, as well as Central America and the Caribbean.",
	"South America - Covers all countries of the South American continent, from Colombia and Brazil down to Argentina and Chile.",
	"Europe - Encompasses the entire European continent from the Atlantic coast to the Ural Mountains, including the European part of Russia.",
	"Middle East & North Africa - Spans North Africa and the Middle East, stretching from Morocco and Egypt across to Turkey, Saudi Arabia, and Iran.",
	"Sub-Saharan Africa - Includes all of Africa south of the Sahara Desert, such as Nigeria, Kenya, and South Africa.",
	"South & Southeast Asia - Covers the Indian subcontinent and Southeast Asia, from India and Pakistan to Thailand, Indonesia, and the Philippines.",
	"East Asia - Includes China, Mongolia, Japan, and the Korean Peninsula.",
	"Australia & Oceania - Includes Australia, New Zealand, Papua New Guinea, and the Pacific Island nations.",
	"Central & Northern Asia - Includes Russia's vast Siberian region and the Central Asian countries such as Kazakhstan and Uzbekistan.",
    "Antarctica - The entire ice-covered continent around the South Pole, with no permanent population."
]

AREA_FOR_QUESTIONS =ALL_AREAS[2]
 
# Path to land shapefile (adjust as needed)
LAND_SHAPEFILE = str(Path(__file__).parent.parent / 'data' / 'natural_earth')
# Example questions for the prompt
EXAMPLE_QUESTIONS = """
Examples of good questions :
- "What are the potential risks and benefits of climate change for growing wheat?"
- "What is the projected impact of rising temperatures on coffee production in this area?"
- "How might changing precipitation patterns affect water availability for irrigation in this region?"
- "Will expanding wheat fields in this region ensure consistent harvests over the next decade?"
- "Can we continue growing grapes at our vineyard located here under future climate scenarios?"
- "What is the expected monthly mean temperature change during the growing season for maize in Iowa, USA between 2020-2030 and 2040-2050?

General format for questions:
- Be specific about the location and time frame
- Focus on measurable impacts
- Include relevant metrics (yield, temperature, precipitation, etc.)

Natural style examples (for everyday/business users):
- "If I'm running a small vineyard near here, will summers in the 2030s be too hot for our grapes?"
- "I'm planning a wind farm near this location—do prevailing wind directions look stable in the next decade?"
- "Will outdoor work days be rained out more often around here in the 2040s?"
- "I manage a logistics hub near this spot—are stronger winds likely in the 2030s that could affect operations?"
- "If we expand our ski resort around here, will winters in the 2040s bring less snowfall (via lower precipitation or warmer temperatures)?"
- "For a rental property near this area, will summer heat waves feel noticeably worse in the next decade?"
- "We're opening a café with a big terrace at this location—how often should we expect rainy days in the 2030s?"
- "I'm evaluating rooftop solar near here—will cloudier months become more frequent due to precipitation changes in the 2040s?"
- "Is gardening near this location going to be tougher in the 2030s because of less regular rainfall?"
- "Could stronger winds around here in the next decade make cycling commutes harder?"
"""

def generate_llm_prompt() -> str:
    """Generate a prompt for the LLM to create climate questions for all categories."""
    return f"""Generate climate-related questions that can be answered with **monthly climatological means** from a climate model. For each category, provide:
- {QUESTIONS_PER_TYPE} general questions
- {QUESTIONS_PER_TYPE} specific questions
- {QUESTIONS_PER_TYPE} natural questions

Every question **must**:
- Include a "type" field of either "general", "specific", or "natural"
- Focus only on variables available in the data: temperature, precipitation, wind speed, or wind direction  
  *For **general** questions, you may mention these variables more broadly (e.g., "climate conditions" or "weather patterns") without listing each one explicitly.*  
- Specify one or more of the three decadal periods (2020-2030, 2030-2040, 2040-2050) or the change between two of them, it is not neccery to include years in the question, you can specify any year range or time period in words between 2020 - 2050, example: next decade
- Be answerable using those decadal **monthly mean** values (no daily extremes or percentiles), no need to include "monthly mean" in the question
- Reference a realistic land location and include its latitude and longitude
- Try to be as specific as possible with focus on particular crop or crop group
- For **natural** questions:
  - Use everyday language suitable for non-experts (business owners, residents, travelers).
  - Keep the focus on the same allowed variables (temperature, precipitation, wind speed/direction).
  - Prefer relative phrases (next decade, coming decade, over the next ten years, between the current decade and the next, toward the 2040-2050 horizon).
  - Phrase the question as a practical decision or concern (e.g., operations, comfort, scheduling, property use).

{EXAMPLE_QUESTIONS}

Format your response as a JSON object with the following structure:
{{
  "category_name": {{
    "general": [
      {{"question": "question text", "lon": longitude, "lat": latitude}},
      ...
    ],
    "specific": [
      {{"question": "question text", "lon": longitude, "lat": latitude}},
      ...
    ],
    "natural": [
      {{"question": "question text", "lon": longitude, "lat": latitude}},
      ...
    ]    
  }},
  ...
}}

Important guidelines:
0. Region to focus on: {AREA_FOR_QUESTIONS}\nAll generated questions must refer to locations inside: {AREA_FOR_QUESTIONS}.
1. Generate {QUESTIONS_PER_TYPE} general and {QUESTIONS_PER_TYPE} specific questions and {QUESTIONS_PER_TYPE} natural questions for each category
2. Choose realistic coordinates on land (not in oceans, seas, or large lakes)
3. Ensure questions are specific to the category and location
4. Include a variety of locations around the world
5. Focus on measurable climate impacts
6. Consider different time horizons (near-term (2020-2030), mid-century (2030-2040))
7. Do not include any coordinates in the question text
8. Prefer relative phrases (next decade, coming decade, over the next ten years, between the current decade and the next, toward the 2040-2050 horizon).

Categories to generate questions for:
{', '.join(CATEGORIES)}

Generate the questions in the specified JSON format:"""

def is_valid_location(lat: float, lon: float) -> bool:
    """Check if a point is on land."""
    try:
        on_land, _ = is_point_onland(lat, lon, LAND_SHAPEFILE)
        return on_land
    except Exception as e:
        logger.error(f"Error checking location ({lat}, {lon}): {e}")
        return False

def find_nearby_land_point(lat: float, lon: float, radius_degrees: float = 0.5, attempts: int = 10) -> Optional[Tuple[float, float]]:
    """Find a nearby point on land within the specified radius."""
    for _ in range(attempts):
        # Generate random offset within the radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius_degrees)
        
        # Convert polar to cartesian coordinates
        new_lat = lat + (distance * math.sin(angle))
        new_lon = lon + (distance * math.cos(angle))
        
        # Normalize coordinates
        new_lat = max(-90, min(90, new_lat))
        new_lon = (new_lon + 180) % 360 - 180  # Normalize longitude to [-180, 180]
        
        if is_valid_location(new_lat, new_lon):
            return new_lat, new_lon
    
    return None

def validate_and_fix_questions(questions: List[Dict]) -> List[Dict]:
    """Validate question locations and fix them if necessary."""
    valid_questions = []
    
    for q in questions:
        try:
            lat = q.get('lat')
            lon = q.get('lon')
            
            if lat is None or lon is None:
                logger.warning(f"Question missing coordinates: {q.get('question')}")
                continue
                
            # First check the original location
            if is_valid_location(lat, lon):
                valid_questions.append(q)
                continue
                
            # If not valid, try to find a nearby point on land
            logger.info(f"Location ({lat}, {lon}) is not on land. Searching for nearby land point...")
            new_point = find_nearby_land_point(lat, lon)
            
            if new_point:
                new_lat, new_lon = new_point
                logger.info(f"Found nearby land point: ({new_lat}, {new_lon})")
                q['lat'] = new_lat
                q['lon'] = new_lon
                valid_questions.append(q)
            else:
                logger.warning(f"Could not find valid land point for question: {q.get('question')}")
                
        except Exception as e:
            logger.error(f"Error processing question: {q.get('question')}. Error: {e}")
    
    return valid_questions

def generate_questions() -> Dict:
    """Generate questions for all categories and types in a single request."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    try:
        # Generate prompt for all categories
        prompt = generate_llm_prompt()
        
        llm_kwargs = {
            'model': model_name,
            'temperature': llm_temperature,
        }
        if "gpt-5" in model_name:
            llm_kwargs.update({
                'verbosity': "low",
                'reasoning_effort': "minimal",
                })

        # Create LLM instance
        llm = ChatOpenAI(
            openai_api_key=api_key,
            **llm_kwargs,
            streaming=False,
        )

        # Create messages for the LLM
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant that generates high-quality, specific climate-related questions "
                "with appropriate geographic locations. You always respond with valid JSON."
            )),
            HumanMessage(content=prompt)
        ]

        logger.info("Generating all questions in a single request...")
        response = llm.invoke(messages)

        # Extract content from AIMessage response
        response_text = response.content

        # Clean up the response to handle potential markdown code blocks
        if '```json' in response_text:
            content = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            content = response_text.split('```')[1].split('```')[0]
        else:
            content = response_text
        
        # Parse the response
        all_questions = json.loads(content)
        
        # Validate and fix question locations
        logger.info("Validating question locations...")
        for category, types in all_questions.items():
            for q_type, questions in types.items():
                valid_questions = validate_and_fix_questions(questions)
                all_questions[category][q_type] = valid_questions
                logger.info(f"Validated {len(valid_questions)} {q_type} questions for {category}")
        
        return all_questions
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise

def save_questions(questions: Dict) -> str:
    """Save questions to a JSON file with an incremental number."""
    # Find the highest existing question file number
    base_dir = Path(__file__).parent
    existing_files = list(base_dir.glob("questions_*.json"))
    
    if existing_files:
        # Extract numbers from filenames and find the max
        numbers = []
        for f in existing_files:
            try:
                num = int(f.stem.split('_')[-1])
                numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        if numbers:
            next_num = max(numbers) + 1
        else:
            next_num = 1
    else:
        next_num = 1
    
    # Create the output filename
    output_file = base_dir / f"questions_{next_num:02d}.json"
    
    # Prepare the output structure
    output = {"themes": questions}
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Questions saved to {output_file}")
    return str(output_file)

def validate_questions_file(file_path: str) -> dict:
    """
    Validate questions from an existing JSON file.
    Returns statistics about the validation.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Initialize statistics
        stats = {
            'total_questions': 0,
            'valid_questions': 0,
            'invalid_questions': 0,
            'categories': {},
            'invalid_locations': []
        }
        
        # Process each category
        for category, types in data.get('themes', data).items():
            if isinstance(types, dict):
                stats['categories'][category] = {t: 0 for t in QUESTION_TYPES}
                stats['categories'][category]['invalid'] = []
                
                for q_type, questions in types.items():
                    if q_type in QUESTION_TYPES and isinstance(questions, list):
                        for i, q in enumerate(questions):
                            stats['total_questions'] += 1
                            stats['categories'][category][q_type] += 1
                            
                            try:
                                lat = q.get('lat')
                                lon = q.get('lon')
                                
                                if lat is None or lon is None:
                                    raise ValueError("Missing coordinates")
                                    
                                if not is_valid_location(lat, lon):
                                    stats['invalid_questions'] += 1
                                    logger.warning(f"Location ({lat}, {lon}) is not on land for question: {q.get('question', 'No question text')}")
                                    stats['categories'][category]['invalid'].append({
                                        'type': q_type,
                                        'index': i,
                                        'question': q.get('question', 'No question text'),
                                        'reason': 'Not on land',
                                        'lat': lat,
                                        'lon': lon
                                    })
                                    stats['invalid_locations'].append({
                                        'category': category,
                                        'type': q_type,
                                        'lat': lat,
                                        'lon': lon,
                                        'question': q.get('question', 'No question text')
                                    })
                                else:
                                    stats['valid_questions'] += 1
                                    
                            except Exception as e:
                                stats['invalid_questions'] += 1
                                logger.warning(f"Error validating question: {q.get('question', 'No question text')}. Error: {e}")
                                stats['categories'][category]['invalid'].append({
                                    'type': q_type,
                                    'index': i,
                                    'question': q.get('question', 'No question text'),
                                    'reason': str(e),
                                    'lat': q.get('lat'),
                                    'lon': q.get('lon')
                                })
                                
        return stats
        
    except Exception as e:
        logger.error(f"Error validating questions file: {e}")
        raise

def print_validation_report(stats: dict):
    """Print a detailed validation report."""
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    
    # Summary
    print(f"\nTotal questions: {stats['total_questions']}")
    print(f"Valid questions: {stats['valid_questions']}")
    print(f"Invalid questions: {stats['invalid_questions']}")
    
    # By category
    print("\nQuestions by category:")
    for category, counts in stats['categories'].items():
        total = sum(counts.get(t, 0) for t in QUESTION_TYPES)
        invalid = len(counts['invalid'])
        print(f"\n{category}:")
        for t in QUESTION_TYPES:
            print(f"  - {t.capitalize()}: {counts.get(t, 0)}")
        print(f"  - Invalid: {invalid} ({(invalid/total)*100:.1f}%)")
    
    # Invalid locations
    if stats['invalid_locations']:
        print("\n" + "-"*80)
        print("INVALID LOCATIONS")
        print("-"*80)
        for loc in stats['invalid_locations']:
            print(f"\nCategory: {loc['category']} ({loc['type']})")
            print(f"Location: {loc['lat']}, {loc['lon']}")
            print(f"Question: {loc['question']}")
    
    print("\n" + "="*80)

def fix_questions_file(file_path: str) -> dict:
    """
    Validate and fix questions from an existing JSON file.
    If a question's location is not on land, find a nearby land point and update it.
    The updated questions are written back to the same file.
    Returns statistics about the fixes performed.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        stats = {
            'total_questions': 0,
            'fixed_questions': 0,
            'unchanged_questions': 0,
            'categories': {},
            'fixed_locations': []
        }

        changed = False

        for category, types in data.get('themes', data).items():
            if isinstance(types, dict):
                stats['categories'][category] = {t: 0 for t in QUESTION_TYPES}
                stats['categories'][category]['fixed'] = []
                for q_type, questions in types.items():
                    if q_type in QUESTION_TYPES and isinstance(questions, list):
                        for i, q in enumerate(questions):
                            stats['total_questions'] += 1
                            lat = q.get('lat')
                            lon = q.get('lon')
                            if lat is None or lon is None:
                                continue
                            if is_valid_location(lat, lon):
                                stats['unchanged_questions'] += 1
                                continue
                            # Try to find a nearby land point
                            new_point = find_nearby_land_point(lat, lon)
                            if new_point:
                                new_lat, new_lon = new_point
                                q['lat'] = new_lat
                                q['lon'] = new_lon
                                stats['fixed_questions'] += 1
                                stats['categories'][category]['fixed'].append({
                                    'type': q_type,
                                    'index': i,
                                    'old_lat': lat,
                                    'old_lon': lon,
                                    'new_lat': new_lat,
                                    'new_lon': new_lon,
                                    'question': q.get('question', 'No question text')
                                })
                                stats['fixed_locations'].append({
                                    'category': category,
                                    'type': q_type,
                                    'old_lat': lat,
                                    'old_lon': lon,
                                    'new_lat': new_lat,
                                    'new_lon': new_lon,
                                    'question': q.get('question', 'No question text')
                                })
                                changed = True
                            else:
                                stats['unchanged_questions'] += 1
        # Write back if any changes were made
        if changed:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        return stats
    except Exception as e:
        logger.error(f"Error fixing questions file: {e}")
        raise

def print_fix_report(stats: dict):
    """Print a report of the location fixes."""
    print("\n" + "="*80)
    print("LOCATION FIX REPORT")
    print("="*80)
    print(f"\nTotal questions: {stats['total_questions']}")
    print(f"Fixed questions: {stats['fixed_questions']}")
    print(f"Unchanged questions: {stats['unchanged_questions']}")
    if stats['fixed_locations']:
        print("\nFixed locations:")
        for loc in stats['fixed_locations']:
            print(f"\nCategory: {loc['category']} ({loc['type']})")
            print(f"Old: {loc['old_lat']}, {loc['old_lon']}")
            print(f"New: {loc['new_lat']}, {loc['new_lon']}")
            print(f"Question: {loc['question']}")
    print("\n" + "="*80)

def main():
    """Main function to handle both question generation and validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate or validate climate questions.')
    parser.add_argument('file', nargs='?', help='Optional: Path to the JSON file to validate or fix')
    parser.add_argument('--fix-locations', action='store_true', help='Fix invalid locations in the questions file')
    args = parser.parse_args()
    
    try:
        if args.file and args.fix_locations:
            # Fix mode
            logger.info(f"Fixing question locations in file: {args.file}")
            stats = fix_questions_file(args.file)
            print_fix_report(stats)
            logger.info("Location fixing completed successfully")
        elif args.file:
            # Validate mode
            logger.info(f"Validating questions from file: {args.file}")
            stats = validate_questions_file(args.file)
            print_validation_report(stats)
            logger.info("Validation completed successfully")
        else:
            # Generation mode
            logger.info("Starting question generation...")
            questions = generate_questions()
            output_file = save_questions(questions)
            logger.info(f"Question generation completed successfully. Output file: {output_file}")
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
