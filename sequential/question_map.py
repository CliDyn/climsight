"""
Streamlit app to visualize climate questions on an interactive map.
"""
import streamlit as st
import json
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import colorsys
import hashlib

# Set page config
st.set_page_config(
    page_title="ClimSight Question Map",
    page_icon="🌍",
    layout="wide"
)

def load_questions(file_path):
    """Load questions from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    themes = data.get('themes', {})
    
    for theme, types in themes.items():
        for q_type in ['general', 'specific']:
            for q in types.get(q_type, []):
                questions.append({
                    'theme': theme,
                    'type': q_type,
                    'question': q['question'],
                    'lat': q['lat'],
                    'lon': q['lon']
                })
    
    return pd.DataFrame(questions)

def get_color_for_theme(theme_name, question_type):
    """Generate a consistent color for each theme with different shades for question types."""
    # Create a hash of the theme name to get consistent colors
    hash_obj = hashlib.md5(theme_name.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    
    # Generate HSV color with fixed saturation and value
    hue = (hash_int % 360) / 360.0
    saturation = 0.8
    value = 0.9 if question_type == 'general' else 0.6  # Lighter for general, darker for specific
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Convert to hex color
    return '#%02x%02x%02x' % (
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

def create_map(questions_df):
    """Create a folium map with question markers."""
    if questions_df.empty:
        return None
    
    # Create a base map
    m = folium.Map(
        location=[questions_df['lat'].mean(), questions_df['lon'].mean()],
        zoom_start=2,
        tiles='cartodbpositron'
    )
    
    # Group by theme and type to create layer controls
    layer_control = folium.map.LayerControl(position='topright')
    
    # Create a feature group for each theme and type combination
    feature_groups = {}
    
    # Add markers for each question
    for _, row in questions_df.iterrows():
        theme = row['theme']
        q_type = row['type']
        layer_name = f"{theme} ({q_type})"
        
        # Create feature group if it doesn't exist
        if layer_name not in feature_groups:
            feature_groups[layer_name] = folium.FeatureGroup(name=layer_name, show=True)
            m.add_child(feature_groups[layer_name])
        
        # Get color based on theme and question type
        color = get_color_for_theme(theme, q_type)
        
        # Create popup with question text
        popup = folium.Popup(
            f"<b>Theme:</b> {theme}<br>"
            f"<b>Type:</b> {q_type}<br><br>"
            f"{row['question']}",
            max_width=300
        )
        
        # Add marker to the appropriate feature group
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6 if q_type == 'general' else 4,  # Larger for general questions
            popup=popup,
            tooltip=f"{theme} - {q_type}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            opacity=1
        ).add_to(feature_groups[layer_name])
    
    # Add layer control to the map
    layer_control.add_to(m)
    
    # Fit bounds to show all markers
    m.fit_bounds(m.get_bounds())
    
    return m

def load_multiple_question_files(file_paths):
    """Load questions from multiple files and combine them with source information."""
    all_questions = []
    
    for file_path in file_paths:
        try:
            df = load_questions(file_path)
            df['source'] = file_path.name
            all_questions.append(df)
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {str(e)}")
    
    if not all_questions:
        return pd.DataFrame()
    
    return pd.concat(all_questions, ignore_index=True)

def main():
    st.title("🌍 ClimSight Question Map")
    st.write("Visualize climate questions on an interactive map.")
    
    # File selection
    data_dir = Path(__file__).parent
    all_question_files = sorted([f for f in data_dir.glob('questions*.json')])
    
    if not all_question_files:
        st.error("No question files found. Please generate some questions first.")
        return
    
    # Multi-select for files
    selected_files = st.multiselect(
        "Select question files to display:",
        options=all_question_files,
        default=all_question_files[:1],  # Default to first file
        format_func=lambda x: x.name
    )
    
    if not selected_files:
        st.warning("Please select at least one question file.")
        return
    
    # Load questions from all selected files
    try:
        questions_df = load_multiple_question_files(selected_files)
        
        # Show stats
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", len(questions_df))
        with col2:
            st.metric("Themes", questions_df['theme'].nunique())
        with col3:
            st.metric("General Questions", len(questions_df[questions_df['type'] == 'general']))
        with col4:
            st.metric("Data Sources", len(selected_files))
        
        # Show file distribution if multiple files
        if len(selected_files) > 1:
            st.subheader("Questions by Source")
            source_counts = questions_df['source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Question Count']
            st.bar_chart(source_counts.set_index('Source'))
        
        # Create and display map
        st.subheader("Question Locations")
        st.write("Click on a marker to view the question. Use the layer control to show/hide themes.")
        
        m = create_map(questions_df)
        if m is not None:
            st_folium(m, use_container_width=True, height=600)
        
        # Show data table with expandable sections
        st.subheader("All Questions")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            selected_theme = st.selectbox(
                "Filter by theme:",
                ["All Themes"] + sorted(questions_df['theme'].unique().tolist())
            )
        with col2:
            selected_source = st.selectbox(
                "Filter by source:",
                ["All Sources"] + sorted(questions_df['source'].unique().tolist())
            )
        
        # Apply filters
        filtered_df = questions_df.copy()
        if selected_theme != "All Themes":
            filtered_df = filtered_df[filtered_df['theme'] == selected_theme]
        if selected_source != "All Sources":
            filtered_df = filtered_df[filtered_df['source'] == selected_source]
        
        st.dataframe(
            filtered_df[['theme', 'type', 'question', 'source']],
            column_config={
                "theme": "Theme",
                "type": "Type",
                "question": "Question",
                "source": "Source File"
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
    except Exception as e:
        st.error(f"Error loading questions: {str(e)}")

if __name__ == "__main__":
    main()
