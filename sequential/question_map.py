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
import yaml

# Set page config
st.set_page_config(
    page_title="ClimSight Question Map",
    page_icon="🌍",
    layout="wide"
)

def load_qa_yaml(file_path: Path) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    questions = []
    for qa in data.get('climsight', []):
        questions.append({
            'theme': 'Unknown',
            'type': 'general',
            'question': qa.get('question', ''),
            'lat': qa.get('lat', None),
            'lon': qa.get('lon', None),
            'answer': qa.get('answer', None),
            'source': file_path.name
        })
    return pd.DataFrame(questions)

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
                    'lon': q['lon'],
                    'answer': q.get('answer', None),
                    'source': file_path.name
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

def create_map(questions_df, answered_only=False):
    """Create a folium map with question markers, showing answer status visually."""
    if questions_df.empty:
        return None
    m = folium.Map(
        location=[questions_df['lat'].mean(), questions_df['lon'].mean()],
        zoom_start=2,
        tiles='cartodbpositron'
    )
    layer_control = folium.map.LayerControl(position='topright')
    feature_groups = {}
    for _, row in questions_df.iterrows():
        theme = row['theme']
        q_type = row['type']
        layer_name = f"{theme} ({q_type})"
        if layer_name not in feature_groups:
            feature_groups[layer_name] = folium.FeatureGroup(name=layer_name, show=True)
            m.add_child(feature_groups[layer_name])
        color = get_color_for_theme(theme, q_type)
        has_answer = row.get('answer') not in [None, '', 'nan']
        popup_html = (
            f"<b>Theme:</b> {theme}<br>"
            f"<b>Type:</b> {q_type}<br><br>"
            f"{row['question']}"
        )
        if has_answer:
            popup_html += "<br><b>Status:</b> Answer available"
        popup = folium.Popup(popup_html, max_width=300)
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6 if q_type == 'general' else 4,
            popup=popup,
            tooltip=f"{theme} - {q_type}",
            color="white" if has_answer else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if has_answer else 0.7,
            weight=2 if has_answer else 1,
            opacity=1
        ).add_to(feature_groups[layer_name])
    layer_control.add_to(m)
    m.fit_bounds(m.get_bounds())
    return m

def load_multiple_question_files(file_paths):
    """Load questions from multiple files and combine them with source information."""
    all_questions = []
    for file_path in file_paths:
        ext = str(file_path).lower()
        try:
            if ext.endswith('.yml') or ext.endswith('.yaml'):
                df = load_qa_yaml(file_path)
            else:
                df = load_questions(file_path)
            # Ensure all columns present
            for col in ['theme', 'type', 'question', 'lat', 'lon', 'answer', 'source']:
                if col not in df:
                    df[col] = None
            all_questions.append(df)
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {str(e)}")
    if not all_questions:
        return pd.DataFrame()
    return pd.concat(all_questions, ignore_index=True)

def main():
    st.title("🌍 ClimSight Question Map")
    st.write("Visualize climate questions on an interactive map.")
    data_dir = Path(__file__).parent
    qa_file = data_dir.parent / "evaluation" / "QA.yml"
    all_question_files = sorted([f for f in data_dir.glob('questions*.json')])
    if qa_file.exists():
        all_question_files = [qa_file] + all_question_files
    if not all_question_files:
        st.error("No question or QA files found. Please generate some questions first.")
        return
    # Default: QA.yml if present, else first JSON
    default_files = [qa_file] if qa_file.exists() else all_question_files[:1]
    selected_files = st.multiselect(
        "Select question files to display:",
        options=all_question_files,
        default=default_files,
        format_func=lambda x: x.name
    )
    st.markdown('<div style="font-size:1.1rem; color:#205080; font-weight:600; padding-bottom:0.5em;">Additional question sets are available in the <code>questions_xx.json</code> files.</div>', unsafe_allow_html=True)
    if not selected_files:
        st.warning("Please select at least one question file.")
        st.session_state.pop('selected_question', None)
        return
    # Load questions
    try:
        questions_df = load_multiple_question_files(selected_files)
        # Metrics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Questions", len(questions_df))
        with col2:
            st.metric("Themes", questions_df['theme'].nunique())
        with col3:
            st.metric("General Questions", len(questions_df[questions_df['type'] == 'general']))
        with col4:
            st.metric("Data Sources", len(selected_files))
        with col5:
            st.metric("Answered Questions", questions_df['answer'].notna().sum())
        # Filters
        st.sidebar.subheader("Filters")
        all_themes = ['All Themes'] + sorted(questions_df['theme'].unique())
        selected_theme = st.sidebar.selectbox("Filter by theme:", all_themes)
        all_types = ['All Types'] + sorted(questions_df['type'].unique())
        selected_type = st.sidebar.selectbox("Filter by type:", all_types)
        has_any_answers = questions_df['answer'].notna().any() and (questions_df['answer'] != '').any()
        if has_any_answers:
            answered_only = st.sidebar.checkbox("Answered only", value=False)
        else:
            answered_only = False
            st.session_state.pop('selected_question', None)
        # Apply filters
        filtered_df = questions_df.copy()
        if selected_theme != 'All Themes':
            filtered_df = filtered_df[filtered_df['theme'] == selected_theme]
        if selected_type != 'All Types':
            filtered_df = filtered_df[filtered_df['type'] == selected_type]
        if answered_only:
            filtered_df = filtered_df[filtered_df['answer'].notna() & (filtered_df['answer'] != '')]
        # Map
        st.subheader("Question Locations")
        m = create_map(filtered_df, answered_only=answered_only)
        map_data = st_folium(m, use_container_width=True, height=600, key="map")
        # Marker click/answer panel logic
        if map_data and map_data.get('last_object_clicked'):
            lat = map_data['last_object_clicked']['lat']
            lon = map_data['last_object_clicked']['lng']
            match = filtered_df[(filtered_df['lat'] == lat) & (filtered_df['lon'] == lon)]
            if not match.empty:
                st.session_state['selected_question'] = match.iloc[0]['question']
        if not has_any_answers or not selected_files or (qa_file not in selected_files and st.session_state.get('selected_question')):
            st.session_state.pop('selected_question', None)
        # Answer panel
        if has_any_answers and 'selected_question' in st.session_state:
            q_text = st.session_state['selected_question']
            row = filtered_df[filtered_df['question'] == q_text]
            if not row.empty and pd.notna(row.iloc[0]['answer']) and row.iloc[0]['answer'] != '':
                st.subheader("Answer")
                st.markdown(row.iloc[0]['answer'])
        # Data table
        st.subheader("Questions")
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
