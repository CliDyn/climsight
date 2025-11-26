"""
UI helper functions for ClimSight
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import textwrap
import os

def prepare_download_content(output, input_params, figs, data_pocket, references):
    """
    Prepare the complete report content for download including main report, 
    additional information, and references.
    """
    download_content = []
    
    # Add header
    download_content.append("=" * 80)
    download_content.append("CLIMSIGHT REPORT")
    download_content.append("=" * 80)
    download_content.append("")
    
    # Add location information
    download_content.append("LOCATION INFORMATION:")
    download_content.append(f"Coordinates: {input_params.get('lat', 'N/A')}, {input_params.get('lon', 'N/A')}")
    if 'location_str' in input_params:
        download_content.append(f"Address: {input_params['location_str']}")
    download_content.append("")
    download_content.append("-" * 80)
    download_content.append("")
    
    # Add main report
    download_content.append("ANALYSIS REPORT:")
    download_content.append("")
    download_content.append(output)
    download_content.append("")
    download_content.append("-" * 80)
    download_content.append("")
    
    # Add additional information if available
    download_content.append("ADDITIONAL INFORMATION:")
    download_content.append("")
    if 'elevation' in input_params:
        download_content.append(f"Elevation: {input_params['elevation']} m")
    if 'current_land_use' in input_params:
        download_content.append(f"Current land use: {input_params['current_land_use']}")
    if 'soil' in input_params:
        download_content.append(f"Soil type: {input_params['soil']}")
    if 'biodiv' in input_params:
        download_content.append(f"Occurring species: {input_params['biodiv']}")
    if 'distance_to_coastline' in input_params:
        download_content.append(f"Distance to shore: {round(float(input_params['distance_to_coastline']), 2)} m")
    
    # Add climate data summary if available
    if 'df_data' in data_pocket.df and data_pocket.df['df_data'] is not None:
        download_content.append("")
        download_content.append("Climate Data Summary:")
        download_content.append("(See generated plots for detailed visualizations)")
    
    download_content.append("")
    download_content.append("-" * 80)
    download_content.append("")
    
    # Add references
    download_content.append("REFERENCES:")
    download_content.append("")
    if references and 'used' in references:
        for ref in references['used']:
            download_content.append(f"- {ref}")
    download_content.append("")
    download_content.append("=" * 80)
    
    # Join all content
    return "\n".join(download_content)


def prepare_pdf_content(output, input_params, figs, data_pocket, references):
    """
    Prepare the report content as a PDF file.
    """
    import re

    def markdown_to_html(text):
        """Convert basic markdown to HTML for reportlab"""
        # Handle bold text **text** -> <b>text</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

        # Handle italic text *text* -> <i>text</i>
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)

        # Handle code blocks ```code``` -> <font name="Courier">code</font>
        text = re.sub(r'```(.+?)```', r'<font name="Courier">\1</font>', text, flags=re.DOTALL)

        # Handle inline code `code` -> <font name="Courier">code</font>
        text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)

        # Handle headers (convert to bold and larger font)
        text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

        # Handle bullet points (- or *)
        text = re.sub(r'^[\-\*]\s+(.+)$', r'• \1', text, flags=re.MULTILINE)

        # Handle numbered lists
        text = re.sub(r'^(\d+)\.\s+(.+)$', r'\1. \2', text, flags=re.MULTILINE)

        return text

    # Register Unicode-compatible font
    try:
        # Try to find and register a system font that supports Unicode
        font_registered = False

        # Try common system fonts that support multiple languages
        font_paths = []
        if os.path.exists('/System/Library/Fonts'):  # macOS
            font_paths.extend([
                '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                '/System/Library/Fonts/Supplemental/Arial.ttf',
                '/Library/Fonts/Arial Unicode.ttf',
            ])
        if os.path.exists('/usr/share/fonts'):  # Linux
            font_paths.extend([
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            ])
        if os.path.exists('C:\\Windows\\Fonts'):  # Windows
            font_paths.extend([
                'C:\\Windows\\Fonts\\arial.ttf',
                'C:\\Windows\\Fonts\\arialuni.ttf',
            ])

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('UniFont', font_path))
                    font_registered = True
                    break
                except:
                    continue

        # If no system font found, use reportlab's built-in Helvetica (limited Unicode support)
        if not font_registered:
            font_name = 'Helvetica'
        else:
            font_name = 'UniFont'
    except:
        # Fallback to Helvetica if font registration fails
        font_name = 'Helvetica'

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Create custom styles with Unicode font
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=24,
        textColor='darkblue',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=16,
        textColor='darkblue',
        spaceAfter=12
    )

    # Create normal style with Unicode font
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10
    )
    
    # Add title
    story.append(Paragraph("CLIMSIGHT REPORT", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Add location information
    story.append(Paragraph("LOCATION INFORMATION", heading_style))
    location_text = f"<b>Coordinates:</b> {input_params.get('lat', 'N/A')}, {input_params.get('lon', 'N/A')}<br/>"
    if 'location_str' in input_params:
        location_text += f"<b>Address:</b> {input_params['location_str']}<br/>"
    story.append(Paragraph(location_text, normal_style))
    story.append(Spacer(1, 0.3*inch))

    # Add main report
    story.append(Paragraph("ANALYSIS REPORT", heading_style))

    # Convert markdown to HTML and format the output text for PDF
    output_html = markdown_to_html(output)

    # Split into paragraphs and process each
    paragraphs = output_html.split('\n\n')
    for para in paragraphs:
        if para.strip():
            # Replace single newlines with breaks
            para = para.replace('\n', '<br/>')
            try:
                story.append(Paragraph(para, normal_style))
                story.append(Spacer(1, 0.1*inch))
            except:
                # If paragraph fails (due to complex formatting), add as plain text
                clean_para = re.sub('<.*?>', '', para)
                story.append(Paragraph(clean_para, normal_style))
                story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.2*inch))

    # Add additional information
    story.append(Paragraph("ADDITIONAL INFORMATION", heading_style))
    add_info = []
    if 'elevation' in input_params:
        add_info.append(f"<b>Elevation:</b> {input_params['elevation']} m")
    if 'current_land_use' in input_params:
        add_info.append(f"<b>Current land use:</b> {input_params['current_land_use']}")
    if 'soil' in input_params:
        add_info.append(f"<b>Soil type:</b> {input_params['soil']}")
    if 'biodiv' in input_params:
        add_info.append(f"<b>Occurring species:</b> {input_params['biodiv']}")
    if 'distance_to_coastline' in input_params:
        add_info.append(f"<b>Distance to shore:</b> {round(float(input_params['distance_to_coastline']), 2)} m")

    if add_info:
        story.append(Paragraph('<br/>'.join(add_info), normal_style))
    story.append(Spacer(1, 0.3*inch))

    # Add climate data summary if available
    if 'df_data' in data_pocket.df and data_pocket.df['df_data'] is not None:
        story.append(Paragraph("<b>Climate Data Summary:</b>", normal_style))
        story.append(Paragraph("(See generated plots for detailed visualizations)", normal_style))
        story.append(Spacer(1, 0.2*inch))

    # Add references
    if references and 'used' in references and references['used']:
        story.append(Paragraph("REFERENCES", heading_style))
        for ref in references['used']:
            # Clean reference text and convert markdown
            ref_html = markdown_to_html(ref)
            try:
                story.append(Paragraph(f"• {ref_html}", normal_style))
            except:
                # If reference fails, add as plain text
                clean_ref = re.sub('<.*?>', '', ref)
                story.append(Paragraph(f"• {clean_ref}", normal_style))
            story.append(Spacer(1, 0.05*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()