import pandas as pd
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def create_cnn_models_table():
    # CSV data
    csv_data = '''Model Type,Parameters,Key Pros,Key Cons,FPGA Suitability,Recommended Use Case
Lightweight 1D CNN,"10,000–20,000","- Extremely lightweight\n- Fast inference\n- Good for local pattern capture","- May struggle with complex seizure patterns\n- Limited spatial relationship learning","Excellent (small FPGAs like Artix-7)","Start with small FPGAs or new to CNN-FPGA integration"
Multi-Channel 1D CNN,"50,000–100,000","- Processes channels independently\n- Captures channel-specific features","- Higher resource demand\n- May miss cross-channel correlations","Good (mid-sized FPGAs like Spartan-7)","Improved accuracy with moderate FPGA resources"
2D CNN with Channel-Time Representation,"100,000–200,000","- Captures spatial and temporal patterns\n- Effective in EEG classification","- Computationally intensive\n- Requires more FPGA resources","Feasible on larger FPGAs (Kintex-7)","Prioritize accuracy with larger FPGA"
Temporal Convolutional Network (TCN),"50,000–150,000","- Captures long-term EEG dependencies\n- Easier to parallelize than RNNs","- Slightly complex\n- Dilation logic can complicate implementation","Good for mid-to-large FPGAs","Seizures with prolonged pre-ictal patterns"
Pre-Trained Model (EEGNet),"20,000–50,000","- Compact, EEG-specific design\n- Low parameter count\n- Widely validated","- Requires dataset tuning\n- Depthwise ops can be tricky","Excellent (low parameter, optimized for EEG)","Proven, efficient model with good accuracy"'''

    # Read CSV data
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Create PDF
    def export_to_pdf(dataframe, filename='CNN_Models_Comparison.pdf'):
        # Use landscape orientation for wider page
        pdf = SimpleDocTemplate(filename, pagesize=landscape(letter))
        
        # Prepare styles
        styles = getSampleStyleSheet()
        
        # Custom styles for table content
        left_style = ParagraphStyle(
            'LeftAlign',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            fontSize=8,
            leading=10
        )
        center_style = ParagraphStyle(
            'CenterAlign',
            parent=styles['Normal'],
            alignment=TA_CENTER,
            fontSize=8,
            leading=10
        )
        
        # Prepare table data with paragraphs
        data = []
        # Add headers
        header = [Paragraph(str(col), center_style) for col in dataframe.columns]
        data.append(header)
        
        # Add rows
        for _, row in dataframe.iterrows():
            pdf_row = []
            for val in row:
                # Replace newlines with HTML line breaks for proper rendering
                val_str = str(val).replace('\n', '<br/>')
                # Use left alignment for text-heavy columns, center for others
                style = left_style if len(val_str) > 20 else center_style
                pdf_row.append(Paragraph(val_str, style))
            data.append(pdf_row)
        
        # Create table with adjusted column widths
        table = Table(data, 
            colWidths=[1*inch, 0.75*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch],
            repeatRows=1
        )
        
        # Table style
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        
        # Title
        title = Paragraph("CNN Models Comparison for Seizure Detection", 
                          styles['Title'])
        
        # Build PDF
        pdf_elements = [title, table]
        pdf.build(pdf_elements)
        
        print(f"PDF exported successfully to {filename}")
    
    # Export to PDF
    export_to_pdf(df)
    
    return df

# Run the function
df = create_cnn_models_table()