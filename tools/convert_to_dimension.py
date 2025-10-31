import pandas as pd
import os

def convert_to_artwork_dimension(input_csv_path: str, artwork_csv_path: str, output_csv_path: str) -> None:
    """
    Convert APDD_enriched_split_text.csv to Artwork_DIMENSION.csv format.
    
    Each row in the input CSV represents one artwork with multiple dimension scores.
    Each dimension score becomes a separate row in the output CSV.
    """
    # Read the input files
    df_apdd = pd.read_csv(input_csv_path)
    # Read Artwork.csv with id:ID as string to preserve leading zeros
    df_artwork = pd.read_csv(artwork_csv_path, dtype={'id:ID': str})
    
    # Create a mapping from filename to artwork ID
    filename_to_id = dict(zip(df_artwork['filename:string'], df_artwork['id:ID']))
    
    # Define the dimension columns and their corresponding reason columns
    dimension_mapping = {
        'theme_and_logic': 'reason_for_theme_and_logic',
        'creativity': 'reason_for_creativity',
        'layout_and_composition': 'reason_for_layout_and_composition',
        'space_and_perspective': 'reason_for_space_and_perspective',
        'sense_of_order': 'reason_for_sense_of_order',
        'light_and_shadow': 'reason_for_light_and_shadow',
        'color': 'reason_for_color',
        'details_and_texture': 'reason_for_details_and_texture',
        'overall': 'reason_for_overall',
        'mood': 'reason_for_mood'
    }
    
    # Prepare output data
    output_rows = []
    
    for idx, row in df_apdd.iterrows():
        filename = row['filename']
        
        # Get artwork ID from filename
        if filename not in filename_to_id:
            print(f"Warning: Filename '{filename}' not found in Artwork.csv, skipping row {idx}")
            continue
            
        artwork_id = filename_to_id[filename]
        
        # Process each dimension
        for dimension, reason_col in dimension_mapping.items():
            level_value = row[dimension]
            reason_value = row[reason_col]
            
            # Skip if level is empty/NaN
            if pd.isna(level_value) or str(level_value).strip() == '' or str(level_value).strip() == 'nan':
                continue
            
            # Handle reason value
            if pd.isna(reason_value) or str(reason_value).strip() == '' or str(reason_value).strip() == 'nan':
                reason_str = '""'  # Empty string with quotes
            else:
                # Clean and quote the reason string
                reason_clean = str(reason_value).strip()
                reason_str = f'"{reason_clean}"'
            
            # Clean and quote the level string
            level_clean = str(level_value).strip()
            level_str = f'"{level_clean}"'
            
            # Create output row
            output_row = {
                ':START_ID': artwork_id,
                ':END_ID': dimension,
                'level:string': level_str,
                'reason:string': reason_str,
                ':TYPE': 'HAS_LEVEL'
            }
            output_rows.append(output_row)
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"Conversion completed!")
    print(f"Total input rows processed: {len(df_apdd)}")
    print(f"Total output rows created: {len(output_df)}")
    print(f"Output saved to: {output_csv_path}")
    
    # Show statistics by dimension
    print(f"\nStatistics by dimension:")
    dimension_counts = output_df[':END_ID'].value_counts()
    for dimension, count in dimension_counts.items():
        print(f"  {dimension}: {count} rows")
    
    # Show sample output
    print(f"\nSample output:")
    print(output_df.head(10).to_string(index=False))

if __name__ == "__main__":
    input_path = "APDD_enriched_split_text.csv"
    artwork_path = "Artwork.csv"
    output_path = "Artwork_DIMENSION.csv"
    
    # Check if input files exist
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        exit(1)
    
    if not os.path.exists(artwork_path):
        print(f"Error: Artwork file '{artwork_path}' not found!")
        exit(1)
    
    try:
        convert_to_artwork_dimension(input_path, artwork_path, output_path)
    except Exception as e:
        print(f"Error processing files: {e}")
