import pandas as pd
import os

def split_artistic_categories(input_csv_path: str, output_csv_path: str) -> None:
    """
    Split artistic_categories column into three separate columns:
    - painting_category (a)
    - artistic_style (b) 
    - subject_matter (c)
    
    Format: a*b*c where * is the separator
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    
    # Check if artistic_categories column exists
    if 'artistic_categories' not in df.columns:
        raise ValueError("Column 'artistic_categories' not found in the CSV file")
    
    # Initialize the new columns
    df['painting_category'] = None
    df['artistic_style'] = None
    df['subject_matter'] = None
    
    # Split the artistic_categories column
    for idx, row in df.iterrows():
        categories = str(row['artistic_categories'])
        
        # Skip if the value is NaN or empty
        if pd.isna(row['artistic_categories']) or categories.strip() == '' or categories == 'nan':
            continue
            
        # Split by '*' and strip whitespace
        parts = [part.strip() for part in categories.split('*')]
        
        # Assign parts to respective columns
        if len(parts) >= 1:
            df.at[idx, 'painting_category'] = parts[0]
        if len(parts) >= 2:
            df.at[idx, 'artistic_style'] = parts[1]
        if len(parts) >= 3:
            df.at[idx, 'subject_matter'] = parts[2]
    
    # Save the modified DataFrame
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Successfully split artistic_categories into three columns.")
    print(f"Output saved to: {output_csv_path}")
    
    # Print some statistics
    total_rows = len(df)
    non_empty_categories = df['artistic_categories'].notna().sum()
    print(f"Total rows: {total_rows}")
    print(f"Rows with artistic_categories: {non_empty_categories}")
    
    # Show sample of the split data
    print("\nSample of split data:")
    sample_cols = ['artistic_categories', 'painting_category', 'artistic_style', 'subject_matter']
    print(df[sample_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    input_path = "APDD_enriched.csv"
    output_path = "APDD_enriched_split.csv"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        exit(1)
    
    try:
        split_artistic_categories(input_path, output_path)
    except Exception as e:
        print(f"Error processing file: {e}")
