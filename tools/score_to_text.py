import pandas as pd
import os

def score_to_description(score):
    """
    Convert numeric score to descriptive text based on the mapping rules.
    Returns None if score is NaN or empty.
    """
    if pd.isna(score) or score == '' or str(score).strip() == '':
        return None
    
    try:
        score_float = float(score)
    except (ValueError, TypeError):
        return None
    
    if score_float < 2:
        return "Abysmal"
    elif score_float >= 2 and score_float < 3:
        return "Horrendous"
    elif score_float >= 3 and score_float < 4:
        return "Poor"
    elif score_float >= 4 and score_float < 5:
        return "Below Average"
    elif score_float >= 5 and score_float < 6:
        return "Average"
    elif score_float >= 6 and score_float < 7:
        return "Good"
    elif score_float >= 7 and score_float < 8:
        return "Very Good"
    elif score_float >= 8 and score_float < 9:
        return "Excellent"
    elif score_float >= 9:
        return "Outstanding"
    else:
        return None

def replace_scores_with_text(input_csv_path: str, output_csv_path: str) -> None:
    """
    Replace numeric scores in specified columns with descriptive text.
    Skip rows where the original value is empty/NaN.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    
    # Define the score columns to replace
    score_columns = [
        'theme_and_logic',
        'creativity', 
        'layout_and_composition',
        'space_and_perspective',
        'sense_of_order',
        'light_and_shadow',
        'color',
        'details_and_texture',
        'overall',
        'mood'
    ]
    
    # Check which columns exist in the dataframe
    existing_columns = [col for col in score_columns if col in df.columns]
    missing_columns = [col for col in score_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following columns were not found: {missing_columns}")
    
    if not existing_columns:
        print("Error: None of the specified score columns were found in the CSV!")
        return
    
    print(f"Processing columns: {existing_columns}")
    
    # Replace scores with descriptions
    total_replacements = 0
    skipped_rows = 0
    
    for col in existing_columns:
        col_replacements = 0
        for idx, value in df[col].items():
            if pd.isna(value) or value == '' or str(value).strip() == '':
                skipped_rows += 1
                continue
            
            description = score_to_description(value)
            if description is not None:
                df.at[idx, col] = description
                col_replacements += 1
        
        total_replacements += col_replacements
        print(f"Column '{col}': {col_replacements} values replaced")
    
    # Save the modified DataFrame
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"\nProcessing complete!")
    print(f"Total replacements made: {total_replacements}")
    print(f"Empty values skipped: {skipped_rows}")
    print(f"Output saved to: {output_csv_path}")
    
    # Show sample of the converted data
    print(f"\nSample of converted data:")
    sample_cols = ['filename'] + existing_columns[:3]  # Show first 3 score columns
    print(df[sample_cols].head(5).to_string(index=False))

if __name__ == "__main__":
    input_path = "APDD_enriched_split.csv"
    output_path = "APDD_enriched_split_text.csv"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        exit(1)
    
    try:
        replace_scores_with_text(input_path, output_path)
    except Exception as e:
        print(f"Error processing file: {e}")
