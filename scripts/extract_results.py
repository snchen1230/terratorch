from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

def extract_tensorboard_data(logdir):
    # Load the tensorboard data
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Get list of all available tags
    tags = event_acc.Tags()['scalars']
    
    # Create a list to store all data
    all_data = []
    
    # Extract scalar values for each tag
    for tag in tags:
        events = event_acc.Scalars(tag)
        
        # Convert events to rows
        for event in events:
            row = {
                'tag': tag,
                'step': event.step,
                'value': event.value,
                'wall_time': event.wall_time
            }
            all_data.append(row)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(all_data)
    
    # Pivot the data to have tags as columns
    if not df.empty:
        df_pivot = df.pivot_table(
            index=['step', 'wall_time'],
            columns='tag',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        return df_pivot
    return pd.DataFrame()

if __name__ == "__main__":
    logdir = "../logs/od_vhr10_dofav1_hpo/od_vhr10_dofav1_lr_0.0001_wd_0.005_warmup_2_e40/version_1"
    
    # Check if the directory exists
    if not os.path.exists(logdir):
        print(f"Error: Log directory not found: {logdir}")
        exit(1)
        
    # Extract data
    results_df = extract_tensorboard_data(logdir)
    
    if results_df.empty:
        print("Warning: No data was extracted from the tensorboard logs")
    else:
        # Save to CSV
        output_file = 'tensorboard_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Data successfully extracted to {output_file}")
        print(f"Number of rows: {len(results_df)}")
        print(f"Columns: {', '.join(results_df.columns)}")
