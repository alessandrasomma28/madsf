import os

def check_xml_in_deep_subfolders(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        # Skip the root level — only check deeper subfolders
        if dirpath == root_dir:
            continue
        
        # Check if there's at least one .xml file in the current folder
        has_xml = any(fname.lower().endswith('.html') for fname in filenames)
        
        if not has_xml:
            print(f"❌ No XML found in: {dirpath}")

# Example usage
check_xml_in_deep_subfolders('/Users/beyzaeken/Desktop/sfdigitalmirror/sumoenv/scenarios')
'''
import pandas as pd

df = pd.read_csv('/Users/beyzaeken/Desktop/sfdigitalmirror/sumoenv/scenarios/normal/social_groups/21111008_21111009/sf_final_metrics.csv')
rc = df['passengers_cancel'].sum()
print(rc)
'''