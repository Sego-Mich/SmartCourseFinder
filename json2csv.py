'''import json
import csv

# Load JSON file
with open("degree.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Open CSV file for writing
with open("degrees.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Define CSV header
    header = [
        "degree",
        "Cluster Subject 1",
        "Cluster Subject 2",
        "Cluster Subject 3",
        "Cluster Subject 4",
        "Subject 1 Requirement",
        "Subject 2 Requirement",
        "Institution",
        "Programme"
    ]
    writer.writerow(header)

    # Write rows
    for item in data:
        row = [
            item.get("degree", ""),
            item.get("minimum_entry_requirements", {}).get("Cluster Subject 1", ""),
            item.get("minimum_entry_requirements", {}).get("Cluster Subject 2", ""),
            item.get("minimum_entry_requirements", {}).get("Cluster Subject 3", ""),
            item.get("minimum_entry_requirements", {}).get("Cluster Subject 4", ""),
            # Flatten subject requirements
            next(iter(item.get("minimum_subject_requirements", {}).get("Subject 1", {}).items()), ("", "")),
            next(iter(item.get("minimum_subject_requirements", {}).get("Subject 2", {}).items()), ("", "")),
            next(iter(item.get("programmes", {}).keys()), ""),
            next(iter(item.get("programmes", {}).values()), "")
        ]

        # Format Subject Requirements as "Subject: Grade"
        subject1 = f"{row[5][0]}: {row[5][1]}" if row[5][0] else ""
        subject2 = f"{row[6][0]}: {row[6][1]}" if row[6][0] else ""

        writer.writerow([
            row[0], row[1], row[2], row[3], row[4],
            subject1, subject2, row[7], row[8]
        ])

print("✅ Conversion complete! File saved as degrees.csv")'''
import pandas as pd

# Load CSV files
degrees_df = pd.read_csv("degrees.csv")
courses_df = pd.read_csv("university_courses.csv")

# Perform an inner join on the 'degree' column
merged_df = pd.merge(degrees_df, courses_df[['id', 'course_id', 'program_code', 'institution_name', 'degree', '2015',
       '2016', '2017', '2018', '2019', '2020', '2021']], on="degree", how="inner")

# Save merged file
merged_df.to_csv("programmes.csv", index=False, encoding="utf-8")

print("✅ Merged file saved as merged_degrees_courses.csv")

print(merged_df.info())
print(courses_df.duplicated().sum())