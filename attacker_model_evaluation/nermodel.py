import os
import csv
from transformers import pipeline

# ------------------------------------------------------
# Load pretrained NER model
# ------------------------------------------------------
ner_model = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

transcript_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1NewsampledTranscriptrate40"
output_csv_path = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/bert_entities_rate_40only.csv"

results = []

print("\n=== Extracting Entities using BERT NER ===\n")
print(f" Directory: {transcript_dir}\n")

# ------------------------------------------------------
# Process all transcript files
# ------------------------------------------------------
for file in os.listdir(transcript_dir):

    if not file.endswith(".txt"):
        continue

    transcript_path = os.path.join(transcript_dir, file)

    # Read transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    print(f" → Processing {file}")

    try:
        # Run NER
        ner_output = ner_model(transcript)

        # Extract entity words only
        entity_list = [ent["word"] for ent in ner_output]

        results.append({
            "filename": file,
            "entities": ", ".join(entity_list)
        })

    except Exception as e:
        print(f"  Error on {file}: {e}")
        results.append({
            "filename": file,
            "entities": ""
        })

# ------------------------------------------------------
# Save CSV
# ------------------------------------------------------
with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "entities"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✓ CSV saved successfully: {output_csv_path}")
print("✓ Completed.\n")
