import csv
from rapidfuzz import fuzz
from metaphone import doublemetaphone

# -----------------------------
# CONFIG & INPUTS
# -----------------------------
CSV_FILE = "/home/us80abag/whisper_output/transcript_stats/proper_nouns_mid_sentence.csv"
EDIT_THRESHOLD = 80
USE_PHONETIC = True

# ADD YOUR CANONICAL NAMES HERE
ACTUAL_LOCATIONS = [
    "Hanson Island",
    "Flower Island",
    "Parsons Island",
    "Cracroft Point",
    "Critical Point",
    "Blackfish Sound",
    "Johnstone Strait",
    "Blackney Pass",
    "Robson Bight",
    "Vancouver Island",
    "Swain Point",
    "Queen Charlotte Strait",
    "Telegraph Cove",
    "Alert Bay",
    "Kelsey Bay"
]

# -----------------------------
# HELPERS
# -----------------------------
def normalize(text):
    return text.lower().strip()

def phonetic_match(a, b):
    # Compares the first word of each term phonetically
    a_first = normalize(a).split()[0] if a else ""
    b_first = normalize(b).split()[0] if b else ""
    codes_a = set(doublemetaphone(a_first)) - {""}
    codes_b = set(doublemetaphone(b_first)) - {""}
    return len(codes_a & codes_b) > 0

# -----------------------------
# PROCESSING
# -----------------------------
# 1. Load CSV candidates
candidates = []
try:
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["term"].strip()
            if term:
                candidates.append(term)
except FileNotFoundError:
    print(f"Error: Could not find {CSV_FILE}")
    candidates = []

# 2. Build the Mapping
# This will store "Variant": "Canonical Name"
location_map = {}

# Ensure canonical names map to themselves first
for actual in ACTUAL_LOCATIONS:
    location_map[actual] = actual

for cand in candidates:
    cand_norm = normalize(cand)
    
    best_match = None
    highest_score = 0
    
    for actual in ACTUAL_LOCATIONS:
        actual_norm = normalize(actual)
        
        # Check Similarity
        edit_score = fuzz.ratio(actual_norm, cand_norm)
        p_match = phonetic_match(actual, cand) if USE_PHONETIC else False
        
        # If it's a strong match, track it
        if edit_score >= EDIT_THRESHOLD or p_match:
            # If multiple actual locations match, pick the one with the highest edit score
            if edit_score > highest_score:
                highest_score = edit_score
                best_match = actual
    
    if best_match:
        location_map[cand] = best_match

# -----------------------------
# OUTPUT GENERATION
# -----------------------------
output_file = "location_map_output.txt"
with open(output_file, "w") as f:
    f.write("LOCATION_MAP = {\n")
    
    # Sort keys alphabetically or by length for readability
    for variant in sorted(location_map.keys()):
        canonical = location_map[variant]
        # Escaping quotes for safety
        v_safe = variant.replace('"', '\\"')
        c_safe = canonical.replace('"', '\\"')
        f.write(f'    "{v_safe}": "{c_safe}",\n')
        
    f.write("}\n")

print(f"Success! Map with {len(location_map)} entries written to {output_file}")