import streamlit as st
import random
import pandas as pd
import math
import re

# ------------------------------
# Configuration / Constants
# ------------------------------
REQUIRED_COLUMNS = ['key', 'institution', 'programme_name', 'cluster', 'pred_cutoff_2025']

compulsory_subjects = ['ENG', 'KISW', 'CHEM', 'BIO']
math_variants = ['MAT A', 'MAT B']
other_subjects = [
    'PHY', 'AGR', 'HSC', 'HIST', 'GEO', 'CRE', 'ART',
    'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HAG', 'FRE', 'GER', 'MUS'
]

grade_points = {
    'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8,
    'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
}

cluster_subjects = {
    1: ["ENG/KIS", "MAT A/B", "BIO/PHY/CHE", "GRP III/IV/V"],
    2: ["ENG/KIS", "ENG/KIS", "MAT A/B", "GRP II/III/IV/V"],
    3: ["ENG/KIS", "MAT A/B or GRP II", "GRP III", "GRP II/III/IV/V"],
    4: ["MAT A", "PHY", "CHE/BIO/GEO", "GRP II/III/IV/V"],
    5: ["MAT A", "PHY", "CHE/BIO/GEO", "GRP II/III/IV/V"],
    6: ["MAT A", "PHY", "GRP III", "GRP II/III/IV/V"],
    7: ["MAT A", "PHY", "BIO/CHE/GEO", "GRP II/III/IV/V"],
    8: ["MAT A", "BIO", "PHY/CHE", "GRP II/III/IV/V"],
    9: ["MAT A", "GRP II", "2nd GRP II", "GRP II/III/IV/V"],
    10:["MAT A", "GRP II", "GRP III", "GRP II/III/IV/V"],
    11:["CHE", "MAT A/B or PHY", "BIO/HSC", "ENG/KIS or GRP III/IV/V"],
    12:["BIO/HSC", "MAT A/B", "GRP II/III", "ENG/KIS or GRP II/III/IV/V"],
    13:["BIO", "CHE", "MAT A or PHY", "ENG/KIS or GRP II/III/IV/V"],
    14:["HIST/GEO", "ENG/KIS", "MAT A/B or GRP II", "GRP II/III/IV/V"],
    15:["BIO", "CHE", "MAT A/PHY/GEO", "ENG/KIS or GRP II/III/IV/V"],
    16:["GEO", "MAT A/B", "GRP II", "GRP II/III/IV/V"],
    17:["FRE/GER", "ENG/KIS", "MAT A/B or GRP II/III", "GRP II/III/IV/V"],
    18:["MUS", "ENG/KIS", "MAT A/B or GRP II/III", "GRP II/III/IV/V"],
    19:["ENG", "MAT A/B or GRP II", "2nd GRP II", "KIS or GRP II/III/IV/V"],
    20:["CRE/IRE/HRE", "ENG/KIS", "2nd GRP III", "GRP II/IV/V"],
}

group_definitions = {
    'ENG/KIS': ['ENG', 'KISW'],
    'MAT A/B': ['MAT A', 'MAT B'],
    'MAT A/B or GRP II': ['MAT A', 'MAT B', 'BIO', 'PHY', 'CHE', 'AGR'],
    'MAT A or PHY': ['MAT A', 'PHY'],
    'GRP II': ['BIO', 'PHY', 'CHE', 'AGR'],
    'GRP III': ['HIST', 'GEO', 'CRE', 'IRE', 'HRE'],
    'GRP IV': ['BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD'],
    'GRP V': ['HSC', 'ART', 'HAG', 'MUS'],
    'BIO/PHY/CHE': ['BIO', 'PHY', 'CHE'],
    'GRP II/III/IV/V': ['BIO','PHY','CHE','AGR','HIST','GEO','CRE','IRE','HRE','BS','COMP','AVT','ELEC','MET','WOOD','HSC','ART','HAG','MUS'],
    'ENG/KIS or GRP III/IV/V': ['ENG','KISW','HIST','GEO','CRE','IRE','HRE','BS','COMP','AVT','ELEC','MET','WOOD','HSC','ART','HAG','MUS'],
    'ENG/KIS or GRP II/III/IV/V': ['ENG','KISW','BIO','PHY','CHE','AGR','HIST','GEO','CRE','IRE','HRE','BS','COMP','AVT','ELEC','MET','WOOD','HSC','ART','HAG','MUS'],
    'KIS or GRP II/III/IV/V': ['KISW','BIO','PHY','CHE','AGR','HIST','GEO','CRE','IRE','HRE','BS','COMP','AVT','ELEC','MET','WOOD','HSC','ART','HAG','MUS'],
    'BIO/HSC': ['BIO', 'HSC'],
    '2nd GRP II': ['BIO','PHY','CHE','AGR'],
    '2nd GRP III': ['HIST','GEO','CRE','IRE','HRE'],
    'PHY/CHE': ['PHY','CHE'],
    'CHE/BIO/GEO': ['CHE','BIO','GEO'],
    'BIO/CHE/GEO': ['BIO','CHE','GEO'],
    'FRE/GER': ['FRE','GER'],
    'MUS': ['MUS'],
    'GEO': ['GEO'],
    'HIST/GEO': ['HIST', 'GEO'],
    'CRE/IRE/HRE': ['CRE', 'IRE', 'HRE'],
    'MAT A/PHY/GEO': ['MAT A', 'PHY', 'GEO'],
    'MAT A/B or GRP II/III': ['MAT A', 'MAT B', 'BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE'],
    'GRP II/III': ['BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE'],
}

field_to_clusters = {
    "law": [1],
    "business": [2],
    "commerce": [2],
    "accounting": [2],
    "finance": [2],
    "economics": [2, 10],
    "social science": [3, 14, 20],
    "arts": [3, 14, 17, 18, 20],
    "humanities": [3, 14, 17, 18, 20],
    "history": [14],
    "philosophy": [3, 20],
    "psychology": [3],
    "languages": [17],
    "music": [18],
    "religion": [20],
    "theology": [20],
    "communication": [3],
    "journalism": [3],
    "education": [19],
    "teaching": [19],
    "science": [9, 10],
    "mathematics": [10],
    "physics": [9],
    "chemistry": [9],
    "biology": [9],
    "statistics": [10],
    "actuarial": [10],
    "engineering": [5],
    "civil engineering": [5],
    "mechanical engineering": [5],
    "electrical engineering": [5],
    "mechatronics": [5],
    "technology": [5, 7],
    "it": [7],
    "ict": [7],
    "computer science": [7],
    "computing": [7],
    "information technology": [7],
    "software": [7],
    "data science": [7, 10],
    "health": [11, 12, 13],
    "medicine": [13],
    "nursing": [13],
    "clinical": [13],
    "medical": [13],
    "pharmacy": [13],
    "public health": [13],
    "nutrition": [13],
    "dietetics": [13],
    "sport science": [12],
    "sport":[12],
    "occupational therapy": [13],
    "physiotherapy": [13],
    "biomedical": [9, 13],
    "agriculture": [8, 15],
    "agribusiness": [8],
    "agronomy": [15],
    "veterinary": [13],
    "environmental science": [15],
    "soil science": [15],
    "forestry": [15],
    "architecture": [6],
    "landscape architecture": [6],
    "urban planning": [6],
    "quantity surveying": [6],
    "construction management": [6],
    "geoscience": [4],
    "geology": [4],
    "earth science": [4],
    "meteorology": [4],
    "environment": [4, 15],
    "interior design": [11],
    "design": [11],
    "fine art": [18],
    "graphic design": [11, 18],
    "applied art": [18],
}



# ------------------------------
# Data loading & validation
# ------------------------------
def load_programmes(path="degree_req_cutoff.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'key' in df.columns:
        df['key'] = df['key'].astype(str).str.strip().str.lower()
    if 'pred_cutoff_2025' in df.columns:
        df['pred_cutoff_2025'] = pd.to_numeric(df['pred_cutoff_2025'], errors='coerce')
    return df

def validate_programmes_df(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing
def resolve_institution_alias(alias_query: str, df: pd.DataFrame):
    """
    Resolve an institution alias (like 'jkuat', 'uon', 'ku') using df['key'].
    Returns a list of matching institution names.
    """
    alias_query = alias_query.strip().lower()
    matched_institutions = set()

    # Try to match alias against df['key']
    alias_matches = df[df['key'].astype(str).str.lower().str.contains(rf'\b{re.escape(alias_query)}\b', na=False)]

    if not alias_matches.empty:
        matched_institutions.update(alias_matches['institution'].str.strip().unique())

    return list(matched_institutions)

try:
    programmes_df = load_programmes()
except FileNotFoundError:
    st.error("CSV file 'degree_req_cutoff.csv' not found. Please place it in the app folder.")
    st.stop()

if 'key' in programmes_df.columns and 'institution' in programmes_df.columns:
    key_to_institution = dict(
        zip(
            programmes_df['key'].astype(str).str.lower().str.strip(),
            programmes_df['institution'].astype(str).str.strip()
        )
    )
else:
    key_to_institution = {} 

if 'county' not in programmes_df.columns:
    programmes_df['county'] = ''
if 'institution_type' not in programmes_df.columns:
    programmes_df['institution_type'] = ''
KENYAN_COUNTIES = [
    "nairobi", "kiambu", "nakuru", "kisumu", "mombasa", "eldoret",
    "nyeri", "machakos", "meru", "embu", "kericho", "bomet",
    "kakamega", "bungoma", "muranga", "tharaka nithi", "nyandarua",
    "kajiado", "kitui", "laikipia", "garissa", "isiolo", "migori",
    "homa bay", "siaya", "busia", "vihiga", "marsabit", "mandera",
    "wajir", "samburu", "narok", "trans nzoia", "uasin gishu"
]

INSTITUTION_TYPE_KEYWORDS = {
    "public": ["public university", "public", "government"],
    "private": ["private university", "private"]
}


missing_cols = validate_programmes_df(programmes_df)
if missing_cols:
    st.error(f"Missing required columns in CSV: {missing_cols}")
    st.stop()

programmes_df['cluster'] = programmes_df['cluster'].astype(int)
all_institutions = programmes_df['institution'].dropna().unique().tolist()

# ------------------------------
# Utility functions
# ------------------------------
def expand_group_option(opt_str):
    """Given an option token, expand via group_definitions or return token itself."""
    opt = opt_str.strip()
    return group_definitions.get(opt, [opt])

def calculate_cluster_points_per_group(subjects_list, grades_dict):
    """Calculate cluster points based on best 4 subjects per cluster requirements."""
    cluster_points = {}
    for cluster_id, cluster_reqs in cluster_subjects.items():
        
        selected_scores = []
        for req in cluster_reqs:
            if ' or ' in req:
                options = req.split(' or ')
            elif '/' in req:
                options = req.split('/')
            else:
                options = [req]
            
            expanded_options = []
            for opt in options:
                expanded_options += expand_group_option(opt)
           
            best_score = -1
            for subj in expanded_options:
                if subj in grades_dict:
                    
                    user_grade = grades_dict.get(subj)
                    if user_grade is None:
                        continue
                    score = grade_points.get(user_grade, 0)
                    if score > best_score:
                        best_score = score
            
            if best_score >= 0:
                selected_scores.append(best_score)
        
        if len(selected_scores) >= 4:
            best_4 = sorted(selected_scores, reverse=True)[:4]
            cluster_points[cluster_id] = float(sum(best_4))
        else:
            cluster_points[cluster_id] = None
    
    return cluster_points

def normalize_institution_name(name):
    """Normalize institution name for better matching."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower()

def match_institution_from_query(text_lower):
    """Match institution from query text using df.key instead of static aliases."""
    matched = set() 
    # 1️⃣ Check using df['key']
    for key, inst in key_to_institution.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            matched.add(inst)

    # 2️⃣ Fallback: partial name matching on full institution name
    if not matched:
        for inst in all_institutions:
            inst_normalized = normalize_institution_name(inst)
            if inst_normalized and text_lower.find(inst_normalized[:inst_normalized.find("university")])>-1 and inst_normalized.find("university")>0:
                matched.add(inst)   
    return list(matched)


def detect_intent(text, institution_list, field_to_clusters, df):
    text_lower = text.lower().strip()
    matched_institutions = match_institution_from_query(text_lower)
    matched_fields = []

    # Detect fields
    sorted_fields = sorted(field_to_clusters.keys(), key=len, reverse=True)
    for field in sorted_fields:
        if field.lower() in text_lower:
            matched_fields.append(field)

    # Detect counties
    matched_counties = [c for c in KENYAN_COUNTIES if c in text_lower]

    # Detect institution type
    matched_types = []
    for inst_type, keywords in INSTITUTION_TYPE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            matched_types.append(inst_type)

    # Intent resolution
    if matched_institutions or matched_fields or matched_counties or matched_types:
        return 'recommend_programme', matched_fields, matched_institutions, matched_counties, matched_types

    return 'general', None, None, None, None


def get_cluster_name(cluster_id):
    """Return human-readable cluster name."""
    cluster_names = {
        1: "Law", 2: "Business", 3: "Social Sciences", 4: "Geosciences",
        5: "Engineering", 6: "Architecture", 7: "Computing", 8: "Agribusiness",
        9: "General Science", 10: "Actuarial/Economics", 11: "Interior Design",
        12: "Sport Science", 13: "Medicine", 14: "History", 15: "Agriculture",
        16: "Geography", 17: "Languages", 18: "Music", 19: "Education",
        20: "Religious Studies"
    }
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")
def expand_subject_token(token: str):
    """Normalize and expand KUCCPS subject requirement tokens into explicit subject names."""
    token = str(token).strip().upper()
    if not token or token in ["-", "C+", "ANY"]:
        return []

    # Normalize underscores and alternative labels
    token = token.replace("ALTERNATIVE_", "")
    token = token.replace("MAT_", "MAT ")
    token = token.replace("A_GROUP", "ANY_GROUP")
    token = token.replace("ANYGROUP", "ANY_GROUP")
    token = token.strip()

    # Handle slashes (e.g. "BIO/CHE/GEO")
    if "/" in token:
        parts = [p.strip() for p in token.split("/") if p.strip()]
        subjects = []
        for p in parts:
            subjects += expand_subject_token(p)
        return list(set(subjects))

    # Define KUCCPS group mappings
    GROUPS = {
        "ANY_GROUP_I": ["ENG", "KISW", "MAT A", "MAT B"],
        "ANY_GROUP_II": ["BIO", "CHEM", "PHY", "AGR"],
        "ANY_GROUP_III": ["HIST", "GEO", "CRE", "HSC", "IRE", "HRE"],
        "ANY_GROUP_IV": ["AGR", "BS", "COMP", "HAG", "ART", "AVT", "ELEC", "MET", "WOOD"],
        "ANY_GROUP_V": ["HSC", "ART", "HAG", "MUS"],
        "2ND_GROUP_II": ["BIO", "CHEM", "PHY", "AGR"],
        "3RD_GROUP_II": ["BIO", "CHEM", "PHY", "AGR"],
        "2ND_GROUP_III": ["HIST", "GEO", "CRE", "HSC", "IRE", "HRE"],
        "3RD_GROUP_III": ["HIST", "GEO", "CRE", "HSC", "IRE", "HRE"],
    }

    # Handle "any_GROUP_X" etc.
    for key, subjects in GROUPS.items():
        if key in token:
            return subjects

    # Handle ENG/KIS
    if "ENG/KIS" in token or "ENG OR KIS" in token:
        return ["ENG", "KISW"]

    # Handle math variants
    if "MAT ALTERNATIVE A" in token or token == "MAT A":
        return ["MAT A"]
    if "MAT ALTERNATIVE B" in token or token == "MAT B":
        return ["MAT B"]

    # Handle BIO/GSC style
    if token in ["BIO/GSC"]:
        return ["BIO","GSC"]

    # Direct aliases
    aliases = {
        "KIS": "KISW",
        "CHE": "CHEM",
        "PHY": "PHY",
        "BIO": "BIO",
        "HSC": "HSC",
        "AGR": "AGR",
        "GEO": "GEO",
        "ENG": "ENG",
    }

    token = aliases.get(token, token)
    return [token]


def recommend_programmes(subjects, grades, programmes_df, margin=2, top_n=15,
                        allowed_clusters=None, fields=None, institutions=None,
                        counties=None, institution_types=None):
    cluster_pts = calculate_cluster_points_per_group(subjects, grades)
    recommendations = []
    seen_programmes = set()

    filtered_df = programmes_df.copy()

    # --- Institution filtering ---
    if institutions:
        institution_mask = pd.Series([False] * len(filtered_df))
        alias_found = False
        for inst in institutions:
            inst_normalized = normalize_institution_name(inst)
            alias_rows = filtered_df[
                filtered_df['key'].astype(str).str.lower().str.contains(
                    rf'\b{re.escape(inst_normalized)}\b', na=False, regex=True
                )
            ]
            if not alias_rows.empty:
                alias_found = True
                matched_institutions = alias_rows['institution'].str.strip().unique()
                for matched_inst in matched_institutions:
                    mask = filtered_df['institution'].str.lower().str.contains(
                        normalize_institution_name(matched_inst)
                    )
                    institution_mask = institution_mask | mask
                break
        if not alias_found:
            for inst in institutions:
                inst_normalized = normalize_institution_name(inst)
                mask = filtered_df['institution'].str.lower().str.contains(inst_normalized)
                institution_mask = institution_mask | mask
        filtered_df = filtered_df[institution_mask]

    # --- County filtering ---
    if counties: 
        county_mask = filtered_df['location'].str.lower().isin([c+" county".lower() for c in counties])
        filtered_df = filtered_df[county_mask]

    # --- Institution type filtering ---
    if institution_types:
        type_mask = filtered_df['institution_type'].str.lower().isin(institution_types)
        filtered_df = filtered_df[type_mask]

    # --- Determine target clusters ---
    if allowed_clusters:
        target_clusters = allowed_clusters
    elif fields:
        # Map field keywords to clusters
        target_clusters = []
        for f in fields:
            target_clusters += field_to_clusters.get(f, [])
        target_clusters = list(set(target_clusters))
    else:
        target_clusters = []

    # --- Generate recommendations ---
    for _, row in filtered_df.iterrows():
        cluster_id = row['cluster']

        # Skip if cluster not in target clusters (if specified)
        if target_clusters and cluster_id not in target_clusters:
            continue

        cutoff = row['pred_cutoff_2025']
        user_points = cluster_pts.get(cluster_id)

        if user_points is None or pd.isna(cutoff):
            continue

        # Check if user qualifies (with margin)
        if cutoff <= user_points + margin:
            prog_key = (row['programme_name'], row['institution'], cluster_id, cutoff)
            if prog_key in seen_programmes:
                continue
            seen_programmes.add(prog_key)

            points_above_cutoff = user_points - cutoff
            priority_score = (points_above_cutoff * 100) - (cluster_id * 0.1)

            rec_text = (
                f"{row['programme_name']} at {row['institution']} "
                f"(Cluster {cluster_id}: {get_cluster_name(cluster_id)}, "
                f"Cutoff: {cutoff:.1f} pts, Your score: {user_points:.1f} pts)"
            )

            recommendations.append((priority_score, rec_text, cluster_id, cutoff))

    recommendations.sort(key=lambda x: x[0], reverse=True)

    # --- Build feedback ---
    if not recommendations and target_clusters:
        qualifying_clusters = [c for c, pts in cluster_pts.items() if pts is not None]
        feedback = []
        for cluster_id in target_clusters:
            pts = cluster_pts.get(cluster_id)
            if pts is None:
                feedback.append(f"❌ {get_cluster_name(cluster_id)} (Cluster {cluster_id}): "
                                f"You don't meet the subject requirements")
            else:
                cluster_programmes = filtered_df[filtered_df['cluster'] == cluster_id]
                if not cluster_programmes.empty:
                    min_cutoff = cluster_programmes['pred_cutoff_2025'].min()
                    feedback.append(f"⚠️ {get_cluster_name(cluster_id)} (Cluster {cluster_id}): "
                                    f"Your score is {pts:.1f} pts, minimum cutoff is {min_cutoff:.1f} pts")
        return [], "\n".join(feedback)

    return [rec[1] for rec in recommendations[:top_n]], None


def institution_info(institution_names):
    """Get information about specific institutions."""
    if not institution_names:
        return "Please specify a university or institution name."

    clean_df = programmes_df.dropna(subset=['institution'])
    results = []
    
    for inst in institution_names:
        inst_normalized = normalize_institution_name(inst)
        matches = clean_df[clean_df['institution'].str.lower().str.contains(
            re.escape(inst_normalized), na=False, regex=True
        )]
        
        if matches.empty:
            results.append(f"Sorry, I couldn't find any programmes for **{inst}**.")
        else:
            actual_name = matches.iloc[0]['institution']
            programs_list = matches['programme_name'].dropna().unique()
            clusters = sorted(matches['cluster'].unique())
            
            result_text = f"📚 **{actual_name}**\n\n"
            result_text += f"Offers **{len(programs_list)}** programmes across **{len(clusters)}** clusters "
            result_text += f"({', '.join(map(str, clusters))})\n\n"
            result_text += "**Sample Programmes:**\n- " + "\n- ".join(programs_list[:15])
            
            if len(programs_list) > 15:
                result_text += f"\n\n... and {len(programs_list) - 15} more programmes."
            
            results.append(result_text)
    
    return "\n\n---\n\n".join(results)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="KUCCPS Recommender", layout="wide")
st.title("🎲 KUCCPS Degree & Institution Recommender Bot")

# --- Subject setup for random profile generation ---
science_subjects = ['BIO', 'PHY']  # Exclude CHEM because it's compulsory
humanities_subjects = ['HSC', 'HIST', 'GEO', 'CRE', 'ART']
technical_subjects = ['AGR', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HAG']

allowed_grades_for_generation = ['A', 'A-']

# --- Generate Random Profile ---
if st.button("🎲 Generate Random KUCCPS Profile"):
    chosen_subjects = ['ENG', 'KISW', 'MAT A']

    # Add 2 or all 3 of BIO, CHEM, PHY
    science_combo = random.sample(['BIO', 'CHE', 'PHY'], k=random.choice([2, 3]))
    chosen_subjects += science_combo

    # Fill remaining slots
    remaining_slots = 8 - len(chosen_subjects)
    remaining_subject_pool = [s for s in other_subjects if s not in chosen_subjects]
    chosen_subjects += random.sample(remaining_subject_pool, remaining_slots)

    # Assign grades
    user_grades = {subj: random.choice(allowed_grades_for_generation) for subj in chosen_subjects}

    # Save in session
    st.session_state['user_profile'] = (chosen_subjects, user_grades)
    st.session_state['chat_history'] = []
    st.success("✅ Random KUCCPS profile generated! You can now ask the bot questions below.")

# --- Require profile before continuing ---
if 'user_profile' not in st.session_state:
    st.info("Please generate a random KUCCPS profile first.")
    st.stop()

subjects, user_grades = st.session_state['user_profile']

# --- Display user profile ---
st.subheader("🎓 Current KUCCPS Profile")
cols = st.columns(min(8, len(subjects)))
for idx, subj in enumerate(subjects):
    with cols[idx]:
        st.markdown(f"**{subj}**")
        st.markdown(f"*Grade: {user_grades[subj]}*")

# --- Optional cluster point display ---
if st.checkbox("📊 Show Cluster Points for All Groups"):
    cluster_pts = calculate_cluster_points_per_group(subjects, user_grades)
    st.subheader("📈 Your Cluster Points (sum of best 4)")

    num_clusters = len(cluster_pts)
    num_cols = 5
    rows = (num_clusters + num_cols - 1) // num_cols
    cluster_items = list(cluster_pts.items())

    idx = 0
    for _ in range(rows):
        cols_row = st.columns(num_cols)
        for col in cols_row:
            if idx < len(cluster_items):
                cluster_id, pts = cluster_items[idx]
                cluster_name = get_cluster_name(cluster_id)
                if pts is not None:
                    col.markdown(f"**{cluster_name} (Cluster {cluster_id})**: {pts} pts")
                else:
                    col.markdown(f"**{cluster_name} (Cluster {cluster_id})**: N/A")
                idx += 1


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input(
    "Ask me about KUCCPS degrees & institutions (e.g., 'law', 'engineering at UoN', 'jkuat', 'medicine'): "
)

def respond_to_user(message):
    if 'user_profile' not in st.session_state:
        return "Please generate a KUCCPS profile first."

    subjects, user_grades = st.session_state['user_profile']
    intent, fields, institutions, counties, inst_types = detect_intent(
        message, all_institutions, field_to_clusters, programmes_df
    )

    if intent == 'recommend_programme':
        recommendations, feedback = recommend_programmes(
            subjects, user_grades, programmes_df,
            fields=fields, institutions=institutions,
            counties=counties, institution_types=inst_types
        ) 

        if recommendations:
            response = "✅ **Here are programmes you might qualify for:**\n\n"
            response += "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))
        else:
            response = "❌ **No qualifying programmes found** for your query.\n\n"
            if feedback:
                response += "**Analysis:**\n" + feedback + "\n\n"
            if fields:
                response += f"**Searched fields:** {', '.join(fields)}\n"
            if institutions:
                response += f"**Searched institutions:** {', '.join(institutions)}\n"
            response += "\n💡 **Tip:** Try searching for programmes in clusters where you have qualifying points."

    elif intent == 'institution_info':
        response = institution_info(institutions)

    else:
        response = ("💬 **I can help you with:**\n\n"
                   "- Programme recommendations by field (e.g., 'medicine', 'engineering', 'it')\n"
                   "- Institution information (e.g., 'jkuat', 'uon', 'Moi University')\n"
                   "- Combined queries (e.g., 'law at uon', 'jkuat it programmes')\n\n"
                   "**Supported abbreviations:** jkuat, uon, ku, moi, tuk, mmust, jooust, etc.")

    return response

if user_input:
    answer = respond_to_user(user_input)
    st.session_state['chat_history'].append(("You", user_input))
    st.session_state['chat_history'].append(("Bot", answer))

if st.session_state['chat_history']:
    st.subheader("💬 Conversation")
    for speaker, text in st.session_state['chat_history']:
        if speaker == "You":
            st.markdown(f"**🙋 {speaker}:** {text}")
        else:
            st.markdown(f"**🤖 {speaker}:**\n\n{text}")
            st.markdown("---")