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
    "sport": [12],
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

try:
    programmes_df = load_programmes()
except FileNotFoundError:
    st.error("CSV file 'degree_req_cutoff.csv' not found. Please place it in the app folder.")
    st.stop()

# Initialize optional columns if not present
if 'county' not in programmes_df.columns:
    programmes_df['county'] = ''
if 'institution_type' not in programmes_df.columns:
    programmes_df['institution_type'] = ''

# Build key-to-institution mapping
if 'key' in programmes_df.columns and 'institution' in programmes_df.columns:
    key_to_institution = dict(
        zip(
            programmes_df['key'].astype(str).str.lower().str.strip(),
            programmes_df['institution'].astype(str).str.strip()
        )
    )
else:
    key_to_institution = {}

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
    """Match institution from query text using df['key'] and partial name matching."""
    matched = set()
    
    # Check using df['key'] with word boundaries
    for key, inst in key_to_institution.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            matched.add(inst)
    
    # Fallback: partial name matching on full institution name
    if not matched:
        for inst in all_institutions:
            inst_normalized = normalize_institution_name(inst)
            if inst_normalized:
                # Check if a significant part of institution name is in query
                inst_words = inst_normalized.split()
                for word in inst_words:
                    if len(word) > 3 and word in text_lower:
                        matched.add(inst)
                        break
    
    return list(matched)

def detect_intent(text, institution_list, field_to_clusters, df):
    """Detect user intent and extract relevant filters from query."""
    text_lower = text.lower().strip()
    matched_institutions = match_institution_from_query(text_lower)
    matched_fields = []
    
    # Detect fields - check longest matches first to avoid partial matches
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

def recommend_programmes(subjects, grades, programmes_df, margin=2, top_n=15,
                        allowed_clusters=None, fields=None, institutions=None,
                        counties=None, institution_types=None):
    """
    Recommend programmes based on user profile and filters.
    All filters work independently and can be combined.
    """
    cluster_pts = calculate_cluster_points_per_group(subjects, grades)
    recommendations = []
    seen_programmes = set()
    
    filtered_df = programmes_df.copy()
    applied_filters = []
    
    # --- FILTER 1: Institution filtering ---
    if institutions:
        institution_mask = pd.Series([False] * len(filtered_df))
        
        for inst in institutions:
            inst_normalized = normalize_institution_name(inst)
            
            # Try matching via key first (for abbreviations like 'jkuat', 'uon')
            key_matches = filtered_df[
                filtered_df['key'].astype(str).str.lower().str.contains(
                    rf'\b{re.escape(inst_normalized)}\b', na=False, regex=True
                )
            ]
            
            if not key_matches.empty:
                # Get the actual institution names from key matches
                matched_inst_names = key_matches['institution'].str.strip().unique()
                for matched_name in matched_inst_names:
                    mask = filtered_df['institution'].str.lower().str.strip() == normalize_institution_name(matched_name)
                    institution_mask = institution_mask | mask
            else:
                # Fallback: partial name matching
                mask = filtered_df['institution'].str.lower().str.contains(
                    re.escape(inst_normalized), na=False, regex=True
                )
                institution_mask = institution_mask | mask
        
        filtered_df = filtered_df[institution_mask]
        applied_filters.append(f"Institution: {', '.join(institutions)}")
    
    # --- FILTER 2: County filtering ---
    if counties:
        county_mask = pd.Series([False] * len(filtered_df))
        for county in counties:
            mask = filtered_df['county'].str.lower().str.strip() == county.lower()
            county_mask = county_mask | mask
        
        filtered_df = filtered_df[county_mask]
        applied_filters.append(f"County: {', '.join(counties)}")
    
    # --- FILTER 3: Institution type filtering ---
    if institution_types:
        type_mask = pd.Series([False] * len(filtered_df))
        for inst_type in institution_types:
            mask = filtered_df['institution_type'].str.lower().str.contains(
                inst_type.lower(), na=False, regex=False
            )
            type_mask = type_mask | mask
        
        filtered_df = filtered_df[type_mask]
        applied_filters.append(f"Type: {', '.join(institution_types)}")
    
    # --- FILTER 4: Determine target clusters from fields ---
    target_clusters = []
    if fields:
        for field in fields:
            field_clusters = field_to_clusters.get(field.lower(), [])
            target_clusters.extend(field_clusters)
        target_clusters = list(set(target_clusters))
        applied_filters.append(f"Field: {', '.join(fields)}")
    
    if allowed_clusters:
        if target_clusters:
            target_clusters = list(set(target_clusters) & set(allowed_clusters))
        else:
            target_clusters = allowed_clusters
    
    # --- Generate recommendations ---
    for _, row in filtered_df.iterrows():
        cluster_id = row['cluster']
        
        # Skip if cluster not in target clusters (only if target clusters specified)
        if target_clusters and cluster_id not in target_clusters:
            continue
        
        cutoff = row['pred_cutoff_2025']
        user_points = cluster_pts.get(cluster_id)
        
        if user_points is None or pd.isna(cutoff):
            continue
        
        # Check if user qualifies (with margin)
        if cutoff <= user_points + margin:
            # Create unique identifier to avoid duplicates
            prog_key = (row['programme_name'], row['institution'], cluster_id, cutoff)
            if prog_key in seen_programmes:
                continue
            seen_programmes.add(prog_key)
            
            # Calculate match quality
            points_above_cutoff = user_points - cutoff
            priority_score = (points_above_cutoff * 100) - (cluster_id * 0.1)
            
            rec_text = (
                f"{row['programme_name']} at {row['institution']} "
                f"(Cluster {cluster_id}: {get_cluster_name(cluster_id)}, "
                f"Cutoff: {cutoff:.1f} pts, Your score: {user_points:.1f} pts)"
            )
            
            recommendations.append((priority_score, rec_text, cluster_id, cutoff))
    
    # Sort by priority score (higher is better)
    recommendations.sort(key=lambda x: x[0], reverse=True)
    
    # --- Build feedback if no recommendations found ---
    if not recommendations:
        feedback_lines = []
        
        if applied_filters:
            feedback_lines.append(f"**Applied filters:** {' | '.join(applied_filters)}")
        
        if target_clusters:
            feedback_lines.append("\n**Cluster Analysis:**")
            for cluster_id in sorted(target_clusters):
                pts = cluster_pts.get(cluster_id)
                if pts is None:
                    feedback_lines.append(
                        f"❌ {get_cluster_name(cluster_id)} (Cluster {cluster_id}): "
                        f"You don't meet the subject requirements"
                    )
                else:
                    # Find minimum cutoff in this cluster within filtered results
                    cluster_programmes = filtered_df[filtered_df['cluster'] == cluster_id]
                    if not cluster_programmes.empty:
                        min_cutoff = cluster_programmes['pred_cutoff_2025'].min()
                        feedback_lines.append(
                            f"⚠️ {get_cluster_name(cluster_id)} (Cluster {cluster_id}): "
                            f"Your score is {pts:.1f} pts, minimum cutoff is {min_cutoff:.1f} pts"
                        )
                    else:
                        feedback_lines.append(
                            f"ℹ️ {get_cluster_name(cluster_id)} (Cluster {cluster_id}): "
                            f"No programmes available after applying filters"
                        )
        
        return [], "\n".join(feedback_lines)
    
    return [rec[1] for rec in recommendations[:top_n]], None

def institution_info(institution_names):
    """Get information about specific institutions."""
    if not institution_names:
        return "Please specify a university or institution name."
    
    clean_df = programmes_df.dropna(subset=['institution'])
    results = []
    
    for inst in institution_names:
        inst_normalized = normalize_institution_name(inst)
        
        # Try exact match first
        exact_matches = clean_df[
            clean_df['institution'].str.lower().str.strip() == inst_normalized
        ]
        
        if not exact_matches.empty:
            matches = exact_matches
        else:
            # Fallback to partial match
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
st.title("🎓 KUCCPS Degree & Institution Recommender Bot")

with st.sidebar:
    st.header("Controls")
    
    allowed_grades_for_generation = st.multiselect(
        "Allowed grades for profile",
        ['A', 'A-', 'B+', 'B', 'B-'],
        default=['A', 'A-']
    )
    
    if st.button("🎲 Generate Random KUCCPS Profile"):
        chosen_subjects = ['ENG', 'KISW', 'MAT A']
        
        # Add 2 or 3 sciences (including CHEM)
        science_combo = random.sample(['BIO', 'CHEM', 'PHY'], k=random.choice([2, 3]))
        chosen_subjects += science_combo
        
        # Fill remaining slots
        remaining_slots = 8 - len(chosen_subjects)
        remaining_subject_pool = [s for s in other_subjects if s not in chosen_subjects]
        if remaining_slots > 0:
            chosen_subjects += random.sample(
                remaining_subject_pool,
                min(remaining_slots, len(remaining_subject_pool))
            )
        
        # Assign grades
        if not allowed_grades_for_generation:
            allowed_grades_for_generation = ['A', 'A-']
        
        user_grades = {
            subj: random.choice(allowed_grades_for_generation)
            for subj in chosen_subjects
        }
        
        st.session_state['user_profile'] = (chosen_subjects, user_grades)
        st.session_state['chat_history'] = []
        st.success("✅ Random KUCCPS profile generated!")

# Require profile
if 'user_profile' not in st.session_state:
    st.info("Please generate a KUCCPS profile from the sidebar to start.")
    st.stop()

subjects_list, user_grades = st.session_state['user_profile']

# Display profile
st.subheader("🎓 Current KUCCPS Profile")
cols_count = min(8, max(1, len(subjects_list)))
cols = st.columns(cols_count)
for idx, subj in enumerate(subjects_list):
    col = cols[idx % cols_count]
    with col:
        st.markdown(f"**{subj}**")
        st.markdown(f"*Grade: {user_grades.get(subj, 'N/A')}*")

# Optional cluster points display
if st.checkbox("📊 Show Cluster Points for All Groups"):
    cluster_pts = calculate_cluster_points_per_group(subjects_list, user_grades)
    st.subheader("📈 Your Cluster Points (sum of best 4)")
    
    cluster_items = sorted(cluster_pts.items(), key=lambda x: x[0])
    num_cols = 5
    rows = math.ceil(len(cluster_items) / num_cols)
    idx = 0
    for _ in range(rows):
        cols_row = st.columns(num_cols)
        for c in cols_row:
            if idx < len(cluster_items):
                cid, pts = cluster_items[idx]
                cluster_name = get_cluster_name(cid)
                if pts is not None:
                    c.markdown(f"**{cluster_name}** ({cid}): {pts} pts")
                else:
                    c.markdown(f"**{cluster_name}** ({cid}): N/A")
                idx += 1

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input(
    "Ask me (e.g., 'medicine', 'engineering at jkuat', 'law in nairobi', 'public universities it'): "
)

def respond_to_user(message):
    """Generate response to user query."""
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
            response = "❌ **No qualifying programmes found**\n\n"
            if feedback:
                response += feedback
        return response
    
    # Fallback for general queries
    elif intent == 'general':
        matched_institutions = match_institution_from_query(message.lower())
        if matched_institutions:
            return institution_info(matched_institutions)
        return "🤖 I didn't quite understand. Try asking for a field, institution, or county."
    
    return "🤖 Sorry, I couldn't process your request."

# Display chat interface
if user_input:
    response_text = respond_to_user(user_input)
    st.session_state['chat_history'].append(("You", user_input))
    st.session_state['chat_history'].append(("Bot", response_text))

# Show chat history
for sender, text in st.session_state['chat_history']:
    if sender == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
