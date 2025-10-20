import streamlit as st
import random
import pandas as pd

# Load the programmes data 
def load_programmes():
    df = pd.read_csv('degree_req_cutoff.csv')
    df = df.sort_values(by="pred_cutoff_2025", ascending=False)
    df['key'] = df['key'].astype(str).str.strip().str.lower() 
    df.columns = df.columns.str.strip()
    
    return df

programmes_df = load_programmes()

# KUCCPS subjects
compulsory_subjects = ['ENG', 'KISW', 'CHEM', 'BIO']
math_variants = ['MAT A', 'MAT B']
other_subjects = [
    'PHY', 'AGR', 'HSC', 'HIST', 'GEO', 'CRE', 'ART',
    'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HAG'
]

grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E']

grade_points = {
    'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8,
    'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
}

cluster_subjects = {
    1: ["ENG/KIS", "MAT A/B", "BIO/PHY/CHE", "GRP III/IV/V"],        # Law
    2: ["ENG/KIS", "ENG/KIS", "MAT A/B", "GRP II/III/IV/V"],         # Business
    3: ["ENG/KIS", "MAT A/B or GRP II", "GRP III", "GRP II/III/IV/V"], # Social Sciences
    4: ["MAT A", "PHY", "CHE/BIO/GEO", "GRP II/III/IV/V"],           # Geosciences
    5: ["MAT A", "PHY", "CHE/BIO/GEO", "GRP II/III/IV/V"],           # Engineering
    6: ["MAT A", "PHY", "GRP III", "GRP II/III/IV/V"],               # Architecture
    7: ["MAT A", "PHY", "BIO/CHE/GEO", "GRP II/III/IV/V"],           # Computing
    8: ["MAT A", "BIO", "PHY/CHE", "GRP II/III/IV/V"],               # Agribusiness
    9: ["MAT A", "GRP II", "2nd GRP II", "GRP II/III/IV/V"],         # General Science
    10: ["MAT A", "GRP II", "GRP III", "GRP II/III/IV/V"],           # Actuarial/Economics
    11: ["CHE", "MAT A/B or PHY", "BIO/HSC", "ENG/KIS or GRP III/IV/V"], # Interior Design
    12: ["BIO/HSC", "MAT A/B", "GRP II/III", "ENG/KIS or GRP II/III/IV/V"], # Sport Science
    13: ["BIO", "CHE", "MAT A or PHY", "ENG/KIS or GRP II/III/IV/V"],     # Medicine
    14: ["HIST/GEO", "ENG/KIS", "MAT A/B or GRP II", "GRP II/III/IV/V"],  # History
    15: ["BIO", "CHE", "MAT A/PHY/GEO", "ENG/KIS or GRP II/III/IV/V"],    # Agriculture
    16: ["GEO", "MAT A/B", "GRP II", "GRP II/III/IV/V"],              # Geography
    17: ["FRE/GER", "ENG/KIS", "MAT A/B or GRP II/III", "GRP II/III/IV/V"], # Languages
    18: ["MUS", "ENG/KIS", "MAT A/B or GRP II/III", "GRP II/III/IV/V"],    # Music
    19: ["ENG", "MAT A/B or GRP II", "2nd GRP II", "KIS or GRP II/III/IV/V"], # Education
    20: ["CRE/IRE/HRE", "ENG/KIS", "2nd GRP III", "GRP II/IV/V"],     # Religious Studies
}
field_to_clusters = {
    # Legal & Governance
    "law": [1],  # Cluster 1: Law

    # Business, Finance, Economics
    "business": [2],  # Cluster 2: Business & Commerce
    "commerce": [2],
    "accounting": [2],
    "finance": [2],
    "economics": [2, 10],  # Cluster 10 includes Actuarial Science & Economics

    # Social Sciences & Arts
    "social science": [3, 14, 20],  # Cluster 3: Social Sciences, 14: History, 20: Religion
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

    # Education & Teaching
    "education": [19],  # Cluster 19: Education
    "teaching": [19],

    # Science & Math
    "science": [9, 10],  # 9: General Sciences, 10: Actuarial/Math-heavy
    "mathematics": [10],
    "physics": [9],
    "chemistry": [9],
    "biology": [9],
    "statistics": [10],
    "actuarial": [10],

    # Engineering & Technology
    "engineering": [5],  # Cluster 5: Engineering
    "civil engineering": [5],
    "mechanical engineering": [5],
    "electrical engineering": [5],
    "mechatronics": [5],
    "technology": [5, 7],  # Engineering + Computing

    # ICT & Computing
    "it": [7],
    "ict": [7],
    "computer science": [7],
    "computing": [7],
    "information technology": [7],
    "software": [7],
    "data science": [7, 10],

    # Health & Medicine
    "health": [11, 12, 13],  # 13: Medicine, 11: Interior Design (partially), 12: Sport Science
    "medicine": [13],
    "nursing": [13],
    "clinical": [13],
    "medical": [13],
    "pharmacy": [13],
    "public health": [13],
    "nutrition": [13],
    "dietetics": [13],
    "sport science": [12],
    "occupational therapy": [13],
    "physiotherapy": [13],
    "biomedical": [9, 13],

    # Agriculture & Environment
    "agriculture": [8, 15],  # 8: Agribusiness, 15: Agriculture
    "agribusiness": [8],
    "agronomy": [15],
    "veterinary": [13],  # Often shares cluster with medicine
    "environmental science": [15],
    "soil science": [15],
    "forestry": [15],

    # Architecture & Built Environment
    "architecture": [6],  # Cluster 6: Architecture
    "landscape architecture": [6],
    "urban planning": [6],
    "quantity surveying": [6],
    "construction management": [6],

    # Geosciences & Earth Sciences
    "geoscience": [4],
    "geology": [4],
    "earth science": [4],
    "meteorology": [4],
    "environment": [4, 15],

    # Interior Design & Art
    "interior design": [11],
    "design": [11],
    "fine art": [18],
    "graphic design": [11, 18],
    "applied art": [18],
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
    'GRP II/III/IV/V': ['BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE',
                       'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'],
    'ENG/KIS or GRP III/IV/V': ['ENG', 'KISW', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE',
                                'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'],
    'ENG/KIS or GRP II/III/IV/V': ['ENG', 'KISW', 'BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE',
                                  'IRE', 'HRE', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'],
    'KIS or GRP II/III/IV/V': ['KISW', 'BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE',
                              'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'],
    'BIO/HSC': ['BIO', 'HSC'],
    '2nd GRP II': ['BIO', 'PHY', 'CHE', 'AGR'],
    '2nd GRP III': ['HIST', 'GEO', 'CRE', 'IRE', 'HRE'],
    'PHY/CHE': ['PHY', 'CHE'],
    'CHE/BIO/GEO': ['CHE', 'BIO', 'GEO'],
    'BIO/CHE/GEO': ['BIO', 'CHE', 'GEO'],
    'FRE/GER': ['FRE', 'GER'],
    'MUS': ['MUS'],
    'GEO': ['GEO'],
}

def calculate_cluster_points_per_group(subjects, grades):
    cluster_points = {}

    for cluster_id, cluster_reqs in cluster_subjects.items():
        selected_subjects = []

        for req in cluster_reqs:
            # Handle 'or' and '/' as options
            if ' or ' in req:
                options = req.split(' or ')
            elif '/' in req:
                options = req.split('/')
            else:
                options = [req]

            expanded_options = []
            for opt in options:
                opt = opt.strip()
                expanded_options += group_definitions.get(opt, [opt])

            # Find highest scoring subject in the group
            best_subject = None
            best_score = -1
            for subj in expanded_options:
                if subj in subjects and subj in grades:
                    score = grade_points.get(grades[subj], 0)
                    if score > best_score:
                        best_score = score
                        best_subject = subj

            if best_subject:
                selected_subjects.append(best_score)

        # Sum of best 4 subjects for cluster points
        if len(selected_subjects) == 4:
            cluster_points[cluster_id] = round(sum(selected_subjects), 2)
        else:
            cluster_points[cluster_id] = None  # Not enough matched subjects

    return cluster_points
import re



# Extract unique institutions
all_institutions = programmes_df['institution'].unique().tolist() 

st.title("🎲 KUCCPS Degree & Institution Recommender Bot")

science_subjects = ['BIO', 'PHY']  # Exclude CHEM because compulsory
humanities_subjects = ['HSC', 'HIST', 'GEO', 'CRE', 'ART']
technical_subjects = ['AGR', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HAG']
 

allowed_grades_for_generation = ['A', 'A-']

if st.button("Generate Random KUCCPS Profile"):
    # Always include ENG, KISW, MAT A
    chosen_subjects = ['ENG', 'KISW', 'MAT A']

    # Add 2 or all 3 of BIO, CHEM, PHY
    science_combo = random.sample(['BIO', 'CHEM', 'PHY'], k=random.choice([2, 3]))
    chosen_subjects += science_combo

    # Total so far
    remaining_slots = 8 - len(chosen_subjects)

    # Fill the remaining subjects from other groups
    remaining_subject_pool = [s for s in other_subjects if s not in chosen_subjects]
    chosen_subjects += random.sample(remaining_subject_pool, remaining_slots)

    # Generate grades (C to A only)
    user_grades = {subj: random.choice(allowed_grades_for_generation) for subj in chosen_subjects}

    # Store in session
    st.session_state['user_profile'] = (chosen_subjects, user_grades)
    st.session_state['chat_history'] = []
    st.success("Random KUCCPS profile generated! You can now ask the bot questions.")

if 'user_profile' not in st.session_state:
    st.info("Please generate a random KUCCPS profile first.")
    st.stop()

subjects, user_grades = st.session_state['user_profile']

st.subheader("🎓 Current KUCCPS Profile")
cols = st.columns(8)
for idx, subj in enumerate(subjects):
    with cols[idx]:
        st.markdown(f"**{subj}**")
        st.markdown(f"*Grade: {user_grades[subj]}*")

if st.checkbox("Show Cluster Points for All Groups"):
    cluster_pts = calculate_cluster_points_per_group(subjects, user_grades)
    st.subheader("📊 Your Cluster Points (out of 48)")

    num_clusters = len(cluster_pts)
    num_cols = 5
    rows = (num_clusters + num_cols - 1) // num_cols

    cluster_items = list(cluster_pts.items())
    idx = 0

    for _ in range(rows):
        cols = st.columns(num_cols)
        for col in cols:
            if idx < len(cluster_items):
                cluster_id, pts = cluster_items[idx]
                if pts is not None:
                    col.markdown(f"**Cluster {cluster_id}: {pts} pts**")
                else:
                    col.markdown(f"**Cluster {cluster_id}: N/A**")
                idx += 1

def detect_intent(text, institution_list, field_to_clusters, df):
    text_lower = text.lower().strip()
    matched_institutions = []
    matched_fields = []

    # 1️⃣ Check if text exactly matches any 'key' in df
    if text_lower in df['key'].values:
        matched_rows = df[df['key'] == text_lower]
        if not matched_rows.empty:
            matched_institutions.extend(matched_rows['institution'].unique())

    # 2️⃣ Check for all institution names contained in the text
    for name in institution_list:
        if isinstance(name, str) and name.lower() in text_lower:
            if name not in matched_institutions:
                matched_institutions.append(name)

    # 3️⃣ Detect all academic fields mentioned in text
    for field in field_to_clusters:
        if field.lower() in text_lower:
            if field not in matched_fields:
                matched_fields.append(field)

    # 4️⃣ Intent resolution with multiple matches:
    if matched_institutions and matched_fields:
        # Return intent to recommend with first field and all institutions
        return 'recommend_programme', matched_fields, matched_institutions
    elif matched_fields:
        return 'recommend_programme', matched_fields, None
    elif matched_institutions:
        return 'institution_info', None, matched_institutions
    else:
        return 'general', None, None


def recommend_programmes(subjects, grades, programmes_df, margin=2, top_n=10, allowed_clusters=None, fields=None):
    """
    subjects: dict or list of user's subjects
    grades: dict or list of user's grades
    programmes_df: DataFrame with programmes info
    margin: score margin for cutoff
    top_n: number of recommendations to return
    allowed_clusters: list of cluster ids to restrict search (optional)
    fields: list of fields (e.g. ['health', 'law']) to filter by clusters
    """
    cluster_pts = calculate_cluster_points_per_group(subjects, grades)
    recommendations = []

    # Combine allowed clusters from fields if fields is given
    if fields:
        combined_clusters = set()
        for field in fields:
            field_clusters = field_to_clusters.get(field.lower())
            if field_clusters:
                combined_clusters.update(field_clusters)
        allowed_clusters = combined_clusters if combined_clusters else allowed_clusters

    for _, row in programmes_df.iterrows():
        cluster_id = row['cluster']
        if allowed_clusters and cluster_id not in allowed_clusters:
            continue  # skip if cluster not allowed

        cutoff = row['pred_cutoff_2025']
        user_points = cluster_pts.get(cluster_id)

        if user_points is None:
            continue

        if cutoff <= user_points + margin:
            diff = abs(user_points - cutoff)
            recommendations.append((diff, f"{row['programme_name']} at {row['institution']} (Cutoff: {cutoff} pts)"))

    recommendations.sort(key=lambda x: x[0])
    return [rec[1] for rec in recommendations[:top_n]]


def respond_to_user(message):
    intent, fields, institutions = detect_intent(
        message, all_institutions, field_to_clusters, programmes_df
    )
    print(intent, fields, institutions)

    if intent == 'recommend_programme':
        # Use first field if multiple found, or None
        field = fields[0] if fields else None

        recommendations = recommend_programmes(subjects, user_grades, programmes_df, field=field)

        if recommendations:
            response = f"Here are some {field or ''} programmes you might consider based on your profile:\n\n"
            response += "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))
            if institutions:
                response += f"\n\nNote: You asked about {', '.join(institutions)}. These programmes may be available there or elsewhere."
        else:
            response = "Sorry, I couldn't find any suitable programmes matching your profile and interests."

    elif intent == 'institution_info':
        # For multiple institutions, you can list info for each
        if isinstance(institutions, list):
            response = ""
            for inst in institutions:
                response += institution_info(inst) + "\n\n"
            response = response.strip()
        else:
            response = institution_info(institutions)

    else:
        response = "Sorry, I can currently only recommend programmes or provide institution info."

    return response


def institution_info(institution_name):
    if not institution_name:
        return "Please specify a university or institution name."

    # Ensure 'institution' column has no NaNs
    clean_df = programmes_df.dropna(subset=['institution'])

    # Do a case-insensitive match
    matches = clean_df[clean_df['institution'].str.contains(institution_name, case=False, na=False)]

    if matches.empty:
        return f"Sorry, I couldn't find any programmes for **{institution_name.title()}**."
    else:
        programs_list = matches['programme_name'].dropna().unique()
        return f"📚 {institution_name.title()} offers the following programmes:\n\n- " + "\n- ".join(programs_list)
def resolve_institution_name_from_input(text, df):
    text_lower = text.lower().strip()

    # First: Check if input matches any key (i.e., acronym)
    matched_rows = df[df['key'].astype(str).str.lower() == text_lower]
    if not matched_rows.empty:
        institution_name = matched_rows.iloc[0]['institution']
        print(f"Matched key: {text} → {institution_name}")
        return institution_name

    # Second: Check if any full institution name is in the text
    unique_institutions = df['institution'].dropna().unique()
    for name in unique_institutions:
        if isinstance(name, str) and name.lower() in text_lower:
            print(f"Matched full name: {name}")
            return name

    print("No institution matched")
    return None





if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Ask me about KUCCPS degree & institution recommendations:")

if user_input:
    answer = respond_to_user(user_input)
    st.session_state['chat_history'].append(("You", user_input))
    st.session_state['chat_history'].append(("Bot", answer))

if st.session_state['chat_history']:
    for speaker, text in st.session_state['chat_history']:
        if speaker == "You":
            st.markdown(f"**{speaker}:** {text}")
        else:
            st.markdown(f"**{speaker}:** {text}")
