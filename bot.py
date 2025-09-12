import streamlit as st
import random
import re
import pandas as pd

# Load the programmes data
@st.cache(allow_output_mutation=True)
def load_programmes():
    # Load from CSV file
    df = pd.read_csv('programmes.csv')
    df=df.sort_values(by="2021", ascending=False)
    
    # Clean up column names (remove any leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    return df

programmes_df = load_programmes()

# KUCCPS subjects
compulsory_subjects = ['ENG', 'KISW', 'CHEM','BIO']
math_variants = ['MAT A', 'MAT B']
other_subjects = [
     'PHY', 'AGR', 'HSC', 'HIST', 'GEO', 'CRE', 'ART',
    'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HAG'
]

all_subjects = compulsory_subjects + math_variants + other_subjects

grades = grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E']


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
group_definitions = { 'ENG/KIS': ['ENG', 'KISW'], 'MAT A/B': ['MAT A', 'MAT B'], 'MAT A/B or GRP II': ['MAT A', 'MAT B', 'BIO', 'PHY', 'CHE', 'AGR'], 'MAT A or PHY': ['MAT A', 'PHY'], 'GRP II': ['BIO', 'PHY', 'CHE', 'AGR'], 'GRP III': ['HIST', 'GEO', 'CRE', 'IRE', 'HRE'], 'GRP IV': ['BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD'], 'GRP V': ['HSC', 'ART', 'HAG', 'MUS'], 'BIO/PHY/CHE': ['BIO', 'PHY', 'CHE'], 'GRP II/III/IV/V': ['BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'], 'ENG/KIS or GRP III/IV/V': ['ENG', 'KISW', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'], 'ENG/KIS or GRP II/III/IV/V': ['ENG', 'KISW', 'BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'], 'KIS or GRP II/III/IV/V': ['KISW', 'BIO', 'PHY', 'CHE', 'AGR', 'HIST', 'GEO', 'CRE', 'IRE', 'HRE', 'BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HSC', 'ART', 'HAG', 'MUS'], 'BIO/HSC': ['BIO', 'HSC'], '2nd GRP II': ['BIO', 'PHY', 'CHE', 'AGR'], '2nd GRP III': ['HIST', 'GEO', 'CRE', 'IRE', 'HRE'], 'PHY/CHE': ['PHY', 'CHE'], 'CHE/BIO/GEO': ['CHE', 'BIO', 'GEO'], 'BIO/CHE/GEO': ['BIO', 'CHE', 'GEO'], 'FRE/GER': ['FRE', 'GER'], 'MUS': ['MUS'], 'GEO': ['GEO'] }

def calculate_cluster_points_per_group(subjects, grades):
    cluster_points = {}

    # Loop through each cluster group (1 to 20)
    for cluster_id, cluster_reqs in cluster_subjects.items():
        selected_subjects = []
        
        for req in cluster_reqs:
            matched = []
            
            # Handle 'OR' and '/' cases
            if ' or ' in req:
                options = req.split(' or ')
            elif '/' in req:
                options = req.split('/')
            else:
                options = [req]

            # Flatten compound group labels (e.g. GRP II/III/IV/V)
            expanded_options = []
            for opt in options:
                opt = opt.strip()
                expanded_options += group_definitions.get(opt, [opt])

            # Find highest scoring subject in the matched group
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

        # Take sum of best 4 subjects
        if len(selected_subjects) == 4:
            total = sum(selected_subjects)
            cluster_points[cluster_id] = round(total, 2)
        else:
            cluster_points[cluster_id] = None  # Not enough subjects matched

    return cluster_points

# Extract unique institutions from the dataframe
all_institutions = programmes_df['Institution'].unique().tolist()
 
st.title("🎲 KUCCPS Degree & Institution Recommender Bot")

def calculate_cluster_points(subjects, grades, cluster_subs):
    points = 0
    math_point = 0
    if 'MAT A' in subjects:
        math_point = grade_points[grades['MAT A']]
    elif 'MAT B' in subjects:
        math_point = grade_points[grades['MAT B']]
    for subj in cluster_subs:
        if subj in ['MAT A', 'MAT B']:
            points += math_point
        elif subj in subjects:
            points += grade_points[grades[subj]]
    return points

def parse_grade_requirement(grade_str):
    """Parse grade requirement strings like 'MAT A: C+'"""
    if pd.isna(grade_str):
        return None, None
    
    if ':' in str(grade_str):
        subject, grade = str(grade_str).split(':')
    
        return subject.strip(), grade.strip()
        
    return None, None

def get_recommended_programmes(subjects, user_grades):
    recommended = []
    for _, programme in programmes_df.iterrows():
        # Check if user has the required subjects 
        required_subjects = [
            programme['Cluster Subject 1'],
            programme['Cluster Subject 2'],
            programme['Cluster Subject 3'],
            programme['Cluster Subject 4']
        ]
        
        # Check if user meets subject requirements
        has_subjects = all(subj in subjects for subj in required_subjects)
        
        if has_subjects:
            # Check if user meets grade requirements for key subjects
            meets_grade_requirements = True
            
            # Parse subject 1 requirement
            subj1_req = programme['Subject 1 Requirement']
            if pd.notna(subj1_req):
                req_subject, req_grade = parse_grade_requirement(subj1_req)
                if req_subject and req_grade:
                    user_grade = user_grades.get(req_subject, 'E')
                     
                    # Compare grades (lower index in grades list is better)
                    if grades.index(user_grade) > grades.index(req_grade):
                        meets_grade_requirements = False
            
            # Parse subject 2 requirement
            subj2_req = programme['Subject 2 Requirement']
            if pd.notna(subj2_req) and meets_grade_requirements:
                req_subject, req_grade = parse_grade_requirement(subj2_req)
                if req_subject and req_grade:
                    user_grade = user_grades.get(req_subject, 'E')
                    if grades.index(user_grade) > grades.index(req_grade):
                        meets_grade_requirements = False
            
            if meets_grade_requirements:
                recommended.append(programme)
    
    return recommended

science_subjects = ['BIO','PHY']  # Include CHEM only if not compulsory (else ignore)
humanities_subjects = ['HSC', 'HIST', 'GEO', 'CRE', 'ART']
technical_subjects = ['AGR','BS', 'COMP', 'AVT', 'ELEC', 'MET', 'WOOD', 'HAG']

if st.button("Generate Random KUCCPS Profile"):
    chosen_subjects = compulsory_subjects.copy()  # ENG, KISW, CHEM
    
    # Choose one math variant
    chosen_math = random.choice(math_variants)
    chosen_subjects.append(chosen_math)
    
    # Now we must select remaining subjects to total 8 subjects
    remaining = 8 - len(chosen_subjects)  # should be 8 - 4 = 4 subjects left
    
    # Pick at least:
    # - 2 science (excluding CHEM because compulsory)
    # - 1 humanities
    # - 1 technical
    
    # Select 2 science subjects (exclude CHEM which is compulsory)
    # Pick 0 or 1 additional science subject (excluding CHEM which is already included)
    selected_sciences = random.sample([s for s in science_subjects if s != 'BIO'], k=random.randint(0, 1))

    
    # Select 1 humanities subject
    selected_humanity = random.sample(humanities_subjects, 1)
    
    # Select 1 technical subject
    selected_technical = random.sample(technical_subjects, 1)
    
    # Combine all selected subjects
    selected_subjects = selected_sciences + selected_humanity + selected_technical
    
    # Add these to chosen_subjects
    chosen_subjects += selected_subjects
    
    # Safety check - if we have less than 8 (should not happen here), fill with random from other subjects excluding already picked
    if len(chosen_subjects) < 8:
        already_chosen = set(chosen_subjects)
        remaining_subjects = [s for s in other_subjects if s not in already_chosen]
        chosen_subjects += random.sample(remaining_subjects, 8 - len(chosen_subjects))
    
    # Generate grades - assuming you want highest grade 'A' for all randomly generated subjects (or random grades from full grades list)
    user_grades = {subj: random.choice(grades) for subj in chosen_subjects}
    
    st.session_state['user_profile'] = (chosen_subjects, user_grades)
    st.session_state['chat_history'] = []
    st.success("Random KUCCPS profile generated! You can now ask the bot questions.")

if 'user_profile' not in st.session_state:
    st.info("Please generate a random KUCCPS profile first.")
    st.stop()

# Show generated profile horizontally
subjects, user_grades = st.session_state['user_profile']
st.subheader("🎓 Current KUCCPS Profile")
cols = st.columns(8)
for idx, subj in enumerate(subjects):
    with cols[idx]:
        st.markdown(f"**{subj}**")
        st.markdown(f"*Grade: {user_grades[subj]}*")

def calculate_weighted_cluster_points(cluster_grades, grade_points, mean_grade):
    """
    Calculate Weighted Cluster Points (WCP) based on cluster grades.

    Parameters:
    - cluster_grades: list of grades for the cluster (e.g. ['A', 'B', 'A'])
    - grade_points: dict mapping grades to points (e.g. {'A': 12, 'B': 10, ...})
    - mean_grade: the mean grade (string) across the cluster subjects

    Returns:
    - Weighted Cluster Points (float)
    """
    raw_cluster_points = sum(grade_points[g] for g in cluster_grades if g in grade_points)

    performance_index = grade_points.get(mean_grade, 0) / 12  # safeguard with get()

    wcp = raw_cluster_points * performance_index
    return wcp

if st.checkbox("Show Cluster Points for All Groups"):
    subjects, user_grades = st.session_state['user_profile']
    cluster_pts = calculate_cluster_points_per_group(subjects, user_grades)

    st.subheader("📊 Your Cluster Points (out of 48)")
    
    # Arrange 5 columns per row for a clean horizontal layout
    num_clusters = len(cluster_pts)
    num_cols = 5
    rows = (num_clusters + num_cols - 1) // num_cols  # total rows needed

    cluster_items = list(cluster_pts.items())
    idx = 0

    for _ in range(rows):
        cols = st.columns(num_cols)
        for col in cols:
            if idx < len(cluster_items):
                cluster_id, pts = cluster_items[idx]
                if pts is not None:
                    col.markdown(f"**Cluster {cluster_id}:** {pts}")
                else:
                    col.markdown(f"**Cluster {cluster_id}:** _N/A_")
                idx += 1

# Chat bot interface
st.markdown("---")
st.subheader("🤖 Chat with the Recommender Bot")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Ask your question here (e.g., 'Which programmes am I recommended?', 'What programmes am I qualified for at X university?')")

# Synonym sets
programme_synonyms = ['programme', 'program', 'course', 'degree', 'study', 'major', 'field']
institution_synonyms = ['institution', 'college', 'university', 'uni', 'school', 'campus']
qualification_synonyms = ['qualify', 'eligible', 'admit', 'accept', 'admission']
recommend_synonyms = ['recommend', 'suggest', 'advise', 'best', 'fit']

def extract_institution(text, institution_list):
    text_lower = text.lower()
    for inst in institution_list:
        if str(inst).lower() in text_lower:
            return inst
    return None

def detect_intent(text):
    text_lower = text.lower()
    # Check for recommend intent
    if any(word in text_lower for word in recommend_synonyms) and any(word in text_lower for word in programme_synonyms):
        return 'recommend_programme'
    if any(word in text_lower for word in institution_synonyms) and any(word in text_lower for word in qualification_synonyms):
        return 'list_institutions'
    if any(word in text_lower for word in programme_synonyms) and any(word in text_lower for word in institution_synonyms):
        return 'institution_programmes'
    # fallback generic queries
    if 'programme' in text_lower or 'course' in text_lower or 'degree' in text_lower:
        return 'recommend_programme'
    if 'institution' in text_lower or 'university' in text_lower:
        return 'list_institutions'
    return 'unknown'

def bot_response_advanced(user_input):
    subjects, grades = st.session_state['user_profile']
    recommended_programmes = get_recommended_programmes(subjects, grades)
     
    intent = detect_intent(user_input)
    institution_in_question = extract_institution(user_input, all_institutions)

    if intent == 'recommend_programme':
        if not recommended_programmes:
            return "Based on your profile, I couldn't find any programmes that match your qualifications."
        
        response = "Based on your profile, I recommend these degree programmes:\n"
        i=0
        for prog in recommended_programmes:
            
            response += f"- {prog['Programme']} at {prog['Institution']} (Cut off 2021: {prog['2021']})\n"
            i+=1
            if i==5:break
        return response
    
    elif intent == 'list_institutions':
        if institution_in_question:
            # Filter programmes by institution
            inst_programmes = [p for p in recommended_programmes if p['Institution'] == institution_in_question]
            if not inst_programmes:
                return f"You don't qualify for any programmes at {institution_in_question} based on your current profile."
            i=0
            response = f"You qualify for these programmes at {institution_in_question}:\n"
            for prog in inst_programmes:
                response += f"- {prog['Programme']} (Cut off 2015: {prog['2015']})\n"
                i+=1
                if i==5:break
            return response
        else:
            # List all institutions the user qualifies for
            qualified_institutions = list(set(p['Institution'] for p in recommended_programmes))
            if not qualified_institutions:
                return "Based on your profile, you don't qualify for any programmes at the institutions in our database."
            
            return ("Based on your profile, you qualify for admission at these institutions:\n- " +
                    "\n- ".join(qualified_institutions))
    
    elif intent == 'institution_programmes':
        if institution_in_question:
            # Filter programmes by institution
            inst_programmes = [p for p in recommended_programmes if p['Institution'] == institution_in_question]
            if not inst_programmes:
                return f"{institution_in_question} doesn't have any programmes that match your qualifications."
            
            response = f"At {institution_in_question}, you can study these programmes:\n"
            for prog in inst_programmes:
                print(type(prog))
                response += f"- {prog['Programme']} (Code: {prog['program_code']})\n"
            return response
        else:
            # No institution mentioned, fallback to programmes
            if not recommended_programmes:
                return "Based on your profile, I couldn't find any programmes that match your qualifications."
            
            response = "I recommend these degree programmes based on your profile:\n"
            for prog in recommended_programmes:
                print(type(prog))
                response += f"- {prog['Programme']} at {prog['Institution']} (Code: {prog['program_code']})\n"
            return response
    else:
        return ("I'm sorry, I didn't quite understand your question. "
                "You can ask things like:\n- 'Which programmes am I recommended?'\n"
                "- 'What institutions am I qualified for?'\n"
                "- 'What programmes can I study at X university?'")

if user_input:
    st.session_state['chat_history'].append(("User", user_input))
    reply = bot_response_advanced(user_input)
    st.session_state['chat_history'].append(("Bot", reply)) 

# Display chat history
for sender, msg in st.session_state['chat_history']:
    if sender == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")