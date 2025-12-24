import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

#  cd C:\Users\harpr\OneDrive\Desktop\ProjectDashboard
#  python -m streamlit run app.py

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Research Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR LARGER TABS ---
st.markdown("""
<style>
    /* Increase font size of tab labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    /* Optional: Add a little color accent to tabs */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: INDIAN CURRENCY FORMATTER ---
def format_inr(number):
    try:
        s, *d = str(int(number)).partition(".")
        r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        return "₹ " + "".join([r] + d)
    except:
        return f"₹ {number}"

# --- HELPER: CUSTOM KNN (FIND SIMILAR PROJECTS) ---
def get_similar_projects(target_project_name, df, top_n=3):
    target = df[df['Name of Project'] == target_project_name].iloc[0]
    others = df[df['Name of Project'] != target_project_name].copy()
    
    max_amt, min_amt = df['Amount'].max(), df['Amount'].min()
    max_dur, min_dur = df['Duration_Days'].max(), df['Duration_Days'].min()
    
    range_amt = max_amt - min_amt if max_amt != min_amt else 1
    range_dur = max_dur - min_dur if max_dur != min_dur else 1

    others['dist_amt'] = ((others['Amount'] - target['Amount']) / range_amt) ** 2
    
    others['dist_dur'] = np.where(
        others['Duration_Days'].notna() & pd.notna(target['Duration_Days']),
        ((others['Duration_Days'] - target['Duration_Days']) / range_dur) ** 2,
        1.0 
    )
    
    others['dist_dept'] = np.where(others['Department'] == target['Department'], 0, 0.5)
    others['similarity_score'] = np.sqrt(others['dist_amt'] + others['dist_dur'] + others['dist_dept'])
    
    return others.sort_values('similarity_score').head(top_n)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        try:
            df = pd.read_csv("data.csv", encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv("data.csv", encoding='ISO-8859-1')
        
        if 'Department' in df.columns:
            df['Department'] = df['Department'].astype(str).str.strip()
            dept_corrections = {
                'TT': 'Textile Technology', 'Textile': 'Textile Technology',
                'Physcis': 'Physics', 'Physics': 'Physics', 
                'BT': 'Bio Technology', 'CH': 'Chemical Engineering',
                'CY': 'Chemistry', 'CE': 'Civil Engineering',
                'CSE': 'Computer Science', 'ECE': 'Electronics & Comm.',
                'EE': 'Electrical Engineering', 'ICE': 'Instrumentation & Control',
                'IPE': 'Industrial & Production', 'ME': 'Mechanical Engineering',
                'IT': 'Information Technology', 'MC': 'Maths & Computing'
            }
            df['Department'] = df['Department'].replace(dept_corrections)

        if 'Amount, Rs.' in df.columns:
            df['Amount, Rs.'] = df['Amount, Rs.'].astype(str).str.replace(',', '', regex=False)
            df['Amount, Rs.'] = pd.to_numeric(df['Amount, Rs.'], errors='coerce').fillna(0)
            df.rename(columns={'Amount, Rs.': 'Amount'}, inplace=True)
        
        # --- NEW: Pre-calculate Formatted Amount for Tooltips ---
        df['Formatted_Amount'] = df['Amount'].apply(format_inr)

        def parse_custom_date(date_str):
            if not isinstance(date_str, str): return pd.NaT
            date_str = date_str.strip() 
            try:
                return pd.to_datetime(date_str, format='%y-%b')
            except:
                return pd.NaT

        if 'From' in df.columns:
            df['Start_Date'] = df['From'].apply(parse_custom_date)
        if 'To' in df.columns:
            df['End_Date'] = df['To'].apply(parse_custom_date)

        if 'Start_Date' in df.columns and 'End_Date' in df.columns:
            df['Duration_Days'] = (df['End_Date'] - df['Start_Date']).dt.days
            df['Start_Str'] = df['Start_Date'].dt.strftime('%d-%b-%Y')
            df['End_Str'] = df['End_Date'].dt.strftime('%d-%b-%Y')

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty: st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("Filters")
st.sidebar.markdown("Refine your analysis below.")

def create_sidebar_filter(label, col_name):
    if col_name in df.columns:
        options = sorted(df[col_name].dropna().unique().tolist())
        if st.sidebar.checkbox(f"Select All {label}", value=True, key=f"all_{col_name}"):
            return options
        else:
            return st.sidebar.multiselect(f"Select {label}", options, default=options)
    return []

selected_dept = create_sidebar_filter("Departments", "Department")
selected_status = create_sidebar_filter("Status", "Status")
selected_agency = create_sidebar_filter("Agencies", "Funding Agency")

filtered_df = df[
    (df['Department'].isin(selected_dept)) &
    (df['Status'].isin(selected_status)) &
    (df['Funding Agency'].isin(selected_agency))
]

# --- DASHBOARD HEADER ---
st.title("Research Intelligence Dashboard")
st.markdown(f"Analyzed **{len(filtered_df)} projects** with a total value of **{format_inr(filtered_df['Amount'].sum())}**")
st.markdown("---")

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Raw Data", "Smart Conclusions", "KNN Analysis"])

# === TAB 1: VISUALIZATIONS ===
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    total_funding = filtered_df['Amount'].sum()
    avg_ticket = filtered_df['Amount'].mean() if not filtered_df.empty else 0
    top_agency = filtered_df['Funding Agency'].mode()[0] if not filtered_df.empty else "N/A"
    
    k1.metric("Total Funding", format_inr(total_funding))
    k2.metric("Projects Count", len(filtered_df))
    k3.metric("Avg. Project Value", format_inr(avg_ticket))
    k4.metric("Top Funding Agency", str(top_agency)[:15] + ".." if len(str(top_agency))>15 else str(top_agency))

    st.markdown("###") 

    st.subheader("Funding by Department")
    if not filtered_df.empty:
        dept_data = filtered_df.groupby('Department')['Amount'].sum().reset_index().sort_values('Amount', ascending=True)
        # Add formatted text for tooltip
        dept_data['Formatted_Amount'] = dept_data['Amount'].apply(format_inr)
        
        fig_bar = px.bar(
            dept_data, 
            y='Department', 
            x='Amount', 
            orientation='h', 
            text_auto='.2s', 
            color='Amount', 
            color_continuous_scale='Blues',
            # Use Formatted Amount in Hover
            hover_data={'Amount': False, 'Formatted_Amount': True, 'Department': True}
        )
        fig_bar.update_layout(xaxis_title="Amount (INR)", yaxis_title=None, height=600, transition_duration=500)
        st.plotly_chart(fig_bar, use_container_width=True)
            
    st.markdown("---")
    st.subheader("Project Status Distribution")
    if not filtered_df.empty:
        status_counts = filtered_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(status_counts, values='Count', names='Status', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(height=500, transition_duration=500)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Duration vs. Funding Size")
    valid_scatter_data = filtered_df.dropna(subset=['Duration_Days', 'Amount'])
    if not valid_scatter_data.empty:
        fig_scat = px.scatter(
            valid_scatter_data, 
            x='Duration_Days', 
            y='Amount', 
            size='Amount', 
            color='Department',
            # Custom Hover Data with Formatted Money
            hover_data={
                'Name of Project': True, 
                'Start_Str': True, 
                'End_Str': True, 
                'Duration_Days': True, 
                'Amount': False,           # Hide raw number
                'Formatted_Amount': True   # Show readable number
            },
            labels={'Duration_Days': 'Days', 'Amount': 'Funding (INR)', 'Start_Str': 'Start', 'End_Str': 'End', 'Formatted_Amount': 'Value'},
            title="Correlation: Project Duration vs Funding"
        )
        fig_scat.update_layout(height=600, transition_duration=500)
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info("Insufficient date data.")

    st.markdown("---")
    st.subheader("Top Agencies Overview")
    if not filtered_df.empty:
        agency_data = filtered_df['Funding Agency'].value_counts().reset_index()
        agency_data.columns = ['Funding Agency', 'Number of Projects']
        st.dataframe(agency_data, use_container_width=True, height=400)

# === TAB 2: RAW DATA ===
with tab2:
    st.subheader("Project Database")
    st.dataframe(filtered_df, height=600)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "research_data.csv", "text/csv")

# === TAB 3: SMART CONCLUSIONS ===
with tab3:
    st.header("Executive Summary & Strategic Insights")
    if filtered_df.empty:
        st.warning("Please select at least one department/status.")
    else:
        dept_sums = filtered_df.groupby('Department')['Amount'].sum()
        top_dept_name = dept_sums.idxmax()
        top_dept_val = dept_sums.max()
        
        agency_counts = filtered_df['Funding Agency'].value_counts()
        top_agency_name = agency_counts.idxmax()
        top_agency_count = agency_counts.max()
        agency_share_pct = (top_agency_count / len(filtered_df)) * 100
        
        avg_amt = filtered_df['Amount'].mean()
        high_val_count = len(filtered_df[filtered_df['Amount'] > avg_amt])
        
        has_dates = 'Duration_Days' in filtered_df.columns and filtered_df['Duration_Days'].notna().any()
        corr_text = "Insufficient date data."
        if has_dates:
            long_term = filtered_df[filtered_df['Duration_Days'] > 730]
            short_term = filtered_df[filtered_df['Duration_Days'] <= 730]
            avg_long = long_term['Amount'].mean() if not long_term.empty else 0
            avg_short = short_term['Amount'].mean() if not short_term.empty else 0
            if avg_long > avg_short:
                corr_text = f"**Positive Correlation:** Long-term projects (>2 years) receive significantly higher funding (Avg: {format_inr(avg_long)}) compared to short-term ones (Avg: {format_inr(avg_short)})."
            else:
                corr_text = f"**Weak Correlation:** Short-term projects are actually securing higher or equal average funding ({format_inr(avg_short)}) compared to long-term ones ({format_inr(avg_long)})."

        st.subheader("✅ Key Strengths")
        st.markdown(f"""
        * **Dominant Research Area:** **{top_dept_name}** leads with **{format_inr(top_dept_val)}** funding.
        * **High-Value Capabilities:** Secured **{high_val_count} high-value projects** exceeding average ticket size.
        * **Strong Agency Relations:** Reliable pipeline with **{top_agency_name}** ({top_agency_count} projects).
        """)
        
        st.markdown("---")
        st.subheader("⚠️ Strategic Risks")
        risk_bullets = []
        if agency_share_pct > 30:
            risk_bullets.append(f"**High Agency Dependency:** **{agency_share_pct:.1f}%** of projects rely on {top_agency_name}.")
        else:
            risk_bullets.append(f"**Moderate Agency Reliance:** Top agency holds {agency_share_pct:.1f}% share.")
        min_dept_val = dept_sums.min()
        min_dept_name = dept_sums.idxmin()
        risk_bullets.append(f"**Departmental Disparity:** **{min_dept_name}** has the lowest funding ({format_inr(min_dept_val)}).")
        
        for bullet in risk_bullets: st.markdown(f"* {bullet}")

        st.markdown("---")
        st.subheader("⏳ Time vs. Money")
        st.markdown(f"* {corr_text}")
        if has_dates: st.markdown(f"* **Average Duration:** {filtered_df['Duration_Days'].mean():.0f} days.")

# === TAB 4: KNN ANALYSIS ===
with tab4:
    st.header("KNN Analysis")
    st.markdown("Select a project to identify the most similar research initiatives based on **Budget, Duration, and Department parameters**.")
    
    project_list = sorted(filtered_df['Name of Project'].dropna().unique().tolist())
    if project_list:
        selected_project = st.selectbox("Search or Select a Project:", project_list)
        
        if selected_project:
            target_row = filtered_df[filtered_df['Name of Project'] == selected_project].iloc[0]
            
            st.markdown("### 📌 Selected Project Profile")
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Dept:** {target_row['Department']}")
            c2.info(f"**Amount:** {format_inr(target_row['Amount'])}")
            dur_disp = f"{target_row['Duration_Days']:.0f} days" if pd.notna(target_row['Duration_Days']) else "N/A"
            c3.info(f"**Duration:** {dur_disp}")
            
            st.markdown("---")
            st.subheader("🔗 Top 3 Similar Projects")
            
            similar_projects = get_similar_projects(selected_project, df)
            
            for _, row in similar_projects.iterrows():
                with st.container():
                    st.markdown(f"#### 🔹 {row['Name of Project']}")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.caption(f"Dept: {row['Department']}")
                    sc2.caption(f"Amount: {format_inr(row['Amount'])}")
                    dur_sim = f"{row['Duration_Days']:.0f} days" if pd.notna(row['Duration_Days']) else "N/A"
                    sc3.caption(f"Duration: {dur_sim}")
                    st.markdown("---")
    else:
        st.error("No projects available to analyze.")

st.caption("Generated by Research Analytics Dashboard v2.9")