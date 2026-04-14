

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(layout="wide", page_title="🏏 Cricket Match Predictor & Analytics")

# Custom CSS for premium feel
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 30px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .winner-text {
        color: #ff4b4b;
        font-size: 3.5rem;
        font-weight: 800;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    df = pd.read_csv("ODI_Match_info.csv")
    df = df.dropna(subset=['winner'])
    return df

df = load_data()
model = load_model()

# Header
st.title("🏏 Enhanced ODI Match Insights & Prediction")
st.markdown("---")



# --- SECTION 1: PREDICTOR ---
st.header("🎯 Match Winner Predictor")

if model is None:
    st.warning("⚠️ Model (model.pkl) not found. Please run the training script (Cricket_M_A.py) first.")
else:
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Select Teams
            teams = sorted(df['team1'].unique())
            t1 = st.selectbox("Select Team 1", teams)
            t2 = st.selectbox("Select Team 2", [t for t in teams if t != t1])
            
            # 2. Select City FIRST
            cities = sorted(df['city'].dropna().unique())
            city = st.selectbox("Select City", cities)
            
        with col2:
            # 3. DYNAMIC VENUE FILTERING
            # Filter the dataframe to show only venues present in the selected city
            filtered_venues = sorted(df[df['city'] == city]['venue'].unique())
            venue = st.selectbox("Match Venue", filtered_venues)
            
            toss_winner = st.selectbox("Toss Winner", [t1, t2])
            toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

    if st.button("Analyze & Predict Winner"):
        # Engineering the same features as training
        input_df = pd.DataFrame([{
            'team1': t1,
            'team2': t2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'city': city,
            'team1_toss_win': 1 if toss_winner == t1 else 0
        }])
        
        try:
            prediction = model.predict(input_df)[0]
            
            # Display Prediction
            st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: #31333F;">🏆 Predicted Winner</h2>
                    <div class="winner-text">{prediction}</div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")


# --- SECTION 2: ANALYTICS ---
st.header("📊 Deep Match Analytics")

tab1, tab2, tab3 = st.tabs(["Team Performance", "Venue Analysis", "Toss Impact"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Wins Distribution by Team")
        team_wins = df['winner'].value_counts().head(10).reset_index()
        team_wins.columns = ['Team', 'Wins']
        fig1 = px.bar(team_wins, x='Wins', y='Team', orientation='h', 
                      color='Wins', color_continuous_scale='Reds',
                      template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_b:
        st.subheader("Win Strategy Preference")
        # Visualizing match types
        win_type = pd.DataFrame({
            'Strategy': ['Defending (Wins by Runs)', 'Chasing (Wins by Wickets)'],
            'Count': [(df['win_by_runs'] > 0).sum(), (df['win_by_wickets'] > 0).sum()]
        })
        fig2 = px.pie(win_type, values='Count', names='Strategy', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Top 10 High-Frequency Venues")
        venue_counts = df['venue'].value_counts().head(10).reset_index()
        venue_counts.columns = ['Venue', 'Matches']
        fig3 = px.funnel(venue_counts, y='Venue', x='Matches', color='Matches')
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_d:
        st.subheader("City-wise Match Distribution")
        city_counts = df['city'].value_counts().head(10)
        fig4 = go.Figure(data=[go.Scatter(
            x=city_counts.index, y=city_counts.values,
            mode='markers',
            marker=dict(size=city_counts.values*0.5, color=city_counts.values, colorscale='Viridis', showscale=True)
        )])
        fig4.update_layout(xaxis_title="City", yaxis_title="Number of Matches", template='plotly_white')
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Toss Decisions Trend")
        toss_dec = df['toss_decision'].value_counts().reset_index()
        fig5 = px.pie(toss_dec, values='count', names='toss_decision', hole=0.5,
                      color_discrete_map={'bat':'#FFA15A', 'field':'#636EFA'})
        st.plotly_chart(fig5, use_container_width=True)
        
    with col_f:
        st.subheader("Correlation: Toss Win vs Match Win")
        df['Toss Success'] = (df['toss_winner'] == df['winner']).map({True: 'Win Match', False: 'Lose Match'})
        toss_impact = df['Toss Success'].value_counts().reset_index()
        fig6 = px.bar(toss_impact, x='Toss Success', y='count', color='Toss Success',
                      color_discrete_map={'Win Match': '#2ecc71', 'Lose Match': '#e74c3c'})
        st.plotly_chart(fig6, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ for Cricket Fans</div>", unsafe_allow_html=True)



