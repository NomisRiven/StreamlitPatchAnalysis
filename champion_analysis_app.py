import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re

# Page configuration
st.set_page_config(
    page_title="Champion Analysis - Patch Comparison",
    page_icon="⚔️",
    layout="wide"
)

# Title
st.title("⚔️ Champion Analysis - Patch Comparison")
st.markdown("---")

def detect_patch_columns(df):
    """
    Automatically detect patch columns in the dataframe.
    Looks for patterns like: winrate_X, pickrate_X, banrate_X, games_X
    Returns two patch identifiers (old and new)
    """
    # Find all unique suffixes after underscore
    all_cols = df.columns.tolist()
    
    # Extract suffixes from metric columns
    suffixes = set()
    for col in all_cols:
        if any(metric in col for metric in ['winrate_', 'pickrate_', 'banrate_', 'games_']):
            parts = col.split('_')
            if len(parts) >= 2:
                suffix = '_'.join(parts[1:])  # Everything after first underscore
                suffixes.add(suffix)
    
    suffixes = sorted(list(suffixes))
    
    if len(suffixes) >= 2:
        return suffixes[0], suffixes[1]  # Return old and new patch
    elif len(suffixes) == 1:
        return suffixes[0], suffixes[0]
    else:
        return None, None

# Sidebar for upload and filters
with st.sidebar:
    st.header("📁 Data Loading")
    uploaded_file = st.file_uploader("Load CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: {len(df)} rows")
        
        # Extract date and time from filename or fetch_timestamp column
        filename = uploaded_file.name
        import re
        from datetime import datetime
        
        file_datetime = None
        
        # Method 1: Try to extract from fetch_timestamp column
        if 'fetch_timestamp' in df.columns:
            # Get the first timestamp (assuming all are the same or very close)
            timestamp_str = df['fetch_timestamp'].iloc[0]
            try:
                # Try to parse various timestamp formats
                # Examples: "2026-01-11 12:04:15", "2026-01-11T12:04:15", "20260111_120415"
                if 'T' in str(timestamp_str):
                    dt = datetime.fromisoformat(str(timestamp_str).replace('Z', ''))
                elif '_' in str(timestamp_str):
                    dt = datetime.strptime(str(timestamp_str), '%Y%m%d_%H%M%S')
                else:
                    dt = datetime.fromisoformat(str(timestamp_str))
                file_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                # If parsing fails, use as is
                file_datetime = str(timestamp_str)
        
        # Method 2: Try to extract from filename if not found in column
        if not file_datetime:
            datetime_match = re.search(r'_(\d{8})_(\d{6})', filename)
            if datetime_match:
                date_str = datetime_match.group(1)  # 20260111
                time_str = datetime_match.group(2)  # 120415
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                file_datetime = f"{formatted_date} {formatted_time}"
        
        # Detect patches automatically
        old_patch, new_patch = detect_patch_columns(df)
        
        # Store in session state to access outside sidebar
        if 'old_patch' not in st.session_state or 'new_patch' not in st.session_state:
            st.session_state.old_patch = old_patch
            st.session_state.new_patch = new_patch
            st.session_state.file_datetime = file_datetime
        
        old_patch = st.session_state.old_patch
        new_patch = st.session_state.new_patch
        file_datetime = st.session_state.get('file_datetime', None)
        
        if old_patch and new_patch:
            # Display patches being compared in main area
            st.markdown(f"""
            <div style='text-align: center; background-color: #2d2d2d; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <span style='color: #888; font-size: 1.0em;'>Analyzing patches:</span><br/>
                <span style='color: #FFA500; font-size: 1.4em; font-weight: bold;'>{old_patch}</span>
                <span style='color: #888; font-size: 1.2em;'> → </span>
                <span style='color: #4CAF50; font-size: 1.4em; font-weight: bold;'>{new_patch}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"🔍 Detected patches: **{old_patch}** → **{new_patch}**")
            
            # Let user override if needed
            with st.expander("🔧 Manual patch selection"):
                all_patches = sorted(set([old_patch, new_patch]))
                # Try to find all possible patches
                for col in df.columns:
                    if 'winrate_' in col:
                        suffix = col.replace('winrate_', '')
                        if suffix not in all_patches:
                            all_patches.append(suffix)
                
                all_patches = sorted(all_patches)
                
                old_patch_manual = st.selectbox("Old patch/snapshot", all_patches, index=0)
                new_patch_manual = st.selectbox("New patch/snapshot", all_patches, index=min(1, len(all_patches)-1))
                
                if st.button("Apply manual selection"):
                    st.session_state.old_patch = old_patch_manual
                    st.session_state.new_patch = new_patch_manual
                    old_patch = old_patch_manual
                    new_patch = new_patch_manual
                    st.success(f"✅ Using: {old_patch} → {new_patch}")
                    st.rerun()
        else:
            st.error("❌ Could not detect patch columns. Please check your CSV format.")
            st.stop()
        
        # Build column names dynamically
        col_mapping = {
            'old_wr': f'winrate_{old_patch}',
            'new_wr': f'winrate_{new_patch}',
            'old_pr': f'pickrate_{old_patch}',
            'new_pr': f'pickrate_{new_patch}',
            'old_br': f'banrate_{old_patch}',
            'new_br': f'banrate_{new_patch}',
            'old_games': f'games_{old_patch}',
            'new_games': f'games_{new_patch}',
            'old_patch_col': f'patch_{old_patch}' if f'patch_{old_patch}' in df.columns else None,
            'new_patch_col': f'patch_{new_patch}' if f'patch_{new_patch}' in df.columns else None,
        }
        
        # Calculate changes if not present
        if 'wr_change' not in df.columns and col_mapping['old_wr'] in df.columns and col_mapping['new_wr'] in df.columns:
            df['wr_change'] = df[col_mapping['new_wr']] - df[col_mapping['old_wr']]
        if 'pr_change' not in df.columns and col_mapping['old_pr'] in df.columns and col_mapping['new_pr'] in df.columns:
            df['pr_change'] = df[col_mapping['new_pr']] - df[col_mapping['old_pr']]
        if 'br_change' not in df.columns and col_mapping['old_br'] in df.columns and col_mapping['new_br'] in df.columns:
            df['br_change'] = df[col_mapping['new_br']] - df[col_mapping['old_br']]
        
        st.markdown("---")
        st.header("🎯 Filters")
        
        # Filters
        roles = ['All'] + sorted(df['lane'].unique().tolist())
        selected_role = st.selectbox("Select a role", roles)
        
        # Choice between slider and manual input
        input_type = st.radio("Input type", ["Slider", "Manual"], horizontal=True)
        
        if input_type == "Slider":
            min_games = st.slider(
                f"Minimum games ({new_patch})",
                min_value=0,
                max_value=int(df[col_mapping['new_games']].max()) if col_mapping['new_games'] in df.columns else 1000,
                value=100,
                step=50
            )
        else:
            min_games = st.number_input(
                f"Minimum games ({new_patch})",
                min_value=0,
                max_value=int(df[col_mapping['new_games']].max()) if col_mapping['new_games'] in df.columns else 100000,
                value=100,
                step=10
            )
        
        st.markdown("---")
        st.header("📊 Graph Type")
        graph_type = st.radio(
            "Choose analysis type",
            ["Top changes", "Scatter plot", "Compare by role", "Specific champions"]
        )
        
        if graph_type == "Top changes":
            top_n = st.slider("Number of champions to display", 10, 50, 20, 5)
            metric_choice = st.selectbox(
                "Metric to analyze",
                ["Winrate", "Pickrate", "Banrate"]
            )
            
            # Advanced filters
            st.markdown("### Advanced Filters")
            
            # WR change filter
            wr_filter = st.slider(
                "Minimum WR change (absolute)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                help="Display only champions with change >= this value"
            )
        
        elif graph_type == "Specific champions":
            champions = sorted(df['champion'].unique().tolist())
            selected_champions = st.multiselect(
                "Select champions",
                champions,
                max_selections=10
            )

# Main content
if uploaded_file is None:
    st.info("👈 Please load a CSV file to start the analysis")
    st.markdown("""
    ### Expected CSV format:
    The app will automatically detect your patch columns!
    
    **Required columns:**
    - `champion`: Champion name
    - `lane`: Role (top, jungle, middle, bottom, support)
    
    **Metric columns (with patch identifier):**
    - `winrate_X`, `winrate_Y`: Win rates
    - `pickrate_X`, `pickrate_Y`: Pick rates
    - `banrate_X`, `banrate_Y`: Ban rates
    - `games_X`, `games_Y`: Number of games
    
    **Examples of valid patch identifiers:**
    - Patches: `15.24`, `16.1`, `14.23`
    - Dates: `2025-01-10`, `2025-01-11`
    - Times: `10h`, `14h`, `20h`
    - Custom: `before`, `after`, `snapshot1`, `snapshot2`
    
    The app will compare the first two patches it detects automatically!
    """)
    
else:
    # Display patches being compared at the top
    if 'old_patch' in st.session_state and 'new_patch' in st.session_state:
        old_label = st.session_state.old_patch
        new_label = st.session_state.new_patch
        file_datetime = st.session_state.get('file_datetime', None)
        
        # Format labels nicely - try to detect if it's a date/time
        def format_patch_label(patch_str):
            # If it looks like a date (contains dashes or slashes)
            if '-' in patch_str or '/' in patch_str:
                return f"📅 {patch_str}"
            # If it contains 'h' it might be a time
            elif 'h' in patch_str.lower() or ':' in patch_str:
                return f"🕐 {patch_str}"
            # Otherwise just return as is (patch number)
            else:
                return f"Patch {patch_str}"
        
        old_formatted = format_patch_label(old_label)
        new_formatted = format_patch_label(new_label)
        
        # Build datetime display if available
        datetime_display = ""
        if file_datetime:
            datetime_display = f"<div style='margin-top: 10px; color: #666; font-size: 0.85em;'>📅 Analysis date: {file_datetime}</div>"
        
        st.markdown(f"""
        <div style='text-align: center; background-color: #2d2d2d; padding: 15px; border-radius: 10px; margin-bottom: 25px; border: 2px solid #444;'>
            <span style='color: #aaa; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;'>Comparing</span><br/>
            <span style='color: #FFA500; font-size: 1.4em; font-weight: bold;'>{old_formatted}</span>
            <span style='color: #888; font-size: 1.3em;'> ➜ </span>
            <span style='color: #4CAF50; font-size: 1.4em; font-weight: bold;'>{new_formatted}</span>
            {datetime_display}
        </div>
        """, unsafe_allow_html=True)
    
    # Filter data
    filtered_df = df.copy()
    
    if selected_role != 'All':
        filtered_df = filtered_df[filtered_df['lane'] == selected_role]
    
    if col_mapping['new_games'] in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col_mapping['new_games']] >= min_games]
    
    # General metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Champions analyzed", len(filtered_df))
    with col2:
        avg_wr_change = filtered_df['wr_change'].mean()
        st.metric("Average WR change", f"{avg_wr_change:.2f}%")
    with col3:
        buffed = len(filtered_df[filtered_df['wr_change'] > 0])
        st.metric("Buffed champions", buffed, delta=f"{buffed/len(filtered_df)*100:.1f}%")
    with col4:
        nerfed = len(filtered_df[filtered_df['wr_change'] < 0])
        st.metric("Nerfed champions", nerfed, delta=f"{nerfed/len(filtered_df)*100:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    
    # Graphs based on selected type
    if graph_type == "Top changes":
        # Determine change column
        change_col = {
            "Winrate": "wr_change",
            "Pickrate": "pr_change",
            "Banrate": "br_change"
        }[metric_choice]
        
        old_col = {
            "Winrate": col_mapping['old_wr'],
            "Pickrate": col_mapping['old_pr'],
            "Banrate": col_mapping['old_br']
        }[metric_choice]
        
        new_col = {
            "Winrate": col_mapping['new_wr'],
            "Pickrate": col_mapping['new_pr'],
            "Banrate": col_mapping['new_br']
        }[metric_choice]
        
        # Filter by minimum change if needed
        filtered_changes = filtered_df.copy()  # Use filtered_df to respect role and min_games filters
        if wr_filter > 0:
            filtered_changes = filtered_changes[filtered_changes['wr_change'].abs() >= wr_filter]
        
        # Separate buffs and nerfs
        buffs = filtered_changes[filtered_changes[change_col] > 0].copy()
        nerfs = filtered_changes[filtered_changes[change_col] < 0].copy()
        
        # Get top N based on change magnitude
        buffs = buffs.nlargest(top_n, change_col)  # Biggest positive changes
        nerfs = nerfs.nsmallest(top_n, change_col)  # Biggest negative changes (most negative)
        
        # Sort for display
        buffs = buffs.sort_values(change_col, ascending=True)  # Smallest buff at bottom, biggest at top
        nerfs = nerfs.sort_values(change_col, ascending=False)  # Biggest nerf at TOP, smallest at bottom (inverted!)
        
        # Function to create a chart
        def create_change_chart(data, title_suffix, color_main):
            if len(data) == 0:
                return None
                
            fig = go.Figure()
            
            # Labels - use index numbers instead of names (images will show champions)
            labels = list(range(len(data)))
            
            # Very light gray bars (current patch value)
            fig.add_trace(go.Bar(
                y=labels,
                x=data[new_col],
                name=f'{metric_choice} {new_patch}',
                orientation='h',
                marker=dict(
                    color='#F8F8F8',
                    opacity=1.0,
                    line=dict(color='#E0E0E0', width=1)
                ),
                text='',
                textposition='inside',
                customdata=[[f"{row['champion']} ({row['lane']})", row[new_col]] for _, row in data.iterrows()],
                hovertemplate='%{customdata[0]}<br>' + f'{metric_choice}: %{{customdata[1]:.2f}}%<extra></extra>'
            ))
            
            # Add champion images
            for i, (idx, row) in enumerate(data.iterrows()):
                champ_name = row['champion']
                champ_formatted = champ_name.capitalize().replace("'", "").replace(" ", "")
                
                # Special champion name mappings
                special_names = {
                    'Wukong': 'MonkeyKing',
                    'Nunu': 'Nunu',
                    'Renata': 'Renata',
                    'Belveth': 'Belveth',
                    'Kogmaw': 'KogMaw',
                    'Leesin': 'LeeSin',
                    'Masteryi': 'MasterYi',
                    'Missfortune': 'MissFortune',
                    'Twistedfate': 'TwistedFate',
                    'Tahmkench': 'TahmKench',
                    'Xinzhao': 'XinZhao',
                    'Aurelionsol': 'AurelionSol',
                    'Chogath': 'Chogath',
                    'Drmundo': 'DrMundo',
                    'Jarvaniv': 'JarvanIV',
                    'Khazix': 'Khazix',
                    'Reksai': 'RekSai',
                    'Velkoz': 'Velkoz',
                    'Kaisa': 'Kaisa',
                    'Ksante': 'KSante'
                }
                if champ_formatted in special_names:
                    champ_formatted = special_names[champ_formatted]
                
                img_url = f"https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/{champ_formatted}.png"
                
                fig.add_layout_image(
                    dict(
                        source=img_url,
                        xref="paper",
                        yref="y",
                        x=-0.02,
                        y=i,
                        sizex=0.025,
                        sizey=0.8,
                        xanchor="right",
                        yanchor="middle",
                        layer="above"
                    )
                )
            
            # Add colored bars and annotations
            for i, (idx, row) in enumerate(data.iterrows()):
                change_value = row[change_col]
                wr_final = row[new_col]
                
                # Colored bar overlay
                fig.add_shape(
                    type="rect",
                    x0=wr_final - abs(change_value),
                    y0=i - 0.4,
                    x1=wr_final,
                    y1=i + 0.4,
                    fillcolor=color_main,
                    opacity=0.9,
                    line=dict(width=0)
                )
                
                # Change value text on colored bar
                fig.add_annotation(
                    x=wr_final - (abs(change_value) / 2),
                    y=i,
                    text=f"<b>{change_value:+.2f}</b>",
                    xanchor='center',
                    showarrow=False,
                    font=dict(size=13, color='white', family='Arial Black')
                )
                
                # Final WR with background
                max_x = data[new_col].max()
                fig.add_annotation(
                    x=max_x + 4,
                    y=i,
                    text=f"<b>{wr_final:.2f}%</b>",
                    xanchor='left',
                    showarrow=False,
                    font=dict(size=14, color='white', family='Arial'),
                    bgcolor='#3a3a3a',
                    bordercolor='#555555',
                    borderwidth=1,
                    borderpad=6,
                    opacity=0.9
                )
            
            fig.update_layout(
                title=dict(
                    text=f"Top {len(data)} {title_suffix} - {metric_choice} Changes ({old_patch} → {new_patch})",
                    font=dict(size=22, family='Arial Black', color='white')
                ),
                xaxis_title=dict(
                    text=f"{metric_choice} (%)",
                    font=dict(color='white')
                ),
                yaxis_title=dict(
                    text=" ",
                    font=dict(color='white')
                ),
                height=max(600, len(data) * 35),
                barmode='overlay',
                showlegend=False,
                hovermode='y unified',
                margin=dict(l=180, r=100, t=100, b=80),
                xaxis=dict(
                    range=[40, data[new_col].max() + 5],
                    title_font=dict(size=16, color='white'),
                    tickfont=dict(color='white'),
                    gridcolor='#333333'
                ),
                yaxis=dict(
                    title_font=dict(size=16, color='white'),
                    tickfont=dict(size=13, color='white'),
                    gridcolor='#333333',
                    showticklabels=False
                ),
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(size=14, color='white')
            )
            
            return fig
        
        # Create both charts
        st.subheader("🟢 Biggest Buffs")
        fig_buffs = create_change_chart(buffs, "Buffs", '#44DD44')
        if fig_buffs:
            st.plotly_chart(fig_buffs, width="stretch")
            
            # Data table for buffs
            with st.expander("📋 Detailed Data - Buffs"):
                display_cols = ['champion', 'lane', old_col, new_col, change_col, col_mapping['new_games']]
                st.dataframe(
                    buffs[display_cols].sort_values(change_col, ascending=False),
                    width="stretch"
                )
        else:
            st.info("No champions with positive changes found")
        
        st.markdown("---")
        
        st.subheader("🔴 Biggest Nerfs")
        fig_nerfs = create_change_chart(nerfs, "Nerfs", '#FF4444')
        if fig_nerfs:
            st.plotly_chart(fig_nerfs, width="stretch")
            
            # Data table for nerfs
            with st.expander("📋 Detailed Data - Nerfs"):
                display_cols = ['champion', 'lane', old_col, new_col, change_col, col_mapping['new_games']]
                st.dataframe(
                    nerfs[display_cols].sort_values(change_col, ascending=True),
                    width="stretch"
                )
        else:
            st.info("No champions with negative changes found")
    
    elif graph_type == "Scatter plot":
        # Scatter plot to see relationships
        fig = px.scatter(
            filtered_df,
            x=col_mapping['old_wr'],
            y=col_mapping['new_wr'],
            size=col_mapping['new_games'],
            color='wr_change',
            hover_data=['champion', 'lane', col_mapping['new_pr']],
            labels={
                col_mapping['old_wr']: f'Winrate {old_patch} (%)',
                col_mapping['new_wr']: f'Winrate {new_patch} (%)',
                'wr_change': 'Change (%)'
            },
            title=f"Winrate Evolution Between Patches ({old_patch} → {new_patch})",
            color_continuous_scale=['red', 'gray', 'green'],
            color_continuous_midpoint=0
        )
        
        # Reference line (no change)
        min_wr = min(filtered_df[col_mapping['old_wr']].min(), filtered_df[col_mapping['new_wr']].min())
        max_wr = max(filtered_df[col_mapping['old_wr']].max(), filtered_df[col_mapping['new_wr']].max())
        fig.add_trace(go.Scatter(
            x=[min_wr, max_wr],
            y=[min_wr, max_wr],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='No change',
            showlegend=True
        ))
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, width="stretch")
    
    elif graph_type == "Compare by role":
        # Boxplot by role
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.box(
                filtered_df if selected_role == 'All' else df[df[col_mapping['new_games']] >= min_games],
                x='lane',
                y='wr_change',
                color='lane',
                title=f"WR Change Distribution by Role ({old_patch} → {new_patch})",
                labels={'wr_change': 'WR Change (%)', 'lane': 'Role'}
            )
            fig1.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig1, width="stretch")
        
        with col2:
            # Average by role
            role_avg = df[df[col_mapping['new_games']] >= min_games].groupby('lane').agg({
                'wr_change': 'mean',
                'pr_change': 'mean',
                'champion': 'count'
            }).reset_index()
            role_avg.columns = ['lane', 'avg_wr_change', 'avg_pr_change', 'count']
            
            fig2 = px.bar(
                role_avg,
                x='lane',
                y='avg_wr_change',
                color='avg_wr_change',
                text='avg_wr_change',
                title=f"Average WR Change by Role ({old_patch} → {new_patch})",
                labels={'avg_wr_change': 'Average WR Change (%)', 'lane': 'Role'},
                color_continuous_scale=['red', 'gray', 'green'],
                color_continuous_midpoint=0
            )
            fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, width="stretch")
        
        # Summary table
        st.subheader("📊 Statistics by Role")
        st.dataframe(role_avg.sort_values('avg_wr_change', ascending=False), width="stretch")
    
    elif graph_type == "Specific champions":
        if not selected_champions:
            st.info("👆 Select champions in the sidebar")
        else:
            champ_data = filtered_df[filtered_df['champion'].isin(selected_champions)]
            
            if len(champ_data) == 0:
                st.warning("No data found for these champions with current filters")
            else:
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Winrate Change', 'Pickrate Change', 
                                  'Banrate Change', 'Games Played'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                labels = [f"{row['champion']} ({row['lane']})" for _, row in champ_data.iterrows()]
                colors_wr = ['red' if x < 0 else 'green' for x in champ_data['wr_change']]
                colors_pr = ['red' if x < 0 else 'green' for x in champ_data['pr_change']]
                colors_br = ['red' if x < 0 else 'green' for x in champ_data['br_change']]
                
                # WR Change
                fig.add_trace(
                    go.Bar(y=labels, x=champ_data['wr_change'], orientation='h',
                           marker_color=colors_wr, text=champ_data['wr_change'].round(2),
                           textposition='auto', name='WR Change'),
                    row=1, col=1
                )
                
                # PR Change
                fig.add_trace(
                    go.Bar(y=labels, x=champ_data['pr_change'], orientation='h',
                           marker_color=colors_pr, text=champ_data['pr_change'].round(2),
                           textposition='auto', name='PR Change'),
                    row=1, col=2
                )
                
                # BR Change
                fig.add_trace(
                    go.Bar(y=labels, x=champ_data['br_change'], orientation='h',
                           marker_color=colors_br, text=champ_data['br_change'].round(2),
                           textposition='auto', name='BR Change'),
                    row=2, col=1
                )
                
                # Games
                fig.add_trace(
                    go.Bar(y=labels, x=champ_data[col_mapping['new_games']], orientation='h',
                           marker_color='lightblue', text=champ_data[col_mapping['new_games']],
                           textposition='auto', name='Games'),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text="Change (%)", row=1, col=1)
                fig.update_xaxes(title_text="Change (%)", row=1, col=2)
                fig.update_xaxes(title_text="Change (%)", row=2, col=1)
                fig.update_xaxes(title_text="Number of games", row=2, col=2)
                
                fig.update_layout(
                    height=600,
                    showlegend=False,
                    title_text=f"Detailed Analysis of Selected Champions ({old_patch} → {new_patch})"
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Detailed table
                st.subheader("📋 Champion Details")
                display_cols = ['champion', 'lane', col_mapping['old_wr'], col_mapping['new_wr'], 'wr_change',
                               col_mapping['old_pr'], col_mapping['new_pr'], 'pr_change', col_mapping['new_games']]
                st.dataframe(champ_data[display_cols], width="stretch")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        <p>⚔️ Champion Analysis App | Built with Streamlit & Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)