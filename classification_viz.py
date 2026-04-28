# classification_viz.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_classification_visualizations(df, target_col, sensitive_col, group_stats):
    """Create visualizations for classification bias"""
    
    # Encode target to numeric if needed
    if df[target_col].dtype == 'object':
        y_temp = pd.factorize(df[target_col])[0]
    else:
        y_temp = df[target_col]
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'sensitive_group': df[sensitive_col],
        'target': y_temp,
        'target_label': df[target_col]
    })
    
    # 1. GROUP DISTRIBUTION BAR CHART
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Group Size Distribution', 'Outcome Rate by Group'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Group sizes
    group_sizes = plot_df['sensitive_group'].value_counts()
    fig1.add_trace(
        go.Bar(x=group_sizes.index, y=group_sizes.values, 
               marker_color='#f5c842', name='Group Size'),
        row=1, col=1
    )
    
    # Outcome rates
    outcome_rates = plot_df.groupby('sensitive_group')['target'].mean()
    colors = ['#4caf50' if rate < 0.1 else '#ff9800' if rate < 0.2 else '#f44336' 
              for rate in outcome_rates.values]
    fig1.add_trace(
        go.Bar(x=outcome_rates.index, y=outcome_rates.values,
               marker_color=colors, name='Outcome Rate'),
        row=1, col=2
    )
    
    fig1.update_layout(
        height=450,
        showlegend=False,
        title_text="Group Distribution & Outcome Analysis",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0')
    )
    fig1.update_yaxes(title_text="Count", row=1, col=1)
    fig1.update_yaxes(title_text="Rate (0-1)", row=1, col=2, tickformat='.0%')
    
    # 2. GAP CHART - Difference from average
    overall_rate = plot_df['target'].mean()
    group_stats['deviation'] = group_stats['Positive_Rate'] - overall_rate
    
    fig2 = go.Figure()
    colors_gap = ['#4caf50' if x >= 0 else '#f44336' for x in group_stats['deviation']]
    fig2.add_trace(go.Bar(
        x=group_stats['group'],
        y=group_stats['deviation'],
        marker_color=colors_gap,
        text=[f"{d:+.1%}" for d in group_stats['deviation']],
        textposition='outside'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="#f5c842")
    fig2.update_layout(
        title="How Each Group Deviates from the Overall Average",
        xaxis_title="Sensitive Group",
        yaxis_title="Deviation from Average Outcome Rate",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0'),
        showlegend=False
    )
    fig2.update_yaxes(tickformat='.1%')
    
    # 3. STACKED BAR - Imbalance visualization
    contingency = pd.crosstab(plot_df['sensitive_group'], plot_df['target_label'])
    fig3 = go.Figure()
    
    for col in contingency.columns:
        fig3.add_trace(go.Bar(
            name=str(col),
            x=contingency.index,
            y=contingency[col],
            text=contingency[col],
            textposition='inside'
        ))
    
    fig3.update_layout(
        barmode='stack',
        title="Outcome Distribution Across Groups (Stacked View)",
        xaxis_title="Sensitive Group",
        yaxis_title="Count",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0'),
        legend_title="Outcome"
    )
    
    # 4. PARITY LINE CHART
    fig4 = go.Figure()
    groups = group_stats['group'].tolist()
    rates = group_stats['Positive_Rate'].tolist()
    
    fig4.add_trace(go.Scatter(
        x=groups, y=rates,
        mode='lines+markers+text',
        name='Outcome Rate',
        line=dict(color='#f5c842', width=3),
        marker=dict(size=12, color='#f5c842', symbol='circle'),
        text=[f"{r:.1%}" for r in rates],
        textposition='top center'
    ))
    
    # Add ideal parity line
    ideal_rate = overall_rate
    fig4.add_hline(y=ideal_rate, line_dash="dash", 
                   line_color="#888", 
                   annotation_text=f"Overall Rate: {ideal_rate:.1%}")
    
    fig4.update_layout(
        title="Parity Analysis: Outcome Rates Across Groups",
        xaxis_title="Sensitive Group",
        yaxis_title="Positive Outcome Rate",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0'),
        showlegend=True
    )
    fig4.update_yaxes(tickformat='.0%')
    
    # 5. HEATMAP (optional)
    fig5 = None
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2 and len(groups) <= 10:
        corr_by_group = {}
        for group in groups:
            group_data = df[df[sensitive_col] == group][numeric_cols[:5]]
            if len(group_data) > 1 and len(group_data.columns) >= 2:
                try:
                    corr_by_group[group] = group_data.corr().iloc[0, 1]
                except:
                    corr_by_group[group] = 0
        
        if corr_by_group:
            fig5 = go.Figure(data=go.Heatmap(
                z=[list(corr_by_group.values())],
                x=list(corr_by_group.keys()),
                y=['Feature Correlation'],
                colorscale='RdYlGn',
                text=[[f"{v:.3f}" for v in corr_by_group.values()]],
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            fig5.update_layout(
                title="Feature Correlation Differences Across Groups",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(19,19,28,0.5)',
                font=dict(color='#e8e6e0')
            )
    
    return fig1, fig2, fig3, fig4, fig5