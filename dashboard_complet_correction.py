"""
================================================================================
DASHBOARD PROFESSIONNEL : OPTIMISATION DES FLUX DE Liquidité BANCAIRE
================================================================================
Application Streamlit pour visualisation complete du pipeline :
- Forecasting (Predictions)
- Determination Stocks Minimum
- Optimisation Transport
- Analyses Consolidees

Auteur: Systeme de Prevision de Liquidite
Version: 1.0
Date: 2025
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import base64
from datetime import datetime

# Configuration page
st.set_page_config(
    page_title="Optimisation Flux Liquidité",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalise
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #1f4788;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2c5aa0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

CONFIG = {
    'agences': ['00001', '00005', '00010', '00021', '00024', '00026', '00031', 
                '00033', '00034', '00037', '00038', '00039', '00040', '00042', 
                '00043', '00045', '00046', '00055', '00056', '00057', '00062', 
                '00063', '00064', '00073', '00075', '00079', '00081', '00085', 
                '00087'],
    'composantes': ['encaissements', 'decaissements', 'Besoin'],
    'modeles': ['xgboost', 'prophet', 'arima', 'moving_avg', 'ensemble_post'],
    'jours': [1, 2, 3, 4, 5],
    'horizon': 5,  # ← AJOUTE CETTE LIGNE
    'base_dir_metrics': 'resultats_metriques',
    'base_dir_optim': 'optimisation_transport__',
    'stocks_file': 'stocks_minimum_agences.csv'
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_data
def load_predictions(agence, composante):
    """Charge les predictions pour une agence et composante"""
    filepath = Path(CONFIG['base_dir_metrics']) / agence / f"{agence}_predictions_{composante}.csv"
    
    if not filepath.exists():
        return None
    
    return pd.read_csv(filepath)

@st.cache_data
def load_stocks_minimum():
    """Charge le fichier des stocks minimum"""
    filepath = Path(CONFIG['stocks_file'])
    
    if not filepath.exists():
        return None
    
    # return pd.read_csv(filepath)
    return pd.read_csv(filepath, dtype={'Code_Agence': str})

# @st.cache_data
# def load_excel_optimisation(jour):
#     """Charge le fichier Excel d'optimisation pour un jour"""
#     filepath = Path(CONFIG['base_dir_optim']) / f"Optimisation_Transport_J{jour}.xlsx"
    
#     if not filepath.exists():
#         return None
    
#     return pd.read_excel(filepath, sheet_name=None)



@st.cache_data
def load_excel_optimisation(jour):
    """Charge le fichier Excel d'optimisation pour un jour"""
    filepath = Path(CONFIG['base_dir_optim']) / f"Optimisation_Transport_J{jour}.xlsx"
    
    if not filepath.exists():
        return None
    
    excel_data = {}
    
    # ✅ FORCER dtype=str pour les colonnes de codes agence
    excel_data['Donnees entree'] = pd.read_excel(
        filepath, 
        sheet_name='Donnees entree',
        dtype={'Code Agence': str}
    )
    
    excel_data['Optimisation transport'] = pd.read_excel(
        filepath, 
        sheet_name='Optimisation transport',
        dtype=str  # ✅ Tout en string pour cette feuille
    )
    
    excel_data['Matrice couts unitaires'] = pd.read_excel(
        filepath, 
        sheet_name='Matrice couts unitaires',
        dtype=str  # ✅ Tout en string
    )
    
    return excel_data


def format_number(value):
    """Formate un nombre avec separateur de milliers"""
    return f"{value:,.2f}".replace(',', ' ')

def get_file_download_link(filepath, link_text):
    """Genere un lien de telechargement pour un fichier"""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{Path(filepath).name}">{link_text}</a>'
    return href

# ============================================================================
# PAGE 1 : ACCUEIL & VUE D'ENSEMBLE
# ============================================================================

def page_accueil():
    """Page d'accueil avec vue d'ensemble du projet"""
    
    st.markdown('<div class="main-header">OPTIMISATION DES FLUX DE LIQUIDITÉ BANCAIRE</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Objectif du Projet</b><br>
    Developper un systeme intelligent d'optimisation des flux de liquidite entre agences bancaires,
    minimisant les couts de transport tout en garantissant l'equilibre entre besoins et excedents.
    </div>
    """, unsafe_allow_html=True)
    
    # Metriques cles
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Agences", "29", "Reseau complet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Horizon", "5 jours", "J+1 a J+5")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Modeles ML", "6", "Evaluation comparative")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Methode", "CBC Simplexe", "Optimalite garantie")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Architecture du pipeline
    st.markdown('<div class="sub-header">Architecture du Pipeline</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        **Phase 1 : Prevision**
        - Collecte donnees historiques
        - Entrainement 6 modeles ML
        - Selection modele optimal
        - Predictions 5 jours
        
        **Phase 2 : Stocks Minimum**
        - Analyse soldes historiques
        - Calcul percentiles (P1-P99)
        - Determination P5 (stock minimum)
        - Validation statistique
        
        **Phase 3 : Optimisation**
        - Generation matrice couts
        - Calcul flux optimaux
        - Resolution CBC Simplexe
        - Allocation inter-agences
        """)
    
    with col2:
        # Diagramme du pipeline
        fig = go.Figure()
        
        # Etapes du pipeline
        etapes = [
            "Donnees Historiques",
            "Forecasting ML",
            "Selection Modele",
            "Stocks Minimum",
            "Calcul Flux",
            "Optimisation CBC",
            "Allocation Optimale"
        ]
        
        y_positions = list(range(len(etapes), 0, -1))
        
        fig.add_trace(go.Scatter(
            x=[1]*len(etapes),
            y=y_positions,
            mode='markers+text',
            marker=dict(size=40, color='#1f4788'),
            text=etapes,
            textposition="middle right",
            textfont=dict(size=12, color='white'),
            hoverinfo='skip'
        ))
        
        # Fleches
        for i in range(len(y_positions)-1):
            fig.add_annotation(
                x=1, y=y_positions[i],
                ax=1, ay=y_positions[i+1],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='#1f4788'
            )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=500,
            margin=dict(l=20, r=200, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown('<div class="sub-header">Navigation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Forecasting**
        
        Visualisation des predictions de liquidite par agence et comparaison des modeles ML.
        """)
    
    with col2:
        st.info("""
        **Stocks Minimum**
        
        Analyse statistique des soldes et determination des reserves minimales par percentiles.
        """)
    
    with col3:
        st.info("""
        **Optimisation**
        
        Allocation optimale des flux inter-agences avec minimisation des couts de transport.
        """)

# ============================================================================
# PAGE 2 : FORECASTING
# ============================================================================

def page_forecasting():
    """Page de visualisation des predictions"""
    
    st.markdown('<div class="main-header">FORECASTING - PREDICTIONS DE LIQUIDITE</div>', 
                unsafe_allow_html=True)
    
    # Selection agence
    agence_selectionnee = st.selectbox(
        "Selectionner une agence",
        CONFIG['agences'],
        index=0
    )
    
    st.markdown("---")
    
    # Charger donnees pour l'agence
    # tabs = st.tabs(["Encaissements", "Decaissements", "Besoin", "Comparaison Modeles"])
    tabs = st.tabs(["Encaissements", "Decaissements", "Besoin"])
    
    for idx, composante in enumerate(CONFIG['composantes']):
        with tabs[idx]:
            df_pred = load_predictions(agence_selectionnee, composante)
            
            if df_pred is None:
                st.warning(f"Donnees indisponibles pour {agence_selectionnee} - {composante}")
                continue
            
            # Graphique predictions vs reel
            fig = go.Figure()
            
            # Valeurs reelles
            fig.add_trace(go.Scatter(
                x=df_pred['Jour'],
                y=df_pred['Reel'],
                mode='lines+markers',
                name='Reel',
                line=dict(color='black', width=3),
                marker=dict(size=10)
            ))
            
            # Predictions des modeles
            colors = {
                'xgboost': '#1f77b4',
                'prophet': '#2ca02c',
                'arima': '#ff7f0e',
                'moving_avg': '#9467bd',
                'ensemble_post': '#d62728'
            }
            
            for modele in CONFIG['modeles']:
                if modele in df_pred.columns:
                    fig.add_trace(go.Scatter(
                        x=df_pred['Jour'],
                        y=df_pred[modele],
                        mode='lines+markers',
                        name=modele.upper(),
                        line=dict(color=colors.get(modele, '#8c564b'), width=2),
                        marker=dict(size=8)
                    ))
            
            fig.update_layout(
                title=f"{composante.capitalize()} - Agence {agence_selectionnee}",
                xaxis_title="Jour",
                yaxis_title="Montant (Millions FCFA)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metriques MAPE
            st.markdown("**Metriques de Performance (MAPE)**")
            
            metriques = []
            for modele in CONFIG['modeles']:
                if modele in df_pred.columns:
                    pred = df_pred[modele].values
                    reel = df_pred['Reel'].values
                    
                    mask = np.abs(reel) > 1e-8
                    if mask.any():
                        mape = np.mean(np.abs((reel[mask] - pred[mask]) / reel[mask])) * 100
                        metriques.append({'Modele': modele.upper(), 'MAPE (%)': round(mape, 2)})
            
            if metriques:
                df_metriques = pd.DataFrame(metriques).sort_values('MAPE (%)')
                st.dataframe(df_metriques, use_container_width=True)
    
    # # Onglet comparaison
    # with tabs[3]:
    #     st.markdown("**Comparaison Globale des Modeles**")
        
    #     # Charger metriques forecast
    #     filepath_metrics = Path(CONFIG['base_dir_metrics']) / agence_selectionnee / f"{agence_selectionnee}_metriques_forecast.csv"
        
    #     if filepath_metrics.exists():
    #         df_metrics = pd.read_csv(filepath_metrics)
            
    #         # Filtrer pour l'agence
    #         df_metrics_agence = df_metrics[df_metrics['Code_Agence'] == agence_selectionnee]
            
    #         if not df_metrics_agence.empty:
    #             # Graphique radar
    #             modeles_uniques = df_metrics_agence['Modele'].unique()
                
    #             fig = go.Figure()
                
    #             for modele in modeles_uniques:
    #                 df_modele = df_metrics_agence[df_metrics_agence['Modele'] == modele]
                    
    #                 categories = df_modele['Colonne'].tolist()
    #                 mapes = df_modele['MAPE'].tolist()
                    
    #                 fig.add_trace(go.Scatterpolar(
    #                     r=mapes,
    #                     theta=categories,
    #                     fill='toself',
    #                     name=modele.upper()
    #                 ))
                
    #             fig.update_layout(
    #                 polar=dict(
    #                     radialaxis=dict(visible=True, range=[0, max(df_metrics_agence['MAPE'])*1.1])
    #                 ),
    #                 showlegend=True,
    #                 title="Comparaison MAPE par Composante",
    #                 height=500
    #             )
                
    #             st.plotly_chart(fig, use_container_width=True)
                
    #             # Tableau detaille
    #             st.dataframe(df_metrics_agence[['Colonne', 'Modele', 'MAE', 'RMSE', 'MAPE', 'R2']], 
    #                        use_container_width=True)
    #         else:
    #             st.info("Donnees de metriques indisponibles pour cette agence")
    #     else:
    #         st.warning("Fichier de metriques introuvable")

# ============================================================================
# PAGE 3 : STOCKS MINIMUM
# ============================================================================

def page_stocks_minimum():
    """Page d'analyse des stocks minimum"""
    
    st.markdown('<div class="main-header">DETERMINATION DES STOCKS MINIMUM</div>', 
                unsafe_allow_html=True)
    
    # Charger donnees stocks
    df_stocks = load_stocks_minimum()
    
    if df_stocks is None:
        st.error("Fichier stocks_minimum_agences.csv introuvable")
        return
    
    # Selection agence
    agence_selectionnee = st.selectbox(
        "Selectionner une agence",
        CONFIG['agences'],
        index=0,
        key='stocks_agence'
    )
    
    # Donnees de l'agence
    data_agence = df_stocks[df_stocks['Code_Agence'] == agence_selectionnee].iloc[0]
    
    st.markdown("---")
    
    # Informations agence
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Stock Minimum (P5)", 
                 f"{abs(data_agence['Stock_Minimum_Millions']):,.2f} M",
                 help="5eme percentile - Standard Bale III")
    
    with col2:
        st.metric("Mediane (P50)", 
                 f"{abs(data_agence['P50_Mediane_M']):,.2f} M")
    
    with col3:
        st.metric("Observations", 
                 f"{data_agence['Nb_Observations']}")
    
    st.markdown("---")
    
    # Graphiques
    tabs = st.tabs(["Distribution", "Boxplot", "Percentiles", "Comparaison Agences"])
    
    # TAB 1: Histogramme
    with tabs[0]:
        st.markdown("**Distribution des Soldes Historiques avec Percentiles**")
        
        st.markdown("""
        <div class="info-box">
        <b>Interpretation :</b> Ce graphique montre la frequence des soldes observes.
        Les lignes verticales representent les percentiles cles (P1, P5, P10, P25, P50, P75, P90, P95, P99).
        Le stock minimum (P5) est la ligne rouge foncee.
        </div>
        """, unsafe_allow_html=True)
        
        # Note: On ne peut pas regenerer l'histogramme exact sans les donnees brutes
        # On affiche un graphique des percentiles
        
        percentiles_cols = ['P1_M', 'P5_M', 'P10_M', 'P25_M', 'P50_Mediane_M', 
                           'P75_M', 'P90_M', 'P95_M', 'P99_M']
        percentiles_labels = ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']
        percentiles_values = [abs(data_agence[col]) for col in percentiles_cols]
        
        fig = go.Figure()
        
        colors_map = {
            'P1': '#FF0000',
            'P5': '#8B0000',
            'P10': '#FFA500',
            'P25': '#FFD700',
            'P50': '#008000',
            'P75': '#0000FF',
            'P90': '#800080',
            'P95': '#FF00FF',
            'P99': '#FFC0CB'
        }
        
        fig.add_trace(go.Bar(
            x=percentiles_labels,
            y=percentiles_values,
            marker_color=[colors_map[p] for p in percentiles_labels],
            text=[f"{v:,.1f}M" for v in percentiles_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Percentiles - Agence {agence_selectionnee}",
            xaxis_title="Percentile",
            yaxis_title="Solde (Millions FCFA)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Boxplot
    with tabs[1]:
        st.markdown("**Boite a Moustaches avec Quartiles**")
        
        st.markdown("""
        <div class="info-box">
        <b>Interpretation :</b> La boite represente les quartiles (Q1=P25, Q2=P50, Q3=P75).
        Les moustaches s'etendent jusqu'aux valeurs extremes (Min et Max).
        Le stock minimum (P5) est indique par la ligne rouge verticale.
        </div>
        """, unsafe_allow_html=True)
        
        # Valeurs pour boxplot
        q1 = abs(data_agence['P25_M'])
        median = abs(data_agence['P50_Mediane_M'])
        q3 = abs(data_agence['P75_M'])
        minimum = abs(data_agence['Min_M'])
        maximum = abs(data_agence['Max_M'])
        p5 = abs(data_agence['P5_M'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            q1=[q1],
            median=[median],
            q3=[q3],
            lowerfence=[minimum],
            upperfence=[maximum],
            name=f"Agence {agence_selectionnee}",
            marker_color='lightblue',
            boxmean=True
        ))
        
        # Ligne P5
        fig.add_vline(x=p5, line_dash="dash", line_color="darkred", line_width=3,
                     annotation_text=f"Stock Min (P5) = {p5:,.1f}M",
                     annotation_position="top")
        
        fig.update_layout(
            title="Distribution avec Quartiles",
            xaxis_title="Solde (Millions FCFA)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Zones quartiles
        st.markdown("**Repartition par Quartiles**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background-color:#ffcccc;padding:1rem;border-radius:0.5rem;">
            <b>Q1 (25% plus bas)</b><br>
            Min - P25<br>
            {minimum:,.1f}M - {q1:,.1f}M
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color:#ffffcc;padding:1rem;border-radius:0.5rem;">
            <b>Q2 (25% suivants)</b><br>
            P25 - P50<br>
            {q1:,.1f}M - {median:,.1f}M
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color:#ccffcc;padding:1rem;border-radius:0.5rem;">
            <b>Q3 (25% suivants)</b><br>
            P50 - P75<br>
            {median:,.1f}M - {q3:,.1f}M
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background-color:#ccccff;padding:1rem;border-radius:0.5rem;">
            <b>Q4 (25% plus hauts)</b><br>
            P75 - Max<br>
            {q3:,.1f}M - {maximum:,.1f}M
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 3: Tableau percentiles
    with tabs[2]:
        st.markdown("**Tableau Complet des Percentiles**")
        
        percentiles_data = {
            'Percentile': percentiles_labels + ['Min', 'Max', 'Moyenne', 'Ecart-Type'],
            'Valeur (Millions)': [abs(data_agence[col]) for col in percentiles_cols] + 
                                [abs(data_agence['Min_M']), abs(data_agence['Max_M']),
                                 abs(data_agence['Moyenne_M']), abs(data_agence['Ecart_Type_M'])]
        }
        
        df_percentiles = pd.DataFrame(percentiles_data)
        df_percentiles['Valeur (Millions)'] = df_percentiles['Valeur (Millions)'].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(df_percentiles, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <b>Stock Minimum Recommande (P5) :</b> {:.2f} Millions FCFA<br><br>
        <b>Justification :</b> Le 5eme percentile (P5) represente le niveau en dessous duquel
        l'agence descend seulement 5% du temps. Ce seuil assure une protection a 95% tout en
        evitant une immobilisation excessive de capital.
        </div>
        """.format(abs(data_agence['Stock_Minimum_Millions'])), unsafe_allow_html=True)
    
    # TAB 4: Comparaison agences
    with tabs[3]:
        st.markdown("**Comparaison Stocks Minimum - Toutes Agences**")
        
        df_stocks_sorted = df_stocks.sort_values('Stock_Minimum_Millions')
        df_stocks_sorted['Stock_Abs'] = df_stocks_sorted['Stock_Minimum_Millions'].abs()
        
        fig = go.Figure()
        
        colors = ['darkred' if i < 5 else 'darkgreen' if i >= len(df_stocks_sorted)-5 
                 else 'steelblue' for i in range(len(df_stocks_sorted))]
        
        fig.add_trace(go.Bar(
            y=df_stocks_sorted['Code_Agence'],
            x=df_stocks_sorted['Stock_Abs'],
            orientation='h',
            marker_color=colors,
            text=df_stocks_sorted['Stock_Abs'].apply(lambda x: f"{x:,.0f}M"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Stocks Minimum par Agence (P5)",
            xaxis_title="Stock Minimum (Millions FCFA)",
            yaxis_title="Code Agence",
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legende
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background-color:#8B0000;color:white;padding:0.5rem;text-align:center;">
            5 stocks les plus eleves
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color:#4682B4;color:white;padding:0.5rem;text-align:center;">
            Autres agences
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background-color:#006400;color:white;padding:0.5rem;text-align:center;">
            5 stocks les plus bas
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 4 : OPTIMISATION TRANSPORT
# ============================================================================

def page_optimisation():
    """Page d'optimisation des flux de transport"""
    
    st.markdown('<div class="main-header">OPTIMISATION DES FLUX DE TRANSPORT</div>', 
                unsafe_allow_html=True)
    
    # Selection jour
    jour_selectionne = st.selectbox(
        "Selectionner un jour",
        CONFIG['jours'],
        format_func=lambda x: f"J+{x}",
        index=0
    )
    
    # Charger donnees Excel
    excel_data = load_excel_optimisation(jour_selectionne)
    
    if excel_data is None:
        st.error(f"Fichier Optimisation_Transport_J{jour_selectionne}.xlsx introuvable")
        return
    
    st.markdown("---")
    
    # Onglets
    tabs = st.tabs(["Donnees Entree", "Repartition & Matrice", "Couts Unitaires", "Synthese"])
    
    # TAB 1: Donnees d'entree
    with tabs[0]:
        st.markdown("**Feuille 1 : Donnees d'Entree**")
        
        df_entree = excel_data['Donnees entree']
        
        # Affichage tableau
        st.dataframe(df_entree.style.format({
            'Solde Caisse veille': '{:,.2f}',
            'Besoin en liquidite': '{:,.2f}',
            'Score de precision': '{:,.2f}',
            'Flux transport optimal': '{:,.2f}'
        }), use_container_width=True)
        
        # Statistiques
        st.markdown("**Statistiques**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        nb_besoin = len(df_entree[df_entree['Type de flux'] == 'Besoin'])
        nb_excedent = len(df_entree[df_entree['Type de flux'] == 'Excedent'])
        total_flux_positifs = df_entree[df_entree['Flux transport optimal'] > 0]['Flux transport optimal'].sum()
        total_flux_negatifs = abs(df_entree[df_entree['Flux transport optimal'] < 0]['Flux transport optimal'].sum())
        
        with col1:
            st.metric("Agences en Besoin", nb_besoin)
        
        with col2:
            st.metric("Agences en Excedent", nb_excedent)
        
        # with col3:
        #     st.metric








        with col3:
            st.metric("Total Besoin", f"{total_flux_positifs:,.2f} M")
        
        with col4:
            st.metric("Total Excedent", f"{total_flux_negatifs:,.2f} M")
    
    
    


    # TAB 2: Repartition et Matrice
    with tabs[1]:
        df_optim = excel_data['Optimisation transport']
        
        st.markdown("**Repartition Besoin / Excedent**")
        
        # ✅ EXTRACTION SIMPLE - Lignes 2 à 30 (index 2 à 30)
        besoin_dict = {}
        excedent_dict = {}
        
        for idx in range(2, 31):  # Lignes 3-31 dans Excel
            if idx >= len(df_optim):
                break
            
            # Colonne A-B : Besoin
            code_a = df_optim.iloc[idx, 0]
            montant_b = df_optim.iloc[idx, 1]
            
            if pd.notna(code_a) and pd.notna(montant_b):
                code_clean = str(code_a).strip()
                if code_clean and code_clean != 'nan':
                    try:
                        montant = float(montant_b)
                        if montant > 0:
                            besoin_dict[code_clean] = montant
                    except:
                        pass
            
            # Colonne D-E : Excédent
            code_d = df_optim.iloc[idx, 3]
            montant_e = df_optim.iloc[idx, 4]
            
            if pd.notna(code_d) and pd.notna(montant_e):
                code_clean = str(code_d).strip()
                if code_clean and code_clean != 'nan':
                    try:
                        montant = float(montant_e)
                        if montant > 0:
                            excedent_dict[code_clean] = montant
                    except:
                        pass
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            if besoin_dict:
                df_besoin = pd.DataFrame(list(besoin_dict.items()), 
                                        columns=['Agence', 'Montant'])
                df_besoin = df_besoin.sort_values('Montant', ascending=False)
                
                fig_besoin = go.Figure()
                fig_besoin.add_trace(go.Bar(
                    x=df_besoin['Agence'],
                    y=df_besoin['Montant'],
                    marker_color='#FF6B6B',
                    text=df_besoin['Montant'].apply(lambda x: f"{x:,.0f}M"),
                    textposition='outside'
                ))
                
                fig_besoin.update_layout(
                    title="Agences en Besoin",
                    xaxis_title="Agence",
                    yaxis_title="Montant (Millions)",
                    height=400
                )
                
                st.plotly_chart(fig_besoin, use_container_width=True)
        
        with col2:
            if excedent_dict:
                df_excedent = pd.DataFrame(list(excedent_dict.items()), 
                                        columns=['Agence', 'Montant'])
                df_excedent = df_excedent.sort_values('Montant', ascending=False)
                
                fig_excedent = go.Figure()
                fig_excedent.add_trace(go.Bar(
                    x=df_excedent['Agence'],
                    y=df_excedent['Montant'],
                    marker_color='#4ECDC4',
                    text=df_excedent['Montant'].apply(lambda x: f"{x:,.0f}M"),
                    textposition='outside'
                ))
                
                fig_excedent.update_layout(
                    title="Agences en Excedent",
                    xaxis_title="Agence",
                    yaxis_title="Montant (Millions)",
                    height=400
                )
                
                st.plotly_chart(fig_excedent, use_container_width=True)
        
        st.markdown("---")
        
        # ✅ MATRICE D'ALLOCATION - LECTURE DIRECTE
        st.markdown("**Matrice d'Allocation Optimale**")
        
        st.markdown("""
        <div class="info-box">
        <b>Interpretation :</b> Montants transférés des agences en excédent (lignes) 
        vers les agences en besoin (colonnes). Valeurs en Millions FCFA.
        </div>
        """, unsafe_allow_html=True)
        
        # Trouver ligne "MATRICE D'ALLOCATION OPTIMALE"
        matrice_start = None
        for idx in range(len(df_optim)):
            val = str(df_optim.iloc[idx, 0]).upper()
            if 'MATRICE' in val and 'ALLOCATION' in val:
                matrice_start = idx + 2  # Sauter titre + ligne vide
                break
        
        if matrice_start and matrice_start < len(df_optim):
            # ✅ LIRE HEADERS (ligne matrice_start)
            headers_row = df_optim.iloc[matrice_start]
            agences_besoin_headers = []
            
            for col_idx in range(1, len(headers_row)):
                val = headers_row.iloc[col_idx]
                if pd.notna(val) and str(val).strip():
                    code = str(val).strip()
                    if 'TOTAL' not in code.upper():
                        agences_besoin_headers.append(code)
                    else:
                        break  # Arrêter à "Total Excedent"
            
            # ✅ LIRE LIGNES AGENCES EXCÉDENT
            matrice_data = []
            agences_excedent_list = []
            
            for row_idx in range(matrice_start + 1, len(df_optim)):
                code_ligne = df_optim.iloc[row_idx, 0]
                
                if pd.isna(code_ligne) or not str(code_ligne).strip():
                    continue
                
                code_ligne_str = str(code_ligne).strip()
                
                # Arrêter si "Total Besoin"
                if 'TOTAL' in code_ligne_str.upper():
                    break
                
                # Construire ligne
                row_dict = {'Agence Excédent': code_ligne_str}
                agences_excedent_list.append(code_ligne_str)
                
                # ✅ Lire valeurs pour chaque colonne + calculer total ligne
                total_ligne = 0
                for j, agence_bes in enumerate(agences_besoin_headers):
                    val = df_optim.iloc[row_idx, j + 1]
                    try:
                        montant = float(val)
                        if montant > 0:
                            row_dict[agence_bes] = montant
                            total_ligne += montant
                        else:
                            row_dict[agence_bes] = ''
                    except:
                        row_dict[agence_bes] = ''
                
                # ✅ Total Excédent = somme de la ligne
                row_dict['Total Excédent'] = total_ligne
                
                matrice_data.append(row_dict)
            
            # Créer DataFrame avant de calculer les totaux
            df_matrice = pd.DataFrame(matrice_data)
            
            # ✅ LIGNE TOTAUX (calculés depuis la matrice)
            total_row = {'Agence Excédent': 'Total Besoin'}
            
            # Pour chaque colonne (agence en besoin), sommer tous les montants reçus
            for agence_bes in agences_besoin_headers:
                total_colonne = 0
                for row_idx in range(len(agences_excedent_list)):
                    val = df_matrice.iloc[row_idx][agence_bes]
                    try:
                        if val != '' and pd.notna(val):
                            total_colonne += float(val)
                    except:
                        pass
                total_row[agence_bes] = total_colonne
            
            # Pour Total Excédent, sommer toute la dernière colonne
            total_excedent_global = 0
            for row_idx in range(len(agences_excedent_list)):
                val = df_matrice.iloc[row_idx]['Total Excédent']
                try:
                    if pd.notna(val):
                        total_excedent_global += float(val)
                except:
                    pass
            
            total_row['Total Excédent'] = total_excedent_global
            
            # Ajouter ligne totaux
            df_matrice = pd.concat([df_matrice, pd.DataFrame([total_row])], ignore_index=True)
            
            # ✅ FONCTION DE FORMATAGE PERSONNALISÉE
            def format_cell(val):
                """Formate uniquement les valeurs numériques non-nulles"""
                if val == '' or val is None or pd.isna(val):
                    return ''
                try:
                    return f"{float(val):,.2f}"
                except:
                    return str(val)
            
            # Appliquer le formatage manuellement
            df_display = df_matrice.copy()
            for col in df_display.columns:
                if col != 'Agence Excédent':
                    df_display[col] = df_display[col].apply(format_cell)
            
            # ✅ AFFICHAGE SIMPLE
            st.dataframe(
                df_display.style.set_properties(**{
                    'text-align': 'center',
                    'font-size': '10px'
                }).set_properties(subset=['Agence Excédent'], **{
                    'font-weight': 'bold',
                    'text-align': 'left',
                    'background-color': '#f0f2f6'
                }),
                use_container_width=True,
                height=600
            )
            
            st.markdown("---")
            
            # ✅ Stocker pour TAB 4
            matrice_values = []
            for row_idx in range(len(agences_excedent_list)):
                row_vals = []
                for agence_bes in agences_besoin_headers:
                    val = df_matrice.iloc[row_idx][agence_bes]
                    try:
                        row_vals.append(float(val) if val != '' else 0.0)
                    except:
                        row_vals.append(0.0)
                matrice_values.append(row_vals)
            
            # Variables globales pour TAB 4
            agences_besoin = agences_besoin_headers
            agences_excedent = agences_excedent_list
            
            # Stats
            total_transferts = sum([sum(row) for row in matrice_values])
            nb_transactions = sum([1 for row in matrice_values for v in row if v > 0])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transférts", f"{total_transferts:,.2f} M")
            
            with col2:
                st.metric("Nombre Transactions", nb_transactions)
            
            with col3:
                if nb_transactions > 0:
                    st.metric("Montant Moyen", f"{total_transferts/nb_transactions:,.2f} M")
            
            st.markdown("---")
            
            # Liste transactions
            st.markdown("**Transactions détaillées**")
            
            trans_list = []
            for i, agence_exc in enumerate(agences_excedent_list):
                for j, agence_bes in enumerate(agences_besoin_headers):
                    montant = matrice_values[i][j]
                    if montant > 0:
                        trans_list.append({
                            'De': agence_exc,
                            'Vers': agence_bes,
                            'Montant (M)': montant
                        })
            
            if trans_list:
                df_trans = pd.DataFrame(trans_list).sort_values('Montant (M)', ascending=False)
                st.dataframe(
                    df_trans.style.format({'Montant (M)': '{:,.2f}'}),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("Aucune transaction enregistrée")
        else:
            st.warning("Matrice d'allocation introuvable dans le fichier Excel")

    
    # with tabs[2]:
    #     st.markdown("**Matrice des Couts Unitaires (FCFA par Million)**")
        
    #     df_couts = excel_data['Matrice couts unitaires']
        
    #     st.markdown("""
    #     <div class="info-box">
    #     <b>Interpretation :</b> Cette matrice symetrique contient les couts de transport
    #     unitaires entre chaque paire d'agences. La diagonale est nulle (pas de cout pour
    #     transfert interne). Les valeurs sont en FCFA par million transfere.
    #     </div>
    #     """, unsafe_allow_html=True)
        
    #     # Nettoyer le DataFrame
    #     # La premiere ligne contient le titre, la ligne 3 les headers
    #     if len(df_couts) > 3:
    #         df_couts_clean = df_couts.iloc[3:].copy()
    #         df_couts_clean.columns = ['Agence'] + list(df_couts.iloc[2, 1:])
    #         df_couts_clean = df_couts_clean.set_index('Agence')

    #         # ✅ FORCER LA CONVERSION EN STRING AVEC ZEROS INITIAUX
    #         df_couts_clean.index = df_couts_clean.index.astype(str).str.zfill(5)
    #         df_couts_clean.columns = df_couts_clean.columns.astype(str).str.zfill(5)
            
    #         # Convertir en numerique
    #         df_couts_numeric = df_couts_clean.apply(pd.to_numeric, errors='coerce').fillna(0)
            
    #         # Heatmap
    #         fig_couts = go.Figure(data=go.Heatmap(
    #             z=df_couts_numeric.values,
    #             x=df_couts_numeric.columns,
    #             y=df_couts_numeric.index,
    #             colorscale='YlOrRd',
    #             text=df_couts_numeric.values,
    #             texttemplate="%{text:.0f}",
    #             textfont={"size": 8},
    #             hoverongaps=False,
    #             hovertemplate='De: %{y}<br>Vers: %{x}<br>Cout: %{z:.0f} FCFA/M<extra></extra>'
    #         ))
            
    #         fig_couts.update_layout(
    #             title="Matrice des Couts Unitaires",
    #             xaxis_title="Agence Destination",
    #             yaxis_title="Agence Source",
    #             height=700
    #         )
            
    #         st.plotly_chart(fig_couts, use_container_width=True)
            
    #         # Statistiques couts
    #         couts_non_nuls = df_couts_numeric.values[df_couts_numeric.values > 0]
            
    #         if len(couts_non_nuls) > 0:
    #             col1, col2, col3, col4 = st.columns(4)
                
    #             with col1:
    #                 st.metric("Cout Minimum", f"{couts_non_nuls.min():.0f} FCFA")
                
    #             with col2:
    #                 st.metric("Cout Maximum", f"{couts_non_nuls.max():.0f} FCFA")
                
    #             with col3:
    #                 st.metric("Cout Moyen", f"{couts_non_nuls.mean():.2f} FCFA")
                
    #             with col4:
    #                 st.metric("Cout Median", f"{np.median(couts_non_nuls):.0f} FCFA")
    

    
    # TAB 3: Couts unitaires
    with tabs[2]:
        st.markdown("**Matrice des Coûts Unitaires (FCFA par Million)**")
        
        df_couts = excel_data['Matrice couts unitaires']
        
        st.markdown("""
        <div class="info-box">
        <b>Interprétation :</b> Cette matrice symétrique contient les coûts de transport
        unitaires entre chaque paire d'agences. La diagonale est nulle (pas de coût pour
        transfert interne). Les valeurs sont en FCFA par million transféré.
        </div>
        """, unsafe_allow_html=True)
        
        # ✅ NETTOYAGE SIMPLE
        if len(df_couts) > 3:
            # Extraire la matrice (commence ligne 4, index 3)
            df_couts_clean = df_couts.iloc[3:33].copy()  # 30 lignes d'agences
            
            # Headers colonnes (ligne 3, index 2)
            headers = []
            for i in range(1, 31):
                val = df_couts.iloc[2, i]
                if pd.notna(val):
                    headers.append(str(val).strip())
                else:
                    headers.append(f"Col{i}")
            
            df_couts_clean.columns = ['Agence'] + headers
            
            # Nettoyer index (colonne A)
            index_agences = []
            for i in range(len(df_couts_clean)):
                val = df_couts_clean.iloc[i, 0]
                if pd.notna(val):
                    index_agences.append(str(val).strip())
                else:
                    index_agences.append(f"Row{i}")
            
            df_couts_clean['Agence'] = index_agences
            df_couts_clean = df_couts_clean.set_index('Agence')
            # ✅ Supprimer doublons silencieusement (si présents)
            if not df_couts_clean.index.is_unique:
                df_couts_clean = df_couts_clean[~df_couts_clean.index.duplicated(keep='first')]
            
            if not df_couts_clean.columns.is_unique:
                df_couts_clean = df_couts_clean.loc[:, ~df_couts_clean.columns.duplicated(keep='first')]
                # Convertir en numérique
            df_couts_numeric = df_couts_clean.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # ✅ AFFICHAGE SANS STYLE (plus sûr)
            st.dataframe(
                df_couts_numeric.style.format("{:.0f}").set_properties(**{
                    'text-align': 'center',
                    'font-size': '10px'
                }),
                use_container_width=True,
                height=600
            )
            
            st.markdown("---")
            
            # Statistiques
            couts_non_nuls = df_couts_numeric.values[df_couts_numeric.values > 0]
            
            if len(couts_non_nuls) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Coût Minimum", f"{couts_non_nuls.min():.0f} FCFA")
                
                with col2:
                    st.metric("Coût Maximum", f"{couts_non_nuls.max():.0f} FCFA")
                
                with col3:
                    st.metric("Coût Moyen", f"{couts_non_nuls.mean():.2f} FCFA")
                
                with col4:
                    st.metric("Coût Médian", f"{np.median(couts_non_nuls):.0f} FCFA")
        else:
            st.error("Matrice des coûts introuvable ou trop courte")





    # TAB 4: Synthese
    with tabs[3]:
        st.markdown(f"**Synthese Optimisation - Jour J+{jour_selectionne}**")
        
        # ✅ RECALCULER SI VARIABLES MANQUANTES
        if 'matrice_values' not in locals() or 'agences_besoin' not in locals() or 'agences_excedent' not in locals():
            # Recharger depuis df_optim
            df_optim = excel_data['Optimisation transport']
            
            besoin_dict = {}
            excedent_dict = {}
            
            for idx in range(2, 31):
                if idx >= len(df_optim):
                    break
                
                # Besoin
                code_a = df_optim.iloc[idx, 0]
                montant_b = df_optim.iloc[idx, 1]
                
                if pd.notna(code_a) and pd.notna(montant_b):
                    code_clean = str(code_a).strip()
                    if code_clean and code_clean != 'nan':
                        try:
                            montant = float(montant_b)
                            if montant > 0:
                                besoin_dict[code_clean] = montant
                        except:
                            pass
                
                # Excédent
                code_d = df_optim.iloc[idx, 3]
                montant_e = df_optim.iloc[idx, 4]
                
                if pd.notna(code_d) and pd.notna(montant_e):
                    code_clean = str(code_d).strip()
                    if code_clean and code_clean != 'nan':
                        try:
                            montant = float(montant_e)
                            if montant > 0:
                                excedent_dict[code_clean] = montant
                        except:
                            pass
            
            # Trouver matrice
            matrice_start = None
            for idx in range(len(df_optim)):
                val = str(df_optim.iloc[idx, 0]).upper()
                if 'MATRICE' in val and 'ALLOCATION' in val:
                    matrice_start = idx + 2
                    break
            
            if matrice_start and matrice_start < len(df_optim):
                headers_row = df_optim.iloc[matrice_start]
                agences_besoin = []
                
                for col_idx in range(1, len(headers_row)):
                    val = headers_row.iloc[col_idx]
                    if pd.notna(val) and str(val).strip():
                        code = str(val).strip()
                        if 'TOTAL' not in code.upper():
                            agences_besoin.append(code)
                        else:
                            break
                
                agences_excedent = []
                
                for row_idx in range(matrice_start + 1, len(df_optim)):
                    code_ligne = df_optim.iloc[row_idx, 0]
                    
                    if pd.isna(code_ligne) or not str(code_ligne).strip():
                        continue
                    
                    code_ligne_str = str(code_ligne).strip()
                    
                    if 'TOTAL' in code_ligne_str.upper():
                        break
                    
                    agences_excedent.append(code_ligne_str)
                
                # Extraire valeurs matrice
                matrice_values = []
                for i, agence_exc in enumerate(agences_excedent):
                    row_vals = []
                    for j, agence_bes in enumerate(agences_besoin):
                        val = df_optim.iloc[matrice_start + 1 + i, j + 1]
                        try:
                            montant = float(val)
                            row_vals.append(montant if montant > 0 else 0.0)
                        except:
                            row_vals.append(0.0)
                    matrice_values.append(row_vals)
        
        # ✅ CALCULER COÛT TOTAL
        if 'matrice_values' in locals() and matrice_values and 'agences_besoin' in locals() and 'agences_excedent' in locals():
            cout_total = 0
            transactions_details = []
            
            # Charger matrice coûts
            df_couts = excel_data['Matrice couts unitaires']
            
            if len(df_couts) > 3:
                # Nettoyer matrice coûts
                df_couts_clean = df_couts.iloc[3:33].copy()
                
                # Headers
                headers = []
                for i in range(1, 31):
                    val = df_couts.iloc[2, i]
                    if pd.notna(val):
                        headers.append(str(val).strip())
                    else:
                        headers.append(f"Col{i}")
                
                df_couts_clean.columns = ['Agence'] + headers
                
                # Index
                index_agences = []
                for i in range(len(df_couts_clean)):
                    val = df_couts_clean.iloc[i, 0]
                    if pd.notna(val):
                        index_agences.append(str(val).strip())
                    else:
                        index_agences.append(f"Row{i}")
                
                df_couts_clean['Agence'] = index_agences
                df_couts_clean = df_couts_clean.set_index('Agence')
                
                # Supprimer doublons
                if not df_couts_clean.index.is_unique:
                    df_couts_clean = df_couts_clean[~df_couts_clean.index.duplicated(keep='first')]
                
                if not df_couts_clean.columns.is_unique:
                    df_couts_clean = df_couts_clean.loc[:, ~df_couts_clean.columns.duplicated(keep='first')]
                
                df_couts_numeric = df_couts_clean.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Calculer coût total
                for i, agence_exc in enumerate(agences_excedent):
                    for j, agence_bes in enumerate(agences_besoin):
                        montant = matrice_values[i][j]
                        
                        if montant > 0:
                            # Matching flexible
                            cout_unitaire = 0
                            
                            try:
                                # Essai direct
                                if agence_exc in df_couts_numeric.index and agence_bes in df_couts_numeric.columns:
                                    cout_unitaire = df_couts_numeric.loc[agence_exc, agence_bes]
                                else:
                                    # Chercher en enlevant zéros initiaux
                                    agence_exc_stripped = agence_exc.lstrip('0')
                                    agence_bes_stripped = agence_bes.lstrip('0')
                                    
                                    for idx in df_couts_numeric.index:
                                        if idx.lstrip('0') == agence_exc_stripped:
                                            for col in df_couts_numeric.columns:
                                                if col.lstrip('0') == agence_bes_stripped:
                                                    cout_unitaire = df_couts_numeric.loc[idx, col]
                                                    break
                                            break
                                
                                if cout_unitaire > 0:
                                    cout_transaction = montant * cout_unitaire
                                    cout_total += cout_transaction
                                    
                                    transactions_details.append({
                                        'De': agence_exc,
                                        'Vers': agence_bes,
                                        'Montant (M)': montant,
                                        'Coût Unitaire': cout_unitaire,
                                        'Coût Total': cout_transaction
                                    })
                            except:
                                pass
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Coût Total Optimal", f"{cout_total:,.2f} FCFA")
                
                with col2:
                    st.metric("Nombre Transactions", len(transactions_details))
                
                with col3:
                    if len(transactions_details) > 0:
                        cout_moyen = cout_total / len(transactions_details)
                        st.metric("Coût Moyen/Transaction", f"{cout_moyen:,.2f} FCFA")
                
                st.markdown("---")
                
                # Top 10 transactions
                if transactions_details:
                    st.markdown("**Top 10 Transactions les plus coûteuses**")
                    df_trans = pd.DataFrame(transactions_details)
                    df_trans = df_trans.sort_values('Coût Total', ascending=False).head(10)
                    
                    st.dataframe(
                        df_trans.style.format({
                            'Montant (M)': '{:,.2f}',
                            'Coût Unitaire': '{:,.0f}',
                            'Coût Total': '{:,.2f}'
                        }),
                        use_container_width=True
                    )
            else:
                st.warning("Matrice des coûts introuvable")
        else:
            st.warning("Données d'allocation manquantes pour calculer le coût total")
        
        st.markdown("---")
        
        # ✅ FLUX BEAC
        df_entree_full = excel_data['Donnees entree']
        flux_beac_row = df_entree_full[df_entree_full['Code Agence'] == 'BEAC']
        
        if not flux_beac_row.empty:
            flux_beac_val = flux_beac_row['Flux transport optimal'].values[0]
            
            if pd.notna(flux_beac_val) and abs(flux_beac_val) > 0.01:
                if flux_beac_val > 0:
                    role = "Versement BEAC"
                    color = '#FF6B6B'
                else:
                    role = "Retrait BEAC"
                    color = '#4ECDC4'
                    flux_beac_val = abs(flux_beac_val)
                
                st.markdown(f"""
                <div class="info-box">
                <b>Rôle BEAC :</b> {role}<br>
                <b>Montant :</b> {flux_beac_val:,.2f} Millions FCFA
                </div>
                """, unsafe_allow_html=True)
                
                fig_beac = go.Figure()
                
                fig_beac.add_trace(go.Bar(
                    x=['BEAC'],
                    y=[flux_beac_val],
                    marker_color=color,
                    text=[f"{flux_beac_val:,.1f}M"],
                    textposition='outside',
                    name=role
                ))
                
                fig_beac.update_layout(
                    title="Intervention BEAC",
                    yaxis_title="Montant (Millions FCFA)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_beac, use_container_width=True)
            else:
                st.info("Aucune intervention BEAC nécessaire pour ce jour")
        else:
            st.warning("Données BEAC introuvables")
        
        st.markdown("---")
        
        # Bouton téléchargement
        filepath_excel = Path(CONFIG['base_dir_optim']) / f"Optimisation_Transport_J{jour_selectionne}.xlsx"
        
        if filepath_excel.exists():
            with open(filepath_excel, 'rb') as f:
                excel_bytes = f.read()
            
            st.download_button(
                label=f"📥 Télécharger Fichier Excel J+{jour_selectionne}",
                data=excel_bytes,
                file_name=f"Optimisation_Transport_J{jour_selectionne}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    




# ============================================================================
# PAGE 5 : ANALYSE CONSOLIDEE
# ============================================================================

def page_analyse_consolidee():
    """Page d'analyse consolidee sur les 5 jours"""
    
    st.markdown('<div class="main-header">ANALYSE CONSOLIDEE (5 JOURS)</div>', 
                unsafe_allow_html=True)
    
    # Charger donnees pour tous les jours
    data_all_days = []
    
    for jour in CONFIG['jours']:
        excel_data = load_excel_optimisation(jour)
        
        if excel_data:
            df_entree = excel_data['Donnees entree']
            df_optim = excel_data['Optimisation transport']
            
            # Extraire infos
            nb_besoin = len(df_entree[df_entree['Type de flux'] == 'Besoin'])
            nb_excedent = len(df_entree[df_entree['Type de flux'] == 'Excedent'])
            total_besoin = df_entree[df_entree['Flux transport optimal'] > 0]['Flux transport optimal'].sum()
            flux_beac = df_entree[df_entree['Code Agence'] == 'BEAC']['Flux transport optimal'].values[0]
            
            # Role BEAC
            if flux_beac > 0:
                role_beac = "Versement"
            elif flux_beac < 0:
                role_beac = "Retrait"
            else:
                role_beac = "Aucun"
            
            data_all_days.append({
                'Jour': f"J+{jour}",
                'Nb_Besoin': nb_besoin,
                'Nb_Excedent': nb_excedent,
                'Total_Besoin': total_besoin,
                'Flux_BEAC': abs(flux_beac),
                'Role_BEAC': role_beac
            })
    
    if not data_all_days:
        st.error("Aucune donnee disponible pour l'analyse consolidee")
        return
    
    df_consolide = pd.DataFrame(data_all_days)
    
    st.markdown("---")
    
    # Graphiques
    tabs = st.tabs(["Evolution Flux", "Repartition Agences", "Role BEAC", "Statistiques Globales"])
    
    # TAB 1: Evolution flux
    with tabs[0]:
        st.markdown("**Evolution des Flux de Liquidite**")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Total Besoin par Jour", "Nombre d'Agences par Categorie"),
            vertical_spacing=0.15
        )
        
        # Graphique 1: Total besoin
        fig.add_trace(
            go.Scatter(
                x=df_consolide['Jour'],
                y=df_consolide['Total_Besoin'],
                mode='lines+markers',
                name='Total Besoin',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Graphique 2: Nb agences
        fig.add_trace(
            go.Bar(
                x=df_consolide['Jour'],
                y=df_consolide['Nb_Besoin'],
                name='Agences en Besoin',
                marker_color='#FF6B6B'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df_consolide['Jour'],
                y=df_consolide['Nb_Excedent'],
                name='Agences en Excedent',
                marker_color='#4ECDC4'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Jour", row=2, col=1)
        fig.update_yaxes(title_text="Montant (Millions)", row=1, col=1)
        fig.update_yaxes(title_text="Nombre d'Agences", row=2, col=1)
        
        fig.update_layout(height=700, showlegend=True, barmode='group')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Repartition agences
    with tabs[1]:
        st.markdown("**Repartition Moyenne Besoin / Excedent**")
        
        nb_besoin_moyen = df_consolide['Nb_Besoin'].mean()
        nb_excedent_moyen = df_consolide['Nb_Excedent'].mean()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Agences en Besoin', 'Agences en Excedent'],
            values=[nb_besoin_moyen, nb_excedent_moyen],
            marker_colors=['#FF6B6B', '#4ECDC4'],
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title="Repartition Moyenne des Agences",
            height=500
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tableau detaille
        st.markdown("**Detail par Jour**")
        st.dataframe(df_consolide[['Jour', 'Nb_Besoin', 'Nb_Excedent', 'Total_Besoin']], 
                    use_container_width=True)
    
    # # TAB 3: Role BEAC
    # with tabs[2]:
    #     st.markdown("**Intervention BEAC sur 5 Jours**")
        
    #     fig_beac = go.Figure()
        
    #     colors_beac = ['#FF6B6B' if r == 'Versement' else '#4ECDC4' if r == 'Retrait' else '#95A5A6' 
    #                   for r in df_consolide['Role_BEAC']]
        
    #     fig_beac.add_trace(go.Bar(
    #         x=df_consolide['Jour'],
    #         y=df_consolide['Flux_BEAC'],
    #         marker_color=colors_beac,
    #         text=df_consolide['Flux_BEAC'].apply(lambda x: f"{x:,.0f}M"),
    #         textposition='outside',
    #         hovertemplate='<b>%{x}</b><br>Montant: %{y:,.2f}M<br>Role: %{customdata}<extra></extra>',
    #         customdata=df_consolide['Role_BEAC']
    #     ))
        
    #     fig_beac.update_layout(
    #         title="Montant d'Intervention BEAC",
    #         xaxis_title="Jour",
    #         yaxis_title="Montant (Millions FCFA)",
    #         height=500,
    #         showlegend=False
    #     )
        
    #     st.plotly_chart(fig_beac, use_container_width=True)
        
    #     # Statistiques BEAC
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         nb_versements = (df_consolide['Role_BEAC'] == 'Versement').sum()
    #         st.metric("Jours avec Versement BEAC", nb_versements)
        
    #     with col2:
    #         nb_retraits = (df_consolide['Role_BEAC'] == 'Retrait').sum()
    #         st.metric("Jours avec Retrait BEAC", nb_retraits)
        
    #     with col3:
    #         flux_beac_moyen = df_consolide['Flux_BEAC'].mean()
    #         st.metric("Flux BEAC Moyen", f"{flux_beac_moyen:,.2f} M")


    # TAB 3: Role BEAC
    with tabs[2]:
        st.markdown("**Intervention BEAC sur 5 Jours**")
        
        # ✅ DONNÉES CORRECTES
        flux_beac_data = []
        
        for jour in CONFIG['jours']:
            excel_data = load_excel_optimisation(jour)
            
            if excel_data:
                df_entree = excel_data['Donnees entree']
                flux_beac_row = df_entree[df_entree['Code Agence'] == 'BEAC']
                
                if not flux_beac_row.empty:
                    flux_val = flux_beac_row['Flux transport optimal'].values[0]
                    
                    if pd.notna(flux_val) and abs(flux_val) > 0.01:
                        if flux_val > 0:
                            role = "Versement"
                            color = '#FF6B6B'
                        else:
                            role = "Retrait"
                            color = '#4ECDC4'
                            flux_val = abs(flux_val)
                        
                        flux_beac_data.append({
                            'Jour': f"J+{jour}",
                            'Montant': flux_val,
                            'Role': role,
                            'Color': color
                        })
        
        if flux_beac_data:
            df_beac = pd.DataFrame(flux_beac_data)
            
            fig_beac = go.Figure()
            
            fig_beac.add_trace(go.Bar(
                x=df_beac['Jour'],
                y=df_beac['Montant'],
                marker_color=df_beac['Color'],
                text=df_beac['Montant'].apply(lambda x: f"{x:,.0f}M"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Montant: %{y:,.2f}M<br>Rôle: %{customdata}<extra></extra>',
                customdata=df_beac['Role']
            ))
            
            fig_beac.update_layout(
                title="Montant d'Intervention BEAC",
                xaxis_title="Jour",
                yaxis_title="Montant (Millions FCFA)",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_beac, use_container_width=True)
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            
            nb_versements = (df_beac['Role'] == 'Versement').sum()
            nb_retraits = (df_beac['Role'] == 'Retrait').sum()
            flux_moyen = df_beac['Montant'].mean()
            
            with col1:
                st.metric("Jours avec Versement BEAC", nb_versements)
            
            with col2:
                st.metric("Jours avec Retrait BEAC", nb_retraits)
            
            with col3:
                st.metric("Flux BEAC Moyen", f"{flux_moyen:,.2f} M")
        else:
            st.success(" Aucune intervention BEAC nécessaire sur la période")


    
    # TAB 4: Statistiques globales
    with tabs[3]:
        st.markdown("**Statistiques Globales (5 Jours)**")
        
        # Metriques principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_besoin_5j = df_consolide['Total_Besoin'].sum()
            st.metric("Total Besoin (5 jours)", f"{total_besoin_5j:,.2f} M")
        
        with col2:
            besoin_moyen = df_consolide['Total_Besoin'].mean()
            st.metric("Besoin Moyen Quotidien", f"{besoin_moyen:,.2f} M")
        
        with col3:
            flux_beac_total = df_consolide['Flux_BEAC'].sum()
            st.metric("Intervention BEAC Totale", f"{flux_beac_total:,.2f} M")
        
        st.markdown("---")
        
        # Top 5 agences
        st.markdown("**Top 5 Agences Contributrices (Excedent Cumule)**")
        
        # Charger toutes les donnees pour calculer
        all_flux_by_agence = {}
        
        for jour in CONFIG['jours']:
            excel_data = load_excel_optimisation(jour)
            if excel_data:
                df_entree = excel_data['Donnees entree']
                
                for _, row in df_entree.iterrows():
                    agence = row['Code Agence']
                    flux = row['Flux transport optimal']
                    
                    if agence != 'BEAC':
                        if agence not in all_flux_by_agence:
                            all_flux_by_agence[agence] = {'besoin': 0, 'excedent': 0}
                        
                        if flux > 0:
                            all_flux_by_agence[agence]['besoin'] += flux
                        elif flux < 0:
                            all_flux_by_agence[agence]['excedent'] += abs(flux)
        
        # Top 5 excedent
        df_agences = pd.DataFrame([
            {'Agence': k, 'Excedent': v['excedent'], 'Besoin': v['besoin']}
            for k, v in all_flux_by_agence.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_top_excedent = df_agences.sort_values('Excedent', ascending=False).head(5)
            
            fig_top_exc = go.Figure()
            fig_top_exc.add_trace(go.Bar(
                y=df_top_excedent['Agence'],
                x=df_top_excedent['Excedent'],
                orientation='h',
                marker_color='#4ECDC4',
                text=df_top_excedent['Excedent'].apply(lambda x: f"{x:,.0f}M"),
                textposition='outside'
            ))
            
            fig_top_exc.update_layout(
                title="Top 5 Agences en Excedent",
                xaxis_title="Excedent Cumule (Millions)",
                height=400
            )
            
            st.plotly_chart(fig_top_exc, use_container_width=True)
        
        with col2:
            df_top_besoin = df_agences.sort_values('Besoin', ascending=False).head(5)
            
            fig_top_bes = go.Figure()
            fig_top_bes.add_trace(go.Bar(
                y=df_top_besoin['Agence'],
                x=df_top_besoin['Besoin'],
                orientation='h',
                marker_color='#FF6B6B',
                text=df_top_besoin['Besoin'].apply(lambda x: f"{x:,.0f}M"),
                textposition='outside'
            ))
            
            fig_top_bes.update_layout(
                title="Top 5 Agences en Besoin",
                xaxis_title="Besoin Cumule (Millions)",
                height=400
            )
            
            st.plotly_chart(fig_top_bes, use_container_width=True)


# ============================================================================
# PAGE 6 : DOCUMENTATION & EXPORT
# ============================================================================

def page_documentation():
    """Page de documentation et exports"""
    
    st.markdown('<div class="main-header">DOCUMENTATION & EXPORTS</div>', 
                unsafe_allow_html=True)
    
    # Sections
    tabs = st.tabs(["Methodologie", "Guide Utilisateur", "Exports", "A Propos"])
    
    # TAB 1: Methodologie
    with tabs[0]:
        st.markdown("**Methodologie Complete du Projet**")
        
        st.markdown("""
        ### 1. Phase Forecasting
        
        **Objectif :** Predire les flux de liquidite (encaissements, decaissements, besoin net)
        pour les 5 prochains jours ouvrables.
        
        **Modeles Evalues :**
        - XGBoost : Modele gradient boosting avec features temporelles
        - Prophet : Modele additif de Facebook pour series temporelles
        - ARIMA : Modele autoregressif integre
        - Moyenne Mobile : Baseline simple
        - Ensembles : Combinaisons ponderees des meilleurs modeles
        
        **Critere de Selection :** MAPE (Mean Absolute Percentage Error) minimal
        
        **Approches Comparees :**
        1. **Directe :** Prediction directe de la colonne Besoin
        2. **Reconstruction :** Besoin = Encaissements - Decaissements
        
        
        
        ### 2. Phase Stocks Minimum
        
        **Objectif :** Determiner le stock de liquidite minimum requis par agence pour
        assurer la continuite operationnelle.
        
        **Methodologie :** Analyse statistique des soldes historiques
        
        **Calcul :** Stock Minimum = |P5| (5eme percentile en valeur absolue)
        
        **Justification :**
        - Standard international Bale III (VaR 95%)
        - Protection dans 95% des scenarios
        - Equilibre entre securite et efficience
        
        **Formule :** Pour chaque agence i,
        ```
        Stock_Min[i] = |Percentile_5(Soldes_Historiques[i])|
        ```
        
        
        
        ### 3. Phase Optimisation
        
        **Objectif :** Minimiser les couts de transport tout en equilibrant les flux.
        
        **Formulation Mathematique :**
        
        Variables de decision :
        ```
        x[i,j] = Montant transfere de l'agence i vers l'agence j
        ```
        
        Fonction objectif :
        ```
        Minimiser Z = Somme(i) Somme(j) C[i,j] * x[i,j]
        ```
        
        Contraintes :
            1. Somme(j) x[i,j] = Excedent[i]  (Conservation excedents)
            2. Somme(i) x[i,j] = Besoin[j]    (Satisfaction besoins)
            3. x[i,j] >= 0                     (Non-negativite)
            4. Somme Excedents = Somme Besoins (Equilibre global

        
        **Methode de Resolution :** CBC (COIN-OR Branch and Cut) - Algorithme du Simplexe
        
        **Garanties :**
        - Optimalite globale mathematiquement prouvee
        - Resolution en temps polynomial
        - Solution unique et reproductible
        
        **Role de la BEAC :**
        La BEAC agit comme variable d'ajustement pour garantir l'equilibre :
        ```
        Flux_BEAC = -Somme(Flux_Agences)
        ```
        - Si Flux_BEAC > 0 : Versement BEAC (systeme en deficit)
        - Si Flux_BEAC < 0 : Retrait BEAC (systeme en excedent)
        
        
        
        ### 4. Calcul des Flux Optimaux
        
        Pour chaque agence, le flux de transport optimal est calcule comme suit :
        ```
        Flux_Optimal = Besoin_Predit + Stock_Securite - Solde_Veille
        ```
        
        Avec :
        - **Besoin_Predit :** Prevision du modele ML selectionne
        - **Stock_Securite :** ENT((1 + MAPE/100) * Stock_Minimum) + 1
        - **Solde_Veille :** Solde en caisse de la veille
        
        **Classification :**
        - Flux > 0 : Agence en Besoin (doit recevoir des fonds)
        - Flux < 0 : Agence en Excedent (peut transferer des fonds)
        - Flux = 0 : Aucun besoin
        """)
        
        # # Bouton telechargement rapport PDF
        # st.markdown("---")
        # st.markdown("**Telecharger la Documentation Complete**")
        
        # rapport_path = Path("Rapport_Methodologie_Stocks_Minimum.docx")
        
        # if rapport_path.exists():
        #     with open(rapport_path, 'rb') as f:
        #         doc_bytes = f.read()
            
        #     st.download_button(
        #         label="Telecharger Rapport Methodologique (Word)",
        #         data=doc_bytes,
        #         file_name="Rapport_Methodologie_Stocks_Minimum.docx",
        #         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        #     )
        # else:
        #     st.info("Rapport methodologique non disponible. Generer d'abord le rapport avec le script Python.")
    
    # TAB 2: Guide utilisateur
    with tabs[1]:
        st.markdown("**Guide d'Utilisation du Dashboard**")
        
        st.markdown("""
        ### Navigation
        
        Le dashboard est organise en 6 pages accessibles via le menu lateral :
        
        1. **Accueil :** Vue d'ensemble et architecture du projet
        2. **Forecasting :** Visualisation des predictions par agence
        3. **Stocks Minimum :** Analyse statistique et determination des reserves
        4. **Optimisation :** Allocation optimale des flux par jour
        5. **Analyse Consolidee :** Synthese sur 5 jours
        6. **Documentation :** Methodologie et exports
        
        
        
        ### Page Forecasting
        
        **Fonctionnalites :**
        - Selection d'une agence dans le menu deroulant
        - 4 onglets pour visualiser les composantes :
          - Encaissements
          - Decaissements
          - Besoin
          - Comparaison modeles
        - Graphiques interactifs avec predictions vs valeurs reelles
        - Tableaux de metriques MAPE par modele
        
        **Interpretation :**
        - Lignes noires : Valeurs reelles (septembre 2025)
        - Lignes colorees : Predictions des differents modeles
        - MAPE faible (< 50%) : Bonne precision phase test
        - MAPE eleve (> 100%) : Precision faible, modele a ameliorer
        
        
        
        ### Page Stocks Minimum
        
        **Fonctionnalites :**
        - Selection d'une agence
        - 4 onglets d'analyse :
          - Distribution : Repartition des percentiles
          - Boxplot : Visualisation quartiles
          - Percentiles : Tableau complet P1-P99
          - Comparaison : Classement toutes agences
        
        **Interpretation :**
        - P5 (ligne rouge foncee) = Stock minimum recommande
        - Q1-Q4 : Zones de repartition par quartile
        - Valeurs negatives = Deficit structurel (normal en banque)
        - Stock minimum eleve = Agence avec forte variabilite
        
        
        
        ### Page Optimisation
        
        **Fonctionnalites :**
        - Selection du jour (J+1 a J+5)
        - 4 onglets :
          - Donnees Entree : Tableau complet des flux calcules
          - Repartition & Matrice : Visualisation allocation optimale
          - Couts Unitaires : Matrice des couts de transport
          - Synthese : Metriques globales et role BEAC
        - Telechargement fichier Excel complet
        
        **Interpretation :**
        - Heatmap allocation : Intensite = montant transfere
        - Cellules vides : Pas de transfert entre ces agences
        - Role BEAC : Equilibrage du systeme
        
        
        
        ### Page Analyse Consolidee
        
        **Fonctionnalites :**
        - Evolution des flux sur 5 jours
        - Repartition moyenne besoin/excedent
        - Historique interventions BEAC
        - Top 5 agences contributrices/beneficiaires
        
        **Interpretation :**
        - Tendances : Identifier patterns hebdomadaires
        - BEAC : Evaluer dependance au financement externe
        - Top agences : Identifier acteurs cles du reseau
        
        
        
        ### Interactivite
        
        **Graphiques Plotly :**
        - Survol : Affichage valeurs detaillees
        - Zoom : Clic + glisser sur zone d'interet
        - Pan : Deplacement avec outil main
        - Reset : Double-clic pour reinitialiser
        - Export : Bouton appareil photo (PNG)
        
        **Tableaux :**
        - Tri : Clic sur entetes de colonnes
        - Recherche : Barre de recherche integree
        - Export : Copier-coller dans Excel
        """)
    
    # TAB 3: Exports
    with tabs[2]:
        st.markdown("**Exports de Donnees**")
        
        st.markdown("""
        <div class="info-box">
        <b>Note :</b> Tous les exports sont generes a partir des fichiers sources.
        Assurez-vous que les repertoires resultats_metriques/ et optimisation_transport__/
        contiennent les fichiers a jour.
        </div>
        """, unsafe_allow_html=True)
        
        # st.markdown("---")
        
        # Section 1: Fichiers Excel Optimisation
        st.markdown("**Fichiers Excel d'Optimisation (par jour)**")
        
        cols = st.columns(5)
        
        for idx, jour in enumerate(CONFIG['jours']):
            with cols[idx]:
                filepath = Path(CONFIG['base_dir_optim']) / f"Optimisation_Transport_J{jour}.xlsx"
                
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        excel_bytes = f.read()
                    
                    st.download_button(
                        label=f"J+{jour}",
                        data=excel_bytes,
                        file_name=f"Optimisation_J{jour}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.warning(f"J+{jour} indisponible")
        
        # st.markdown("---")
        
        # Section 2: Fichier Stocks Minimum
        st.markdown("**Fichier Stocks Minimum (CSV)**")
        
        stocks_path = Path(CONFIG['stocks_file'])
        
        if stocks_path.exists():
            with open(stocks_path, 'rb') as f:
                csv_bytes = f.read()
            
            st.download_button(
                label="Telecharger stocks_minimum_agences.csv",
                data=csv_bytes,
                file_name="stocks_minimum_agences.csv",
                mime="text/csv"
            )
        else:
            st.warning("Fichier stocks_minimum_agences.csv introuvable")
        
        # st.markdown("---")
        
        # # Section 3: Rapport Word
        # st.markdown("**Rapport Consolide (Word)**")
        
        # rapport_path = Path(CONFIG['base_dir_optim']) / "Rapport_Validation_Consolide.docx"
        
        # if rapport_path.exists():
        #     with open(rapport_path, 'rb') as f:
        #         word_bytes = f.read()
            
        #     st.download_button(
        #         label="Telecharger Rapport Consolide",
        #         data=word_bytes,
        #         file_name="Rapport_Validation_Consolide.docx",
        #         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        #     )
        # else:
        #     st.info("Rapport Word non encore genere. Executer le pipeline d'optimisation complet.")
        
        # st.markdown("---")
        
        # Section 4: Export donnees predictions
        st.markdown("**Export Donnees Predictions (par agence)**")
        
        agence_export = st.selectbox(
            "Selectionner une agence pour export",
            CONFIG['agences'],
            key='export_agence'
        )
        
        if st.button("Generer Export CSV Predictions"):
            # Charger toutes les predictions pour l'agence
            export_data = []
            
            for composante in CONFIG['composantes']:
                df_pred = load_predictions(agence_export, composante)
                
                if df_pred is not None:
                    df_pred['Composante'] = composante
                    df_pred['Code_Agence'] = agence_export
                    export_data.append(df_pred)
            
            if export_data:
                df_export = pd.concat(export_data, ignore_index=True)
                
                csv_export = df_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label=f"Telecharger predictions_{agence_export}.csv",
                    data=csv_export,
                    file_name=f"predictions_{agence_export}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"Aucune donnee disponible pour l'agence {agence_export}")
    
    # TAB 4: A propos
    with tabs[3]:
        st.markdown("**A Propos du Projet**")
        
        st.markdown("""
        ### Systeme d'Optimisation des Flux de Liquidité Bancaire
        
        **Version :** 1.2  
        **Date :** Novembre 2025  
        
        
        
        Ce systeme vise a :
        1. Predire les besoins de liquidite des agences bancaires
        2. Determiner les stocks de securite optimaux
        3. Minimiser les couts de transport inter-agences
        4. Garantir l'equilibre global du reseau
        
        
        
        ### Technologies Utilisees
        
        **Machine Learning :**
        - XGBoost 1.7+
        - Prophet (Facebook)
        - Statsmodels (ARIMA)
        - Scikit-learn
        
        **Optimisation :**
        - PuLP (Python LP Modeler)
        - CBC (COIN-OR Branch and Cut)
        
        **Visualisation :**
        - Streamlit
        - Plotly
        - Pandas
        
        **Export :**
        - OpenPyXL (Excel)
        - Python-docx (Word)
        
        
        
        ### Performances
        
        **Forecasting :**
        - Horizon : 5 jours
        - Frequence : Quotidienne
        - MAPE moyen : Variable selon agence et composante
        
        **Optimisation :**
        - Methode : CBC Simplexe
        - Temps de resolution : < 1 seconde pour 30 agences
        - Garantie : Optimalite globale
        
        
        
        ### Contact & Support
        
        Pour toute question ou demande d'amelioration, contactez l'equipe.
        
        
        
        ### Licence
        
        Usage interne uniquement. Tous droits reserves.
        """)
        
        st.markdown("---")
        
        # Informations systeme
        st.markdown("**Informations Systeme**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Configuration :**
            - Nombre d'agences : {len(CONFIG['agences'])}
            - Horizon prevision : {CONFIG['horizon']} jours
            - Repertoire metriques : `{CONFIG['base_dir_metrics']}`
            - Repertoire optimisation : `{CONFIG['base_dir_optim']}`
            """)
        
        with col2:
            st.markdown(f"""
            **Fichiers Disponibles :**
            - Predictions : {len(list(Path(CONFIG['base_dir_metrics']).glob('*/*.csv')))} fichiers
            - Optimisation : {len(list(Path(CONFIG['base_dir_optim']).glob('*.xlsx')))} fichiers
            - Stocks minimum : {'Oui' if Path(CONFIG['stocks_file']).exists() else 'Non'}
            """)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale du dashboard"""
    
    # Sidebar navigation
    st.sidebar.title("NAVIGATION")
    
    page = st.sidebar.radio(
        "Selectionner une page",
        [
            "Accueil",
            "Forecasting",
            "Stocks Minimum",
            "Optimisation Transport",
            "Analyse Consolidee",
            "Documentation & Exports"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Informations sidebar
    st.sidebar.markdown("**Informations**")
    st.sidebar.info(f"""
    **Agences :** {len(CONFIG['agences'])}  
    **Horizon :** {CONFIG['horizon']} jours  
    
    """)
    # **Date :** {datetime.now().strftime('%d/%m/%Y')}

    st.sidebar.markdown("---")
    
    # Credits
    st.sidebar.markdown("""
    **Systeme d'Optimisation**  
    Version 1.2  
    Novembre 2025
    """)
    
    # Routing vers les pages
    if page == "Accueil":
        page_accueil()
    elif page == "Forecasting":
        page_forecasting()
    elif page == "Stocks Minimum":
        page_stocks_minimum()
    elif page == "Optimisation Transport":
        page_optimisation()
    elif page == "Analyse Consolidee":
        page_analyse_consolidee()
    elif page == "Documentation & Exports":
        page_documentation()

# ============================================================================
# POINT D'ENTREE
# ============================================================================

if __name__ == "__main__":
    main()
