import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Prévisions Flux de Liquidité",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2c5aa0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2c5aa0;
        margin-bottom: 1rem;
    }
    .interpretation-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin-top: 1rem;
    }
    .stTab {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_data
def load_all_agencies():
    """Charger la liste des agences disponibles"""
    base_dir = Path('resultats_metriques')
    if not base_dir.exists():
        return []
    
    agencies = [d.name for d in base_dir.iterdir() if d.is_dir()]
    return sorted(agencies)


@st.cache_data
def load_metrics(code_agence, phase):
    """Charger les métriques pour une agence et une phase"""
    filepath = Path('resultats_metriques') / code_agence / f'{code_agence}_metriques_{phase}.csv'
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    return df


@st.cache_data
def load_predictions(code_agence, component):
    """Charger les prédictions pour une agence et une composante"""
    filepath = Path('resultats_metriques') / code_agence / f'{code_agence}_predictions_{component}.csv'
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    return df


def get_metric_color(mape):
    """Obtenir la couleur selon le MAPE"""
    if mape < 15:
        return "#28a745"  # Vert
    elif mape < 30:
        return "#ffc107"  # Jaune
    elif mape < 50:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Rouge


def get_performance_label(mape):
    """Obtenir le label de performance"""
    if mape < 15:
        return "EXCELLENT"
    elif mape < 30:
        return "BON"
    elif mape < 50:
        return "MOYEN"
    else:
        return "INSUFFISANT"


def interpret_metrics(df_metrics, phase):
    """Générer une interprétation des métriques"""
    
    interpretations = []
    
    # Trouver le meilleur modèle
    best_model = df_metrics.loc[df_metrics['MAPE'].idxmin()]
    
    interpretations.append(f"**Meilleur modèle** : {best_model['Modele'].upper()}")
    interpretations.append(f"**Performance** : {get_performance_label(best_model['MAPE'])} (MAPE = {best_model['MAPE']:.2f}%)")
    
    # Analyse par métrique
    if best_model['MAPE'] < 15:
        interpretations.append(
            f"Le modèle {best_model['Modele']} présente une **excellente précision** "
            f"avec une erreur moyenne de {best_model['MAPE']:.2f}%. Ce niveau de performance "
            f"permet un déploiement en production avec confiance."
        )
    elif best_model['MAPE'] < 30:
        interpretations.append(
            f"Le modèle {best_model['Modele']} affiche une **bonne précision** "
            f"avec une erreur moyenne de {best_model['MAPE']:.2f}%. Un monitoring quotidien "
            f"est recommandé pour maintenir cette performance."
        )
    elif best_model['MAPE'] < 50:
        interpretations.append(
            f"Le modèle {best_model['Modele']} présente une **précision moyenne** "
            f"avec une erreur de {best_model['MAPE']:.2f}%. Une surveillance renforcée "
            f"et un enrichissement des données sont recommandés."
        )
    else:
        interpretations.append(
            f"Le modèle {best_model['Modele']} montre une **précision insuffisante** "
            f"avec une erreur de {best_model['MAPE']:.2f}%. Des données complémentaires "
            f"ou une révision de la stratégie de modélisation sont nécessaires."
        )
    
    # Analyse R²
    if not pd.isna(best_model['R2']):
        if best_model['R2'] > 0.8:
            interpretations.append(
                f"Le R² de {best_model['R2']:.3f} indique que le modèle explique "
                f"**{best_model['R2']*100:.1f}% de la variance** des données, "
                f"ce qui est excellent."
            )
        elif best_model['R2'] > 0.6:
            interpretations.append(
                f"Le R² de {best_model['R2']:.3f} montre que le modèle capture "
                f"**{best_model['R2']*100:.1f}% de la variance**, ce qui est satisfaisant."
            )
        else:
            interpretations.append(
                f"Le R² de {best_model['R2']:.3f} suggère que le modèle explique "
                f"seulement **{best_model['R2']*100:.1f}% de la variance**, "
                f"indiquant une capacité prédictive limitée."
            )
    
    # Comparaison avec d'autres modèles
    mape_values = df_metrics['MAPE'].values
    if len(mape_values) > 1:
        ecart_type = np.std(mape_values)
        if ecart_type < 5:
            interpretations.append(
                "Les performances des différents modèles sont **homogènes** "
                "(écart-type < 5%), suggérant une stabilité des prévisions."
            )
        elif ecart_type < 15:
            interpretations.append(
                "Les performances varient **modérément** entre les modèles "
                f"(écart-type = {ecart_type:.1f}%)."
            )
        else:
            interpretations.append(
                "Les performances varient **significativement** entre les modèles "
                f"(écart-type = {ecart_type:.1f}%), indiquant des approches "
                "de modélisation très différentes."
            )
    
    return interpretations


def create_comparison_chart(df_predictions, component, models_to_show=None):
    """Créer un graphique de comparaison interactif"""
    
    if models_to_show is None:
        models_to_show = [col for col in df_predictions.columns 
                         if col not in ['Code_Agence', 'Composante', 'Jour', 'Reel']]
    
    fig = go.Figure()
    
    # Valeurs réelles
    fig.add_trace(go.Scatter(
        x=df_predictions['Jour'],
        y=df_predictions['Reel'] / 1e6,
        mode='lines+markers',
        name='Réel',
        line=dict(color='black', width=3),
        marker=dict(size=10, symbol='circle')
    ))
    
    # Modèles
    colors = {
        'xgboost': '#0066cc',
        'prophet': '#00cc66',
        'arima': '#ff9933',
        'moving_avg': '#9933ff',
        'ensemble_post': '#cc0000',
        'ensemble_cv_best': '#990000',
        'ensemble_cv_second': '#660000'
    }
    
    for model in models_to_show:
        if model in df_predictions.columns:
            fig.add_trace(go.Scatter(
                x=df_predictions['Jour'],
                y=df_predictions[model] / 1e6,
                mode='lines+markers',
                name=model.upper(),
                line=dict(color=colors.get(model, '#666666'), width=2),
                marker=dict(size=7)
            ))
    
    fig.update_layout(
        title=f'Prévisions vs Réel - {component.capitalize()}',
        xaxis_title='Jour',
        yaxis_title='Montant (Millions FCFA)',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        height=600,
        template='plotly_white'
    )
    
    return fig


def create_error_chart(df_predictions, model_name):
    """Créer un graphique des erreurs jour par jour"""
    
    errors = []
    for _, row in df_predictions.iterrows():
        if model_name in row and not pd.isna(row[model_name]):
            error = abs((row['Reel'] - row[model_name]) / row['Reel']) * 100
            errors.append(error)
        else:
            errors.append(np.nan)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_predictions['Jour'],
        y=errors,
        name='Erreur (%)',
        marker_color=['#28a745' if e < 20 else '#ffc107' if e < 40 else '#dc3545' 
                      for e in errors]
    ))
    
    # Ligne de référence à 20%
    fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                  annotation_text="Seuil acceptable (20%)")
    
    fig.update_layout(
        title=f'Erreur par Jour - {model_name.upper()}',
        xaxis_title='Jour',
        yaxis_title='MAPE (%)',
        height=400,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    
    # En-tête
    st.markdown('<div class="main-header">Système de Prédiction du Flux de Liquidité</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Charger les agences
    agencies = load_all_agencies()
    
    if not agencies:
        st.error("Aucune agence trouvée dans le dossier 'resultats_metriques'.")
        st.info("Veuillez d'abord exécuter le pipeline de prévision pour générer les résultats.")
        return
    
    # Sélection agence
    selected_agency = st.sidebar.selectbox(
        "Sélectionner une agence",
        agencies,
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Sélection page
    page = st.sidebar.radio(
        "Section",
        ["Vue d'ensemble", "Métriques détaillées", "Visualisations", "Analyse comparative"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Agence sélectionnée** : {selected_agency}\n\n"
        f"**Total agences** : {len(agencies)}"
    )
    
    # ========================================================================
    # PAGE 1: VUE D'ENSEMBLE
    # ========================================================================
    
    if page == "Vue d'ensemble":
        
        st.markdown(f'<div class="section-header">Vue d\'ensemble - Agence {selected_agency}</div>', 
                    unsafe_allow_html=True)
        
        # Tabs pour les phases
        tab1, tab2, tab3 = st.tabs(["Entraînement", "Test", "Prédictions Futures"])
        
        # TAB 1: TRAIN
        with tab1:
            st.subheader("Phase d'Entraînement")
            st.write("Performances des modèles sur les données d'entraînement (jusqu'au 28 février 2025)")
            
            df_train = load_metrics(selected_agency, 'train')
            
            if df_train is not None:
                
                components = df_train['Colonne'].unique()
                
                for component in components:
                    st.markdown(f"**{component.capitalize()}**")
                    
                    df_comp = df_train[df_train['Colonne'] == component].copy()
                    df_comp = df_comp.sort_values('MAPE')
                    
                    # Afficher tableau
                    st.dataframe(
                        df_comp[['Modele', 'MAE', 'RMSE', 'MAPE', 'R2']].style.format({
                            'MAE': '{:,.0f}',
                            'RMSE': '{:,.0f}',
                            'MAPE': '{:.2f}%',
                            'R2': '{:.3f}'
                        }).background_gradient(subset=['MAPE'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
            
            else:
                st.warning(f"Données d'entraînement non trouvées pour l'agence {selected_agency}")
        
        # TAB 2: TEST
        with tab2:
            st.subheader("Phase de Test")
            st.write("Performances des modèles sur les données de test (1er mars - 30 juin 2025)")
            
            df_test = load_metrics(selected_agency, 'test')
            
            if df_test is not None:
                
                components = df_test['Colonne'].unique()
                
                for component in components:
                    st.markdown(f"**{component.capitalize()}**")
                    
                    df_comp = df_test[df_test['Colonne'] == component].copy()
                    df_comp = df_comp.sort_values('MAPE')
                    
                    st.dataframe(
                        df_comp[['Modele', 'MAE', 'RMSE', 'MAPE', 'R2']].style.format({
                            'MAE': '{:,.0f}',
                            'RMSE': '{:,.0f}',
                            'MAPE': '{:.2f}%',
                            'R2': '{:.3f}'
                        }).background_gradient(subset=['MAPE'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
            
            else:
                st.warning(f"Données de test non trouvées pour l'agence {selected_agency}")
        
        # TAB 3: FORECAST
        with tab3:
            st.subheader("Prédictions Futures (Validation)")
            st.write("Performances des modèles sur les prévisions futures (après 30 juin 2025)")
            
            df_forecast = load_metrics(selected_agency, 'forecast')
            
            if df_forecast is not None:
                
                components = df_forecast['Colonne'].unique()
                
                for component in components:
                    st.markdown(f"**{component.capitalize()}**")
                    
                    df_comp = df_forecast[df_forecast['Colonne'] == component].copy()
                    df_comp = df_comp.sort_values('MAPE')
                    
                    # Mise en évidence du meilleur
                    best_model = df_comp.iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Meilleur Modèle",
                            best_model['Modele'].upper(),
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "MAPE",
                            f"{best_model['MAPE']:.2f}%",
                            delta=None,
                            delta_color="inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "Performance",
                            get_performance_label(best_model['MAPE']),
                            delta=None
                        )
                    
                    st.dataframe(
                        df_comp[['Modele', 'MAE', 'RMSE', 'MAPE', 'R2']].style.format({
                            'MAE': '{:,.0f}',
                            'RMSE': '{:,.0f}',
                            'MAPE': '{:.2f}%',
                            'R2': '{:.3f}'
                        }).background_gradient(subset=['MAPE'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    # Interprétation
                    st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
                    st.markdown("**Interprétation**")
                    interpretations = interpret_metrics(df_comp, 'forecast')
                    for interp in interpretations:
                        st.markdown(f"- {interp}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            else:
                st.warning(f"Données de prévisions non trouvées pour l'agence {selected_agency}")
    
    # ========================================================================
    # PAGE 2: MÉTRIQUES DÉTAILLÉES
    # ========================================================================
    
    elif page == "Métriques détaillées":
        
        st.markdown(f'<div class="section-header">Métriques Détaillées - Agence {selected_agency}</div>', 
                    unsafe_allow_html=True)
        
        # Sélection phase et composante
        col1, col2 = st.columns(2)
        
        with col1:
            phase_selected = st.selectbox(
                "Phase",
                ["train", "test", "forecast"],
                format_func=lambda x: {
                    'train': 'Entraînement',
                    'test': 'Test',
                    'forecast': 'Prédictions Futures'
                }[x]
            )
        
        with col2:
            component_selected = st.selectbox(
                "Composante",
                ["encaissements", "decaissements", "Besoin"]
            )
        
        # Charger les données
        df_metrics = load_metrics(selected_agency, phase_selected)
        
        if df_metrics is not None:
            
            df_comp = df_metrics[df_metrics['Colonne'] == component_selected].copy()
            df_comp = df_comp.sort_values('MAPE')
            
            # Graphique en barres des MAPE
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_comp['Modele'],
                y=df_comp['MAPE'],
                marker_color=[get_metric_color(m) for m in df_comp['MAPE']],
                text=df_comp['MAPE'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'Comparaison MAPE - {component_selected.capitalize()}',
                xaxis_title='Modèle',
                yaxis_title='MAPE (%)',
                height=500,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.subheader("Tableau Détaillé")
            
            st.dataframe(
                df_comp[['Modele', 'MAE', 'RMSE', 'MAPE', 'R2']].style.format({
                    'MAE': '{:,.0f}',
                    'RMSE': '{:,.0f}',
                    'MAPE': '{:.2f}%',
                    'R2': '{:.3f}'
                }).background_gradient(subset=['MAPE', 'MAE', 'RMSE'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
            
            # Statistiques
            st.subheader("Statistiques")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAPE Minimum", f"{df_comp['MAPE'].min():.2f}%")
            
            with col2:
                st.metric("MAPE Maximum", f"{df_comp['MAPE'].max():.2f}%")
            
            with col3:
                st.metric("MAPE Moyen", f"{df_comp['MAPE'].mean():.2f}%")
            
            with col4:
                st.metric("Écart-type", f"{df_comp['MAPE'].std():.2f}%")
        
        else:
            st.warning(f"Données non trouvées")
    
    # ========================================================================
    # PAGE 3: VISUALISATIONS
    # ========================================================================
    
    elif page == "Visualisations":
        
        st.markdown(f'<div class="section-header">Visualisations - Agence {selected_agency}</div>', 
                    unsafe_allow_html=True)
        
        st.info("Cette section affiche les prévisions détaillées jour par jour comparées aux valeurs réelles.")
        
        # Sélection composante
        component_selected = st.selectbox(
            "Sélectionner une composante",
            ["encaissements", "decaissements", "Besoin"]
        )
        
        # Charger les prédictions
        df_predictions = load_predictions(selected_agency, component_selected)
        
        if df_predictions is not None:
            
            # Informations générales
            st.write(f"**Période de prévision** : {len(df_predictions)} jours ouvrables")
            st.write(f"**Plage** : Jour 1 à Jour {len(df_predictions)}")
            
            # Sélection des modèles à afficher
            available_models = [col for col in df_predictions.columns 
                               if col not in ['Code_Agence', 'Composante', 'Jour', 'Reel']]
            
            selected_models = st.multiselect(
                "Sélectionner les modèles à afficher",
                available_models,
                default=available_models[:5] if len(available_models) > 5 else available_models
            )
            
            if selected_models:
                # Graphique principal
                fig = create_comparison_chart(df_predictions, component_selected, selected_models)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des valeurs
                with st.expander("Voir les valeurs détaillées"):
                    display_cols = ['Jour', 'Reel'] + selected_models
                    st.dataframe(
                        df_predictions[display_cols].style.format({
                            col: '{:,.0f}' for col in display_cols if col != 'Jour'
                        }),
                        use_container_width=True,
                        height=400
                    )
                
                # Analyse des erreurs pour un modèle
                st.subheader("Analyse des Erreurs par Jour")
                
                model_for_error = st.selectbox(
                    "Sélectionner un modèle pour l'analyse des erreurs",
                    selected_models
                )
                
                fig_error = create_error_chart(df_predictions, model_for_error)
                st.plotly_chart(fig_error, use_container_width=True)
                
                # Calcul erreurs moyennes
                errors_list = []
                for _, row in df_predictions.iterrows():
                    if model_for_error in row and not pd.isna(row[model_for_error]):
                        error = abs((row['Reel'] - row[model_for_error]) / row['Reel']) * 100
                        errors_list.append(error)
                
                if errors_list:
                    col1, col2, col3 = st.columns(3)
                    





                    with col1:
                        st.metric("Erreur Minimale", f"{min(errors_list):.2f}%")
                    
                    with col2:
                        st.metric("Erreur Maximale", f"{max(errors_list):.2f}%")
                    
                    with col3:
                        st.metric("Erreur Moyenne", f"{np.mean(errors_list):.2f}%")
            
            else:
                st.warning("Veuillez sélectionner au moins un modèle à afficher")
        
        else:
            st.warning(f"Données de prédictions non trouvées pour {component_selected}")
    
    # ========================================================================
    # PAGE 4: ANALYSE COMPARATIVE
    # ========================================================================
    
    elif page == "Analyse comparative":
        
        st.markdown(f'<div class="section-header">Analyse Comparative - Agence {selected_agency}</div>', 
                    unsafe_allow_html=True)
        
        st.write("Cette section compare les performances globales des différents modèles sur les trois composantes.")
        
        # Charger toutes les métriques forecast
        df_forecast = load_metrics(selected_agency, 'forecast')
        
        if df_forecast is not None:
            
            # 1. Comparaison globale par modèle
            st.subheader("1. Performance Globale par Modèle")
            
            # Calculer MAPE moyen par modèle sur toutes les composantes
            model_performance = df_forecast.groupby('Modele')['MAPE'].agg(['mean', 'std', 'min', 'max']).reset_index()
            model_performance.columns = ['Modele', 'MAPE_Moyen', 'Ecart_Type', 'MAPE_Min', 'MAPE_Max']
            model_performance = model_performance.sort_values('MAPE_Moyen')
            
            # Graphique radar
            models_top5 = model_performance.head(5)['Modele'].tolist()
            
            fig_radar = go.Figure()
            
            for model in models_top5:
                df_model = df_forecast[df_forecast['Modele'] == model]
                
                values = []
                categories = []
                
                for component in ['encaissements', 'decaissements', 'Besoin']:
                    df_comp = df_model[df_model['Colonne'] == component]
                    if not df_comp.empty:
                        values.append(df_comp['MAPE'].values[0])
                        categories.append(component.capitalize())
                
                # Fermer le radar
                values.append(values[0])
                categories.append(categories[0])
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model.upper()
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(df_forecast['MAPE'].max(), 50)]
                    )
                ),
                title="Comparaison MAPE par Composante (Top 5 Modèles)",
                height=600
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Tableau performance globale
            st.dataframe(
                model_performance.style.format({
                    'MAPE_Moyen': '{:.2f}%',
                    'Ecart_Type': '{:.2f}%',
                    'MAPE_Min': '{:.2f}%',
                    'MAPE_Max': '{:.2f}%'
                }).background_gradient(subset=['MAPE_Moyen'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            # 2. Matrice de performance
            st.subheader("2. Matrice de Performance (MAPE)")
            
            # Créer pivot table
            pivot_mape = df_forecast.pivot_table(
                values='MAPE', 
                index='Modele', 
                columns='Colonne', 
                aggfunc='first'
            )
            
            # Heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_mape.values,
                x=pivot_mape.columns,
                y=pivot_mape.index,
                colorscale='RdYlGn_r',
                text=pivot_mape.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 12},
                colorbar=dict(title="MAPE (%)")
            ))
            
            fig_heatmap.update_layout(
                title="Matrice MAPE : Modèles × Composantes",
                xaxis_title="Composante",
                yaxis_title="Modèle",
                height=600
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 3. Meilleur modèle par composante
            st.subheader("3. Recommandations par Composante")
            
            for component in ['encaissements', 'decaissements', 'Besoin']:
                df_comp = df_forecast[df_forecast['Colonne'] == component].copy()
                df_comp = df_comp.sort_values('MAPE')
                
                best = df_comp.iloc[0]
                second_best = df_comp.iloc[1] if len(df_comp) > 1 else None
                
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown(f"**{component.upper()}**")
                    st.metric("Meilleur Modèle", best['Modele'].upper())
                    st.metric("MAPE", f"{best['MAPE']:.2f}%")
                    st.metric("Performance", get_performance_label(best['MAPE']))
                
                with col2:
                    st.markdown("**Analyse**")
                    
                    if best['MAPE'] < 15:
                        st.success(
                            f"Le modèle {best['Modele']} est **excellent** pour prévoir "
                            f"les {component}. Déploiement recommandé en production."
                        )
                    elif best['MAPE'] < 30:
                        st.info(
                            f"Le modèle {best['Modele']} est **bon** pour prévoir "
                            f"les {component}. Déploiement possible avec monitoring."
                        )
                    elif best['MAPE'] < 50:
                        st.warning(
                            f"Le modèle {best['Modele']} a une précision **moyenne** "
                            f"pour les {component}. Surveillance renforcée nécessaire."
                        )
                    else:
                        st.error(
                            f"Le modèle {best['Modele']} a une précision **insuffisante** "
                            f"pour les {component}. Enrichissement des données requis."
                        )
                    
                    if second_best is not None:
                        st.write(
                            f"Alternative : **{second_best['Modele']}** "
                            f"(MAPE = {second_best['MAPE']:.2f}%)"
                        )
                
                st.markdown("---")
            
            # 4. Synthèse globale
            st.subheader("4. Synthèse Globale")
            
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            
            # Trouver le modèle le plus polyvalent
            best_overall = model_performance.iloc[0]
            
            st.markdown("**Recommandation Globale**")
            st.write(
                f"Le modèle **{best_overall['Modele'].upper()}** présente "
                f"les meilleures performances en moyenne sur les trois composantes "
                f"(MAPE moyen = {best_overall['MAPE_Moyen']:.2f}%)."
            )
            
            # Analyse de stabilité
            if best_overall['Ecart_Type'] < 5:
                st.write(
                    f"Sa performance est **très stable** entre les composantes "
                    f"(écart-type = {best_overall['Ecart_Type']:.2f}%), "
                    f"ce qui en fait un choix fiable pour un déploiement unifié."
                )
            elif best_overall['Ecart_Type'] < 15:
                st.write(
                    f"Sa performance est **relativement stable** "
                    f"(écart-type = {best_overall['Ecart_Type']:.2f}%), "
                    f"mais une approche spécifique par composante pourrait améliorer les résultats."
                )
            else:
                st.write(
                    f"Sa performance **varie significativement** entre les composantes "
                    f"(écart-type = {best_overall['Ecart_Type']:.2f}%). "
                    f"Il est recommandé d'utiliser des modèles différents par composante."
                )
            
            # Comparaison approches
            st.markdown("**Comparaison des Approches**")
            
            individual_models = ['xgboost', 'prophet', 'arima', 'moving_avg']
            ensemble_models = [m for m in df_forecast['Modele'].unique() 
                              if 'ensemble' in m.lower()]
            
            if ensemble_models:
                mape_individual = df_forecast[df_forecast['Modele'].isin(individual_models)]['MAPE'].mean()
                mape_ensemble = df_forecast[df_forecast['Modele'].isin(ensemble_models)]['MAPE'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Modèles Individuels", f"{mape_individual:.2f}%")
                
                with col2:
                    st.metric("Modèles Ensemblistes", f"{mape_ensemble:.2f}%")
                
                with col3:
                    improvement = ((mape_individual - mape_ensemble) / mape_individual) * 100
                    st.metric(
                        "Amélioration",
                        f"{improvement:.1f}%",
                        delta=f"{improvement:.1f}%",
                        delta_color="normal"
                    )
                
                if mape_ensemble < mape_individual:
                    st.success(
                        "Les **modèles ensemblistes** surpassent les modèles individuels "
                        f"avec une amélioration de {improvement:.1f}%, validant l'approche "
                        "de combinaison de modèles."
                    )
                else:
                    st.info(
                        "Les **modèles individuels** restent compétitifs face aux ensembles, "
                        "suggérant que la simplicité peut être préférable dans ce contexte."
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.warning("Données de forecast non disponibles")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            Système de Prédiction du Flux de Liquidité - Version 1.0<br>
            Données actualisées en temps réel
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    main()