import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ============= PARTIE 0: ANALYSE INITIALE (EDA) =============
def initial_eda(file_path='Agence_00001.csv'):
    """Analyse statistique initiale des données pour comprendre les patterns"""
    print("="*60)
    print("ANALYSE INITIALE DES DONNÉES (EDA)")
    print("="*60)
    
    df = pd.read_csv(file_path, parse_dates=['Date Opération'])
    df = df.sort_values('Date Opération').set_index('Date Opération')
    print(f"Données chargées : {len(df)} observations")
    print(f"Période : {df.index.min().date()} à {df.index.max().date()}")
    
    print("\nStatistiques descriptives du Besoin :")
    print(df['Besoin'].describe())
    print("\nCommentaire : Le Besoin moyen est de {:.2f}, avec une volatilité (std) de {:.2f}. Min/Max indiquent des extrêmes possibles en surplus/déficit.".format(df['Besoin'].mean(), df['Besoin'].std()))
    
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Besoin'], label='Besoin (Flux Net)', color='blue')
    plt.title('Série Temporelle du Besoin')
    plt.xlabel('Date')
    plt.ylabel('Besoin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Commentaire : Le plot montre des patterns potentiels (ex. : pics mensuels). Vérifiez la saisonnalité.")
    
    plt.figure(figsize=(14, 6))
    plot_acf(df['Besoin'], lags=30, title='Autocorrélation (ACF) du Besoin')
    plt.show()
    print("Commentaire : L'ACF indique des corrélations fortes aux lags 1-7 (hebdo), et peut-être mensuelles ~lag 20-22. Cela justifie les features lags et rolling sur 7 jours.")
    
    return df

# ============= PARTIE 1: PRÉPARATION DES DONNÉES =============
def prepare_xgboost_data(df):
    """Préparer les features temporelles pour capturer patterns"""
    df_feat = df[['Besoin']].copy()
    
    for i in range(1, 8):
        df_feat[f'lag_{i}'] = df_feat['Besoin'].shift(i)
    
    df_feat['rolling_mean_7'] = df_feat['Besoin'].rolling(7).mean()
    df_feat['rolling_std_7'] = df_feat['Besoin'].rolling(7).std()
    df_feat['rolling_min_7'] = df_feat['Besoin'].rolling(7).min()
    df_feat['rolling_max_7'] = df_feat['Besoin'].rolling(7).max()
    
    df_feat['jour_semaine'] = df_feat.index.dayofweek
    df_feat['mois'] = df_feat.index.month
    df_feat['jour_mois'] = df_feat.index.day
    df_feat['trimestre'] = df_feat.index.quarter
    df_feat['annee'] = df_feat.index.year
    
    df_feat['debut_mois'] = (df_feat.index.day <= 5).astype(int)
    df_feat['fin_mois'] = (df_feat.index.day >= 25).astype(int)
    df_feat['milieu_mois'] = ((df_feat.index.day >= 14) & (df_feat.index.day <= 16)).astype(int)
    
    df_feat['diff_1'] = df_feat['Besoin'].diff(1)
    df_feat['diff_5'] = df_feat['Besoin'].diff(5)
    df_feat['mois_sin'] = np.sin(2 * np.pi * df_feat['mois'] / 12)
    df_feat['mois_cos'] = np.cos(2 * np.pi * df_feat['mois'] / 12)
    print("Ajout sine/cos mois pour mieux capturer saisonnalité cyclique.")
    
    df_feat = df_feat.dropna()
    print(f"Après création features : {len(df_feat)} observations")
    print("Commentaire : Features prêtes. Les temporelles (mois, jour) capturent saisonnalité ; lags/rolling gèrent autocorrélation.")
    
    return df_feat

# ============= PARTIE 2: ENTRAÎNEMENT ET PRÉDICTION =============
def train_and_predict_xgboost(df_feat, train_end='2025-02-28', test_end='2025-06-30'):
    """Entraîner XGBoost avec splits chronologiques et params optimisés"""
    
    train = df_feat[df_feat.index <= train_end]
    test = df_feat[(df_feat.index > train_end) & (df_feat.index <= test_end)]
    validate = df_feat[df_feat.index > test_end]
    
    print(f"\nDivision des données :")
    print(f"Train : {len(train)} obs (jusqu'au {train.index.max().date()}) - Pour apprendre patterns longs.")
    print(f"Test  : {len(test)} obs ({test.index.min().date()} au {test.index.max().date()}) - Pour évaluation intermédiaire.")
    print(f"Validate : {len(validate)} obs (dès {validate.index.min().date()}) - Pour forecasting final, non vu avant.")
    
    X_train = train.drop('Besoin', axis=1)
    y_train = train['Besoin']
    X_test = test.drop('Besoin', axis=1)
    y_test = test['Besoin']
    X_validate = validate.drop('Besoin', axis=1)
    y_validate = validate['Besoin']
    
    print("\nEntraînement du modèle XGBoost...")
    # model = XGBRegressor(
    #     n_estimators=1000,
    #     max_depth=5,            # Augmenté pour capturer patterns complexes
    #     learning_rate=0.05,     # Réduit pour stabilité
    #     reg_alpha=1.0,
    #     reg_lambda=2.0,
    #     gamma=1.0,
    #     min_child_weight=20,
    #     subsample=0.5,
    #     colsample_bytree=0.5,
    #     random_state=42,
    #     eval_metric='rmse',
    #     early_stopping_rounds=20
    # )





    # model = XGBRegressor(
    #     n_estimators=1000,
    #     max_depth=4,  # Ajusté
    #     learning_rate=0.03,  # Réduit
    #     reg_alpha=1.0,
    #     reg_lambda=2.0,
    #     gamma=1.0,
    #     min_child_weight=20,
    #     subsample=0.5,
    #     colsample_bytree=0.5,
    #     colsample_bynode=0.5,  # Ajout pour diversité
    #     random_state=42,
    #     eval_metric='rmse',
    #     early_stopping_rounds=20
    # )
    # Dans XGBoost
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        reg_alpha=0.5,  # Réduit
        reg_lambda=2.0,
        gamma=1.0,
        min_child_weight=20,
        subsample=0.8,  # Augmenté
        colsample_bytree=0.8,  # Augmenté
        colsample_bynode=0.8,
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=30  # Plus tolérant
    )





    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    print(f"Meilleur iteration : {model.best_iteration}. Commentaire : Early_stop empêche overfitting ; params ajustés pour meilleurs forecasts.")
    
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    preds_validate = model.predict(X_validate)
    
    return model, X_train, y_train, X_test, y_test, X_validate, y_validate, preds_train, preds_test, preds_validate, train, test, validate

# ============= PARTIE 3: MÉTRIQUES D'ÉVALUATION =============
def calculate_metrics(y_true, y_pred, dataset_name=""):
    """Calculer métriques avec gestion zéros"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) != 0, np.abs(y_true), 1))) * 100
    incertitude = np.mean(np.abs(y_true - y_pred) / np.where(np.abs(y_true) != 0, np.abs(y_true), 1)) * 100
    
    print(f"\n{'='*50}")
    print(f"MÉTRIQUES {dataset_name}")
    print(f"{'='*50}")
    print(f"MAE  : {mae:,.0f} - Erreur absolue moyenne.")
    print(f"RMSE : {rmse:,.0f} - Erreur quadratique (punit plus les grosses erreurs).")
    print(f"MAPE : {mape:.2f}% - Erreur en % (bon si <20-30% pour banking).")
    print(f"R²   : {r2:.4f} - Explication variance (proche 1 = bon fit).")
    print(f"Incertitude moyenne : {incertitude:.2f}% - Variabilité relative.")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'incertitude': incertitude}

# ============= PARTIE 4: FORECASTING RÉCURSIF =============
def recursive_forecast(model, last_known_data, horizon):
    """Forecast récursif jour par jour, avec stabilisation via moyenne mobile"""
    predictions = []
    current_df = last_known_data.copy()
    
    for d in range(horizon):
        next_date = current_df.index.max() + pd.Timedelta(days=1)
        while next_date.dayofweek >= 5:
            next_date += pd.Timedelta(days=1)
        
        next_row = pd.DataFrame(index=[next_date], columns=current_df.columns)
        next_row['Besoin'] = np.nan
        
        next_row['jour_semaine'] = next_date.dayofweek
        next_row['mois'] = next_date.month
        next_row['jour_mois'] = next_date.day
        next_row['trimestre'] = next_date.quarter
        next_row['annee'] = next_date.year
        next_row['debut_mois'] = int(next_date.day <= 5)
        next_row['fin_mois'] = int(next_date.day >= 25)
        next_row['milieu_mois'] = int(14 <= next_date.day <= 16)
        
        for i in range(1, 8):
            next_row[f'lag_{i}'] = current_df['Besoin'].iloc[-i] if len(current_df) >= i else np.nan
        
        next_row['diff_1'] = current_df['Besoin'].iloc[-1] - current_df['Besoin'].iloc[-2] if len(current_df) >= 2 else 0
        next_row['diff_5'] = current_df['Besoin'].iloc[-1] - current_df['Besoin'].iloc[-6] if len(current_df) >= 6 else 0
        
        last_7 = current_df['Besoin'].tail(7).values
        next_row['rolling_mean_7'] = np.nanmean(last_7)
        next_row['rolling_std_7'] = np.nanstd(last_7)
        next_row['rolling_min_7'] = np.nanmin(last_7)
        next_row['rolling_max_7'] = np.nanmax(last_7)
        
        X_next = next_row.drop('Besoin', axis=1)
        pred = model.predict(X_next)[0]
        
        # Stabilisation : moyenne mobile sur 3 dernières préds (si dispo)
        # if len(predictions) >= 3:
        #     pred = np.mean([pred] + predictions[-3:])


        # if len(predictions) >= 5:
        #     pred = np.mean([pred] + predictions[-5:])

        
        
        if len(predictions) >= 5:
            pred = np.mean([pred] + predictions[-5:])


        
        predictions.append(pred)
        next_row['Besoin'] = pred
        current_df = pd.concat([current_df, next_row])
    
    pred_index = pd.date_range(start=last_known_data.index.max() + pd.Timedelta(days=1), periods=horizon, freq='B')
    print(f"Commentaire : Forecast récursif sur {horizon} jours terminé avec stabilisation. Erreur cumulée réduite.")
    return pd.Series(predictions, index=pred_index)

# ============= PARTIE 5: VISUALISATIONS COMPLÈTES =============
def create_xgboost_visualizations(model, X_train, y_train, preds_train, X_test, y_test, preds_test, X_validate, y_validate, preds_validate, train, test, validate, rec_preds_dict):
    """Visualisations pour analyser fit, overfitting, résidus, etc."""
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Prédictions vs Réel
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(y_test.index, y_test, 'o-', color=colors[0], label='Test Réel')
    ax.plot(y_test.index, preds_test, 's-', color=colors[1], label='Test Prédit')
    ax.plot(y_validate.index, y_validate, 'o-', color=colors[2], label='Validate Réel')
    ax.plot(y_validate.index, preds_validate, 's-', color=colors[3], label='Validate Prédit')
    ax.set_title('Prédictions vs Valeurs Réelles (Test et Validate)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Besoin')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    print("Commentaire : Si préds collent au réel sur test/validate, bon généralisation.")
    
    # 2. Forecasts Récursifs
    horizons = [7, 14, 30]
    for h in horizons:
        if h > len(validate):
            print(f"Horizon {h} trop long pour validate ({len(validate)} obs). Skip.")
            continue
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(validate.index[:h], validate['Besoin'][:h], 'o-', color=colors[0], label='Réel')
        ax.plot(rec_preds_dict[h].index, rec_preds_dict[h], 's-', color=colors[1], label=f'Forecast {h} jours')
        ax.fill_between(rec_preds_dict[h].index, rec_preds_dict[h] * 0.9, rec_preds_dict[h] * 1.1, alpha=0.2, color=colors[1], label='±10%')
        ax.set_title(f'Forecast Récursif sur {h} Jours')
        ax.set_xlabel('Date')
        ax.set_ylabel('Besoin')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"Commentaire Horizon {h} : Déviation dans ±10% = bon forecast.")
    
    # 3. Résidus
    residuals = y_test - preds_test
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(y_test.index, residuals, 'o-', color=colors[3])
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_title('Résidus (Test)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Résidu')
    axes[1].hist(residuals, bins=20, color=colors[2], alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title('Distribution Résidus')
    plt.tight_layout()
    plt.show()
    print("Commentaire : Résidus centrés sur 0 = bon modèle.")
    
    # 4. Importance Features
    feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feat_imp, color=colors[0])
    plt.title('Importance des Features')
    plt.show()
    print("Commentaire : Lags et mois souvent clés pour TS.")
    
    # 5. Learning Curves
    if hasattr(model, 'evals_result_'):
        results = model.evals_result()
        plt.figure(figsize=(10, 5))
        plt.plot(results['validation_0']['rmse'], label='Train RMSE')
        plt.plot(results['validation_1']['rmse'], label='Test RMSE')
        plt.title('Courbes d\'Apprentissage')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Commentaire : Stabilité entre Train/Test = pas d'overfitting.")

# ============= PARTIE 6: FORECAST MULTI-HORIZONS ET SYNTHÈSE =============
def forecast_multiple_horizons(model, train, test, validate):
    """Forecasts récursifs sur validate avec métriques"""
    horizons = [7, 14, 30]
    results = {}
    rec_preds_dict = {}
    
    last_known = pd.concat([train, test])
    
    for h in horizons:
        print(f"\n{'='*50}")
        print(f"FORECAST RÉCURSIF HORIZON: {h} JOURS")
        print(f"{'='*50}")
        
        rec_preds = recursive_forecast(model, last_known, h)
        actuals = validate['Besoin'].head(h)
        
        if len(actuals) < h:
            print(f"Seulement {len(actuals)} jours réels disponibles.")
            h = len(actuals)
        
        metrics = calculate_metrics(actuals, rec_preds[:h], f"HORIZON {h} JOURS")
        results[h] = metrics
        rec_preds_dict[h] = rec_preds[:h]
    
    summary_data = [{'Horizon (jours)': h, **m} for h, m in results.items()]
    df_summary = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF - PERFORMANCE FORECAST")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)
    print("Commentaire : MAPE <20% sur 7j = bon pour trésorerie.")
    
    df_summary.to_csv('xgboost_performance_summary.csv', index=False)
    print("\nRésultats sauvegardés dans 'xgboost_performance_summary.csv'")
    
    return results, rec_preds_dict

# ============= EXÉCUTION PRINCIPALE =============
def main_xgboost_analysis():
    """Analyse complète XGBoost pour prédiction Besoin"""
    
    df = initial_eda('Agence_00001.csv')
    df_feat = prepare_xgboost_data(df)
    
    model, X_train, y_train, X_test, y_test, X_validate, y_validate, preds_train, preds_test, preds_validate, train, test, validate = \
        train_and_predict_xgboost(df_feat)
    
    calculate_metrics(y_train, preds_train, "TRAIN")
    calculate_metrics(y_test, preds_test, "TEST")
    calculate_metrics(y_validate, preds_validate, "VALIDATE (Direct)")
    
    results_horizons, rec_preds_dict = forecast_multiple_horizons(model, train, test, validate)
    
    create_xgboost_visualizations(model, X_train, y_train, preds_train, X_test, y_test, preds_test, X_validate, y_validate, preds_validate, train, test, validate, rec_preds_dict)
    
    return model, results_horizons

if __name__ == "__main__":
    model, results = main_xgboost_analysis()