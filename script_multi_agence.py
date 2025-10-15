import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FOLDER = 'agences_split'
OUTPUT_FOLDER = 'results'
GRAPHS_FOLDER = 'results/graphs'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

# ============================================================================
# FONCTIONS ORIGINALES (NON MODIFIÉES)
# ============================================================================

def prepare_component_features(df, component_name):
    """Créer features spécialisées pour encaissements OU decaissements"""
    df_feat = df[[component_name]].copy()
    
    # 1. LAGS - Valeurs passées
    for i in range(1, 8):
        df_feat[f'lag_{i}'] = df_feat[component_name].shift(i)
    
    # 2. ROLLING WINDOWS - Tendances locales
    df_feat['rolling_mean_7'] = df_feat[component_name].rolling(7).mean()
    df_feat['rolling_std_7'] = df_feat[component_name].rolling(7).std()
    df_feat['rolling_min_7'] = df_feat[component_name].rolling(7).min()
    df_feat['rolling_max_7'] = df_feat[component_name].rolling(7).max()
    
    # 3. FEATURES TEMPORELLES - Saisonnalité
    df_feat['jour_semaine'] = df_feat.index.dayofweek
    df_feat['mois'] = df_feat.index.month
    df_feat['jour_mois'] = df_feat.index.day
    df_feat['trimestre'] = df_feat.index.quarter
    df_feat['annee'] = df_feat.index.year
    
    # 4. FEATURES BANCAIRES - Patterns métier
    df_feat['debut_mois'] = (df_feat.index.day <= 5).astype(int)
    df_feat['fin_mois'] = (df_feat.index.day >= 25).astype(int)
    df_feat['milieu_mois'] = ((df_feat.index.day >= 14) & (df_feat.index.day <= 16)).astype(int)
    
    # 5. DIFFÉRENCES - Momentum
    df_feat['diff_1'] = df_feat[component_name].diff(1)
    df_feat['diff_5'] = df_feat[component_name].diff(5)
    
    # 6. FEATURES CYCLIQUES - Saisonnalité circulaire
    df_feat['mois_sin'] = np.sin(2 * np.pi * df_feat['mois'] / 12)
    df_feat['mois_cos'] = np.cos(2 * np.pi * df_feat['mois'] / 12)
    
    # 7. FEATURES CROISÉES - Influence de l'autre composante
    other_component = 'decaissements' if component_name == 'encaissements' else 'encaissements'
    if other_component in df.columns:
        # Ratio par rapport à l'autre composante
        df_feat['ratio_vs_other'] = df[component_name] / (df[other_component] + 1e-8)
        
        # Lags de l'autre composante (influence croisée)
        for i in [1, 3, 7]:
            df_feat[f'other_lag_{i}'] = df[other_component].shift(i)
            
        # Différence avec l'autre composante
        df_feat['diff_vs_other'] = df[component_name] - df[other_component]
    
    # Nettoyage
    df_feat = df_feat.dropna()
    
    return df_feat


def train_separate_component_models(df, train_end='2025-02-28', test_end='2025-06-30'):
    """Entraîner des modèles XGBoost distincts pour chaque composante"""
    components = ['encaissements', 'decaissements']
    models = {}
    results = {}
    
    for component in components:
        # Créer features spécialisées pour cette composante
        df_feat = prepare_component_features(df, component)
        
        # Splits temporels chronologiques
        train = df_feat[df_feat.index <= train_end]
        test = df_feat[(df_feat.index > train_end) & (df_feat.index <= test_end)]
        validate = df_feat[df_feat.index > test_end]
        
        if len(validate) == 0:
            continue
        
        # Préparation X, y
        X_train = train.drop(component, axis=1)
        y_train = train[component]
        X_test = test.drop(component, axis=1)
        y_test = test[component]
        X_validate = validate.drop(component, axis=1)
        y_validate = validate[component]
        
        # Modèle XGBoost spécialisé
        model = XGBRegressor(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.03,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=1.0,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bynode=0.8,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=30
        )
        
        # Entraînement avec early stopping
        model.fit(X_train, y_train, 
                 eval_set=[(X_train, y_train), (X_test, y_test)], 
                 verbose=False)
        
        # Prédictions sur tous les sets
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)
        preds_validate = model.predict(X_validate)
        
        # Calcul métriques pour chaque dataset
        results[component] = {}
        
        for dataset_name, y_true, y_pred in [
            ('train', y_train, preds_train),
            ('test', y_test, preds_test), 
            ('validate', y_validate, preds_validate)
        ]:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
            r2 = r2_score(y_true, y_pred)
            
            results[component][dataset_name] = {
                'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2
            }
        
        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Stockage modèle et informations
        models[component] = {
            'model': model,
            'train_data': train,
            'test_data': test,
            'validate_data': validate,
            'feature_importance': feature_importance,
            'predictions': {
                'train': preds_train,
                'test': preds_test,
                'validate': preds_validate
            }
        }
    
    return models, results


def recursive_forecast_single_component(model_info, last_known_data, component, horizon):
    """
    Forecasting récursif pour UNE SEULE composante
    OBJECTIF : Voir performance individuelle sans mélanger avec l'autre composante
    """
    model = model_info['model']
    
    # Vérifier disponibilité données composante
    if component not in last_known_data.columns:
        return None
    
    # Historique récent de cette composante
    base_values = last_known_data[component].tail(15).values.tolist()
    
    # Features attendues par le modèle
    expected_features = list(model_info['train_data'].drop(component, axis=1).columns)
    
    predictions = []
    current_date = last_known_data.index[-1]
    current_history = base_values.copy()
    
    for day in range(horizon):
        # Calcul date suivante (skip weekends)
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.dayofweek >= 5:
            next_date += pd.Timedelta(days=1)
        
        # Nouvelle ligne pour features
        next_row = pd.DataFrame(index=[next_date])
        
        # 1. FEATURES TEMPORELLES (toujours exactes en production)
        next_row['jour_semaine'] = next_date.dayofweek
        next_row['mois'] = next_date.month
        next_row['jour_mois'] = next_date.day
        next_row['trimestre'] = next_date.quarter
        next_row['annee'] = next_date.year
        next_row['debut_mois'] = int(next_date.day <= 5)
        next_row['fin_mois'] = int(next_date.day >= 25)
        next_row['milieu_mois'] = int(14 <= next_date.day <= 16)
        next_row['mois_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        next_row['mois_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        
        # 2. LAGS (utilisant historique + prédictions récentes)
        for i in range(1, 8):
            if len(current_history) >= i:
                next_row[f'lag_{i}'] = current_history[-i]
            else:
                next_row[f'lag_{i}'] = current_history[-1]
        
        # 3. ROLLING FEATURES
        window = current_history[-7:] if len(current_history) >= 7 else current_history
        if len(window) > 0:
            next_row['rolling_mean_7'] = np.mean(window)
            next_row['rolling_std_7'] = np.std(window) if len(window) > 1 else 0
            next_row['rolling_min_7'] = np.min(window)
            next_row['rolling_max_7'] = np.max(window)
        else:
            next_row['rolling_mean_7'] = base_values[-1]
            next_row['rolling_std_7'] = 0
            next_row['rolling_min_7'] = base_values[-1]
            next_row['rolling_max_7'] = base_values[-1]
        
        # 4. DIFFÉRENCES
        if len(current_history) >= 2:
            next_row['diff_1'] = current_history[-1] - current_history[-2]
        else:
            next_row['diff_1'] = 0
            
        if len(current_history) >= 6:
            next_row['diff_5'] = current_history[-1] - current_history[-6]
        else:
            next_row['diff_5'] = 0
        
        # 5. FEATURES CROISÉES (si autre composante disponible)
        other_component = 'decaissements' if component == 'encaissements' else 'encaissements'
        if other_component in last_known_data.columns:
            other_base = last_known_data[other_component].tail(15).values
            
            # Lags de l'autre composante
            for i in [1, 3, 7]:
                if len(other_base) >= i:
                    next_row[f'other_lag_{i}'] = other_base[-i]
                else:
                    next_row[f'other_lag_{i}'] = other_base[-1] if len(other_base) > 0 else 0
            
            # Ratio avec autre composante
            if len(other_base) > 0:
                next_row['ratio_vs_other'] = current_history[-1] / (other_base[-1] + 1e-8)
            else:
                next_row['ratio_vs_other'] = 1.0
                
            # Différence avec autre composante
            if len(other_base) > 0:
                next_row['diff_vs_other'] = current_history[-1] - other_base[-1]
            else:
                next_row['diff_vs_other'] = 0
        
        # 6. COMPLÉTER FEATURES MANQUANTES
        for feat in expected_features:
            if feat not in next_row.columns:
                next_row[feat] = 0
        
        # 7. RÉORGANISER DANS L'ORDRE EXACT DU MODÈLE
        next_row = next_row[expected_features]
        
        # 8. PRÉDICTION
        pred = model.predict(next_row)[0]
        
        # 9. CONTRAINTES DE COHÉRENCE (éviter valeurs aberrantes)
        if len(current_history) >= 10:
            recent_mean = np.mean(current_history[-10:])
            recent_std = np.std(current_history[-10:])
            
            if abs(pred - recent_mean) > 3 * recent_std:
                pred = recent_mean + np.sign(pred - recent_mean) * 2.5 * recent_std
        
        predictions.append(pred)
        current_history.append(pred)
        current_date = next_date
    
    # Créer série résultat avec dates correctes
    pred_dates = []
    date = last_known_data.index[-1]
    for _ in range(horizon):
        date += pd.Timedelta(days=1)
        while date.dayofweek >= 5:
            date += pd.Timedelta(days=1)
        pred_dates.append(date)
    
    result = pd.Series(predictions, index=pred_dates)
    
    return result


def calculate_component_metrics(y_true, y_pred):
    """Calculer métriques pour une composante"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    
    if mape <= 10:
        evaluation = "excellente"
    elif mape <= 20:
        evaluation = "bonne"
    elif mape <= 35:
        evaluation = "moyenne"
    else:
        evaluation = "insuffisante"
    
    return {
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
        'evaluation': evaluation
    }


# ============================================================================
# PIPELINE POUR UNE AGENCE
# ============================================================================

def process_single_agency(file_path):
    """Traiter une seule agence avec le code original"""
    
    agency_code = Path(file_path).stem
    print(f"\n{'='*80}")
    print(f"TRAITEMENT: {agency_code}")
    print(f"{'='*80}")
    
    try:
        # Chargement données
        df = pd.read_csv(file_path, parse_dates=['Date Opération'])
        df = df.sort_values('Date Opération').set_index('Date Opération')
        
        # Vérification colonnes
        required_cols = ['encaissements', 'decaissements', 'Besoin']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"ERREUR - Colonnes manquantes: {missing_cols}")
            return None
        
        print(f"Données chargées: {len(df)} observations")
        print(f"Période: {df.index.min().date()} à {df.index.max().date()}")
        
        # Entraînement modèles
        print("\nEntraînement modèles séparés...")
        models, results = train_separate_component_models(df)
        
        if not models:
            print("ERREUR - Aucun modèle entraîné")
            return None
        
        # Affichage métriques validation
        for component in models.keys():
            val_metrics = results[component]['validate']
            print(f"\n{component}:")
            print(f"  Train MAPE: {results[component]['train']['mape']:.2f}%")
            print(f"  Test MAPE: {results[component]['test']['mape']:.2f}%")
            print(f"  Validate MAPE: {val_metrics['mape']:.2f}%")
            print(f"  Validate R2: {val_metrics['r2']:.4f}")
        
        # Forecasting récursif
        print("\nForecasting récursif (5 jours)...")
        
        last_known = df[df.index <= '2025-06-30']
        validate_data = df[df.index > '2025-06-30']
        
        horizon = min(5, len(validate_data))
        
        if horizon == 0:
            print("ERREUR - Pas de données validation")
            return None
        
        component_forecasts = {}
        component_metrics = {}
        forecast_daily_errors = {}
        
        for component in models.keys():
            forecast_result = recursive_forecast_single_component(
                models[component], last_known, component, horizon
            )
            
            if forecast_result is not None:
                component_forecasts[component] = forecast_result
                
                # Validation
                actual_values = validate_data[component].head(horizon)
                
                if len(actual_values) >= horizon:
                    # Métriques globales
                    metrics = calculate_component_metrics(actual_values, forecast_result)
                    component_metrics[component] = metrics
                    
                    print(f"\n{component} Forecast MAPE: {metrics['mape']:.2f}%")
                    
                    # Erreurs quotidiennes
                    daily_errors = []
                    for i in range(horizon):
                        actual = actual_values.iloc[i]
                        predicted = forecast_result.iloc[i]
                        error_pct = abs(predicted - actual) / abs(actual) * 100
                        
                        daily_errors.append({
                            'day': i + 1,
                            'prediction': predicted,
                            'actual': actual,
                            'error_pct': error_pct,
                            'error_abs': abs(predicted - actual)
                        })
                    
                    forecast_daily_errors[component] = daily_errors
        
        # Calcul Besoin (encaissements - decaissements)
        if 'encaissements' in component_forecasts and 'decaissements' in component_forecasts:
            besoin_forecast = component_forecasts['encaissements'] - component_forecasts['decaissements']
            besoin_actual = validate_data['Besoin'].head(horizon)
            
            if len(besoin_actual) >= horizon:
                besoin_metrics = calculate_component_metrics(besoin_actual, besoin_forecast)
                component_metrics['Besoin'] = besoin_metrics
                
                print(f"\nBesoin Forecast MAPE: {besoin_metrics['mape']:.2f}%")
                
                # Erreurs quotidiennes Besoin
                daily_errors = []
                for i in range(horizon):
                    actual = besoin_actual.iloc[i]
                    predicted = besoin_forecast.iloc[i]
                    error_pct = abs(predicted - actual) / abs(actual) * 100
                    
                    daily_errors.append({
                        'day': i + 1,
                        'prediction': predicted,
                        'actual': actual,
                        'error_pct': error_pct,
                        'error_abs': abs(predicted - actual)
                    })
                
                forecast_daily_errors['Besoin'] = daily_errors
        
        # Graphiques
        print("\nGénération graphiques...")
        plot_training_curves(models, agency_code)
        plot_forecast_results(component_forecasts, validate_data, agency_code, horizon)
        
        print(f"\nSUCCES - {agency_code} traité")
        
        return {
            'agency_code': agency_code,
            'models': models,
            'results': results,
            'forecasts': component_forecasts,
            'forecast_metrics': component_metrics,
            'forecast_daily_errors': forecast_daily_errors,
            'horizon': horizon
        }
        
    except Exception as e:
        print(f"ERREUR - {agency_code}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_training_curves(models, agency_code):
    """Tracer courbes d'apprentissage"""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (col_name, model_info) in enumerate(models.items()):
        ax = axes[idx]
        
        model = model_info['model']
        train_rmse = model.evals_result()['validation_0']['rmse']
        test_rmse = model.evals_result()['validation_1']['rmse']
        best_iter = model.best_iteration
        
        ax.plot(train_rmse, label='Train RMSE', alpha=0.7)
        ax.plot(test_rmse, label='Test RMSE', alpha=0.7)
        ax.axvline(best_iter, color='red', linestyle='--', 
                  label=f'Best Iteration ({best_iter})')
        
        ax.set_title(f'{agency_code} - {col_name}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_FOLDER}/{agency_code}_training.png', dpi=150)
    plt.close()


def plot_forecast_results(forecasts, validate_data, agency_code, horizon):
    """Tracer résultats forecast"""
    n_cols = len(forecasts)
    
    if n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5*n_cols))
    
    if n_cols == 1:
        axes = [axes]
    
    for idx, (col_name, forecast) in enumerate(forecasts.items()):
        ax = axes[idx]
        
        actual = validate_data[col_name].head(horizon)
        
        days = list(range(1, len(actual) + 1))
        
        ax.plot(days, actual.values, 'o-', label='Actual', linewidth=2, markersize=8)
        ax.plot(days, forecast.values[:len(actual)], 's-', label='Predicted', linewidth=2, markersize=8)
        
        ax.fill_between(days,
                        forecast.values[:len(actual)] * 0.9,
                        forecast.values[:len(actual)] * 1.1,
                        alpha=0.2, label='10% Uncertainty')
        
        ax.set_title(f'{agency_code} - {col_name}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_FOLDER}/{agency_code}_forecast.png', dpi=150)
    plt.close()


# ============================================================================
# ORCHESTRATEUR MULTI-AGENCES
# ============================================================================

def process_all_agencies():
    """Traiter toutes les agences"""
    
    print("\n" + "="*80)
    print("SYSTEME DE PREDICTION MULTI-AGENCES")
    print("="*80)
    
    # Trouver fichiers
    data_path = Path(DATA_FOLDER)
    
    if not data_path.exists():
        print(f"ERREUR - Dossier {DATA_FOLDER} introuvable")
        return
    
    csv_files = sorted(list(data_path.glob('*.csv')))
    print(f"\nFichiers trouvés: {len(csv_files)}")
    
    if not csv_files:
        print("ERREUR - Aucun fichier CSV trouvé")
        return
    
    # Traitement
    all_results = []
    summary_data = []
    
    for idx, file_path in enumerate(csv_files, 1):
        print(f"\n\n{'#'*80}")
        print(f"PROGRESSION: {idx}/{len(csv_files)}")
        print(f"{'#'*80}")
        
        result = process_single_agency(str(file_path))
        
        if result:
            all_results.append(result)
            
            # Collecter données pour rapports
            agency_code = result['agency_code']
            
            # Métriques validation (prédiction directe)
            for component, metrics_dict in result['results'].items():
                for dataset_name, metrics in metrics_dict.items():
                    summary_data.append({
                        'agency_code': agency_code,
                        'target_column': component,
                        'dataset': dataset_name,
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape'],
                        'r2': metrics['r2']
                    })
            
            # Métriques forecast (production réelle)
            for component, metrics in result['forecast_metrics'].items():
                summary_data.append({
                    'agency_code': agency_code,
                    'target_column': component,
                    'dataset': 'forecast_global',
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'r2': metrics['r2'],
                    'avg_daily_error': np.mean([e['error_pct'] for e in result['forecast_daily_errors'][component]])
                })
            
            # Erreurs quotidiennes forecast
            for component, daily_errors in result['forecast_daily_errors'].items():
                for error_data in daily_errors:
                    summary_data.append({
                        'agency_code': agency_code,
                        'target_column': component,
                        'dataset': f"forecast_day_{error_data['day']}",
                        'prediction': error_data['prediction'],
                        'actual': error_data['actual'],
                        'error_pct': error_data['error_pct'],
                        'error_abs': error_data['error_abs']
                    })
    
    # Génération rapports
    print("\n" + "="*80)
    print("GENERATION RAPPORTS")
    print("="*80)
    
    if not summary_data:
        print("ERREUR - Aucune donnée à reporter")
        return
    
    df_summary = pd.DataFrame(summary_data)
    
    # Rapport complet
    output_file = f"{OUTPUT_FOLDER}/rapport_complet.csv"
    df_summary.to_csv(output_file, index=False)
    print(f"\n[1] Rapport complet: {output_file}")
    
    # Rapport validation (prédiction directe)
    df_validation = df_summary[df_summary['dataset'] == 'validate'].copy()
    if not df_validation.empty:
        validation_file = f"{OUTPUT_FOLDER}/rapport_validation.csv"
        df_validation.to_csv(validation_file, index=False)
        print(f"[2] Rapport validation: {validation_file}")
    
    # Rapport forecast global (PRODUCTION RÉELLE)
    df_forecast = df_summary[df_summary['dataset'] == 'forecast_global'].copy()
    if not df_forecast.empty:
        forecast_file = f"{OUTPUT_FOLDER}/rapport_forecast_global.csv"
        df_forecast.to_csv(forecast_file, index=False)
        print(f"[3] Rapport forecast global (PRODUCTION): {forecast_file}")
    
    # Rapport erreurs quotidiennes
    df_daily = df_summary[df_summary['dataset'].str.contains('forecast_day_', na=False)].copy()
    if not df_daily.empty:
        daily_file = f"{OUTPUT_FOLDER}/rapport_erreurs_quotidiennes.csv"
        df_daily.to_csv(daily_file, index=False)
        print(f"[4] Rapport erreurs quotidiennes: {daily_file}")
    
    # Statistiques par colonne
    generate_statistics(df_summary)
    
    # Fenêtres de prédiction
    generate_prediction_windows(df_daily)
    
    print("\n" + "="*80)
    print("TRAITEMENT TERMINE")
    print("="*80)
    print(f"\nAgences traitées avec succès: {len(all_results)}/{len(csv_files)}")
    print(f"Résultats dans: {OUTPUT_FOLDER}/")
    print(f"Graphiques dans: {GRAPHS_FOLDER}/")


def generate_statistics(df_summary):
    """Générer statistiques agrégées"""
    print("\n[5] Statistiques agrégées...")
    
    # Stats validation
    df_val = df_summary[df_summary['dataset'] == 'validate']
    
    if not df_val.empty:
        stats_val = df_val.groupby('target_column').agg({
            'mape': ['mean', 'std', 'min', 'max'],
            'r2': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        stats_val.to_csv(f"{OUTPUT_FOLDER}/stats_validation_par_colonne.csv")
        print(f"    Stats validation: {OUTPUT_FOLDER}/stats_validation_par_colonne.csv")
    
    # Stats forecast
    df_forecast = df_summary[df_summary['dataset'] == 'forecast_global']
    
    if not df_forecast.empty:
        stats_forecast = df_forecast.groupby('target_column').agg({
            'mape': ['mean', 'std', 'min', 'max'],
            # 'avg_daily_error': ['mean
            

            'avg_daily_error': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        stats_forecast.to_csv(f"{OUTPUT_FOLDER}/stats_forecast_par_colonne.csv")
        print(f"    Stats forecast: {OUTPUT_FOLDER}/stats_forecast_par_colonne.csv")
    
    # Résumé exécutif
    print("\n[6] Résumé exécutif...")
    
    if not df_val.empty:
        print("\nPERFORMANCES VALIDATION (prédiction directe):")
        for col in ['encaissements', 'decaissements']:
            col_data = df_val[df_val['target_column'] == col]
            if not col_data.empty:
                print(f"  {col}: MAPE moyen = {col_data['mape'].mean():.2f}%")
    
    if not df_forecast.empty:
        print("\nPERFORMANCES FORECAST (PRODUCTION RÉELLE - 5 jours):")
        for col in ['encaissements', 'decaissements', 'Besoin']:
            col_data = df_forecast[df_forecast['target_column'] == col]
            if not col_data.empty:
                print(f"  {col}: MAPE moyen = {col_data['mape'].mean():.2f}%")


def generate_prediction_windows(df_daily):
    """Générer rapport fenêtres de prédiction"""
    print("\n[7] Fenêtres de prédiction...")
    
    if df_daily.empty:
        print("    Aucune donnée quotidienne")
        return
    
    # Extraction jour
    df_daily['day_number'] = df_daily['dataset'].str.extract(r'forecast_day_(\d+)').astype(int)
    
    # Calcul précision par agence/colonne/jour
    precision_data = []
    
    for agency in df_daily['agency_code'].unique():
        for col in df_daily['target_column'].unique():
            df_subset = df_daily[
                (df_daily['agency_code'] == agency) & 
                (df_daily['target_column'] == col)
            ].copy()
            
            if df_subset.empty:
                continue
            
            for day in sorted(df_subset['day_number'].unique()):
                day_data = df_subset[df_subset['day_number'] == day]
                
                if not day_data.empty:
                    error_pct = day_data['error_pct'].iloc[0]
                    precision_pct = 100 - min(error_pct, 100)
                    
                    precision_data.append({
                        'agency_code': agency,
                        'target_column': col,
                        'day': day,
                        'error_pct': error_pct,
                        'precision_pct': precision_pct,
                        'prediction': day_data['prediction'].iloc[0],
                        'actual': day_data['actual'].iloc[0]
                    })
    
    if precision_data:
        df_precision = pd.DataFrame(precision_data)
        
        # Rapport détaillé
        precision_file = f"{OUTPUT_FOLDER}/fenetre_prediction_detaillee.csv"
        df_precision.to_csv(precision_file, index=False)
        print(f"    Fenêtre détaillée: {precision_file}")
        
        # Synthèse par agence/colonne
        precision_summary = df_precision.groupby(['agency_code', 'target_column']).agg({
            'error_pct': 'mean',
            'precision_pct': 'mean'
        }).round(2)
        
        precision_summary_file = f"{OUTPUT_FOLDER}/fenetre_prediction_synthese.csv"
        precision_summary.to_csv(precision_summary_file)
        print(f"    Fenêtre synthèse: {precision_summary_file}")
        
        # Fenêtres optimales
        identify_optimal_windows(df_precision)


def identify_optimal_windows(df_precision):
    """Identifier fenêtres optimales par agence/colonne"""
    print("\n[8] Fenêtres optimales...")
    
    optimal_windows = []
    thresholds = [90, 85, 80, 75, 70]
    
    for agency in df_precision['agency_code'].unique():
        for col in df_precision['target_column'].unique():
            df_subset = df_precision[
                (df_precision['agency_code'] == agency) & 
                (df_precision['target_column'] == col)
            ].sort_values('day')
            
            if df_subset.empty:
                continue
            
            for threshold in thresholds:
                days_valid = 0
                
                for _, row in df_subset.iterrows():
                    if row['precision_pct'] >= threshold:
                        days_valid += 1
                    else:
                        break
                
                optimal_windows.append({
                    'agency_code': agency,
                    'target_column': col,
                    'precision_threshold': threshold,
                    'optimal_window_days': days_valid,
                    'avg_error_pct': df_subset.head(days_valid)['error_pct'].mean() if days_valid > 0 else np.nan
                })
    
    if optimal_windows:
        df_optimal = pd.DataFrame(optimal_windows)
        
        optimal_file = f"{OUTPUT_FOLDER}/fenetres_optimales.csv"
        df_optimal.to_csv(optimal_file, index=False)
        print(f"    Fenêtres optimales: {optimal_file}")


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    process_all_agencies()