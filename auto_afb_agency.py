import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
import traceback

warnings.filterwarnings('ignore')


class EnsembleForecaster:
    """
    Modele ensembliste combinant 4 approches:
    1. XGBoost (ML sur features)
    2. Prophet (saisonnalite et tendance)
    3. ARIMA (dependances temporelles)
    4. Moyenne Mobile (baseline robuste)
    
    Strategie de combinaison:
    - Ponderation adaptative basee sur performance historique
    - Moyenne ponderee des 4 predictions
    """
    
    def __init__(self, component_name):
        """
        Initialiser l'ensembliste pour une composante
        
        Args:
            component_name: 'encaissements', 'decaissements' ou 'Besoin'
        """
        self.component_name = component_name
        self.models = {}
        self.weights = {}
        self.performance_history = []
        
        print(f"\n{'='*70}")
        print(f"INITIALISATION ENSEMBLISTE - {component_name.upper()}")
        print(f"{'='*70}")
    
    def _prepare_xgboost_features(self, df, component_name):
        """Creer features pour XGBoost"""
        df_feat = df[[component_name]].copy()
        
        for i in range(1, 8):
            df_feat[f'lag_{i}'] = df_feat[component_name].shift(i)
        
        df_feat['rolling_mean_7'] = df_feat[component_name].rolling(7).mean()
        df_feat['rolling_std_7'] = df_feat[component_name].rolling(7).std()
        df_feat['rolling_min_7'] = df_feat[component_name].rolling(7).min()
        df_feat['rolling_max_7'] = df_feat[component_name].rolling(7).max()
        
        df_feat['jour_semaine'] = df_feat.index.dayofweek
        df_feat['mois'] = df_feat.index.month
        df_feat['jour_mois'] = df_feat.index.day
        df_feat['trimestre'] = df_feat.index.quarter
        
        df_feat['debut_mois'] = (df_feat.index.day <= 5).astype(int)
        df_feat['fin_mois'] = (df_feat.index.day >= 25).astype(int)
        
        df_feat['diff_1'] = df_feat[component_name].diff(1)
        df_feat['diff_5'] = df_feat[component_name].diff(5)
        
        df_feat['mois_sin'] = np.sin(2 * np.pi * df_feat['mois'] / 12)
        df_feat['mois_cos'] = np.cos(2 * np.pi * df_feat['mois'] / 12)
        
        df_feat = df_feat.dropna()
        return df_feat
    
    def _train_xgboost(self, train_data, test_data):
        """Entrainer XGBoost"""
        print(f"\n[1/4] Entrainement XGBoost...")
        
        train_feat = self._prepare_xgboost_features(train_data, self.component_name)
        test_feat = self._prepare_xgboost_features(test_data, self.component_name)
        
        X_train = train_feat.drop(self.component_name, axis=1)
        y_train = train_feat[self.component_name]
        X_test = test_feat.drop(self.component_name, axis=1)
        y_test = test_feat[self.component_name]
        
        model = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            reg_alpha=0.5,
            reg_lambda=2.0,
            subsample=0.8,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=30
        )
        
        model.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)], 
                 verbose=False)
        
        pred_test = model.predict(X_test)
        mape_test = np.mean(np.abs((y_test - pred_test) / np.abs(y_test))) * 100
        
        print(f"  XGBoost MAPE Test: {mape_test:.2f}%")
        
        self.models['xgboost'] = {
            'model': model,
            'feature_names': list(X_train.columns),
            'mape_test': mape_test
        }
    
    def _forecast_xgboost(self, last_known_data, horizon):
        """Forecasting XGBoost recursif"""
        model = self.models['xgboost']['model']
        feature_names = self.models['xgboost']['feature_names']
        
        base_values = last_known_data[self.component_name].tail(15).values.tolist()
        predictions = []
        current_date = last_known_data.index[-1]
        current_history = base_values.copy()
        
        for day in range(horizon):
            next_date = current_date + pd.Timedelta(days=1)
            while next_date.dayofweek >= 5:
                next_date += pd.Timedelta(days=1)
            
            next_row = pd.DataFrame(index=[next_date])
            
            next_row['jour_semaine'] = next_date.dayofweek
            next_row['mois'] = next_date.month
            next_row['jour_mois'] = next_date.day
            next_row['trimestre'] = next_date.quarter
            next_row['debut_mois'] = int(next_date.day <= 5)
            next_row['fin_mois'] = int(next_date.day >= 25)
            next_row['mois_sin'] = np.sin(2 * np.pi * next_date.month / 12)
            next_row['mois_cos'] = np.cos(2 * np.pi * next_date.month / 12)
            
            for i in range(1, 8):
                next_row[f'lag_{i}'] = current_history[-i] if len(current_history) >= i else current_history[-1]
            
            window = current_history[-7:] if len(current_history) >= 7 else current_history
            next_row['rolling_mean_7'] = np.mean(window)
            next_row['rolling_std_7'] = np.std(window) if len(window) > 1 else 0
            next_row['rolling_min_7'] = np.min(window)
            next_row['rolling_max_7'] = np.max(window)
            
            next_row['diff_1'] = current_history[-1] - current_history[-2] if len(current_history) >= 2 else 0
            next_row['diff_5'] = current_history[-1] - current_history[-6] if len(current_history) >= 6 else 0
            
            for feat in feature_names:
                if feat not in next_row.columns:
                    next_row[feat] = 0
            
            next_row = next_row[feature_names]
            
            pred = model.predict(next_row)[0]
            predictions.append(pred)
            current_history.append(pred)
            current_date = next_date
        
        return np.array(predictions)
    
    def _train_prophet(self, train_data, test_data):
        """Entrainer Prophet"""
        print(f"\n[2/4] Entrainement Prophet...")
        
        df_prophet = train_data[[self.component_name]].reset_index()
        df_prophet.columns = ['ds', 'y']
        
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(df_prophet)
        
        df_test = test_data[[self.component_name]].reset_index()
        df_test.columns = ['ds', 'y']
        
        forecast = model.predict(df_test[['ds']])
        pred_test = forecast['yhat'].values
        
        mape_test = np.mean(np.abs((df_test['y'] - pred_test) / np.abs(df_test['y']))) * 100
        
        print(f"  Prophet MAPE Test: {mape_test:.2f}%")
        
        self.models['prophet'] = {
            'model': model,
            'mape_test': mape_test
        }
    
    def _forecast_prophet(self, last_known_data, horizon):
        """Forecasting Prophet"""
        model = self.models['prophet']['model']
        
        last_date = last_known_data.index[-1]
        future_dates = []
        current_date = last_date
        
        for _ in range(horizon):
            current_date += pd.Timedelta(days=1)
            while current_date.dayofweek >= 5:
                current_date += pd.Timedelta(days=1)
            future_dates.append(current_date)
        
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        
        return forecast['yhat'].values
    
    def _train_arima(self, train_data, test_data):
        """Entrainer ARIMA"""
        print(f"\n[3/4] Entrainement ARIMA...")
        
        try:
            model = ARIMA(
                train_data[self.component_name],
                order=(5, 1, 2),
                seasonal_order=(1, 0, 1, 5)
            )
            
            fitted = model.fit()
            
            pred_test = fitted.forecast(steps=len(test_data))
            y_test = test_data[self.component_name].values
            
            mape_test = np.mean(np.abs((y_test - pred_test) / np.abs(y_test))) * 100
            
            print(f"  ARIMA MAPE Test: {mape_test:.2f}%")
            
            self.models['arima'] = {
                'model': fitted,
                'mape_test': mape_test
            }
            
        except Exception as e:
            print(f"  ERREUR ARIMA: {str(e)}")
            print(f"  Utilisation ARIMA simplifie (1,1,1)")
            
            model = ARIMA(train_data[self.component_name], order=(1, 1, 1))
            fitted = model.fit()
            
            pred_test = fitted.forecast(steps=len(test_data))
            y_test = test_data[self.component_name].values
            mape_test = np.mean(np.abs((y_test - pred_test) / np.abs(y_test))) * 100
            
            print(f"  ARIMA simplifie MAPE Test: {mape_test:.2f}%")
            
            self.models['arima'] = {
                'model': fitted,
                'mape_test': mape_test
            }
    
    def _forecast_arima(self, last_known_data, horizon):
        """Forecasting ARIMA"""
        model = self.models['arima']['model']
        
        predictions = model.forecast(steps=horizon)
        
        return predictions.values if hasattr(predictions, 'values') else predictions
    
    def _train_moving_average(self, train_data, test_data):
        """Entrainer Moyenne Mobile (baseline)"""
        print(f"\n[4/4] Entrainement Moyenne Mobile...")
        
        window = 7
        
        predictions = []
        for i in range(len(test_data)):
            if i == 0:
                recent = train_data[self.component_name].tail(window).mean()
            else:
                recent_values = test_data[self.component_name].iloc[:i].tail(window).values
                if len(recent_values) < window:
                    padding = train_data[self.component_name].tail(window - len(recent_values)).values
                    recent_values = np.concatenate([padding, recent_values])
                recent = np.mean(recent_values)
            
            predictions.append(recent)
        
        y_test = test_data[self.component_name].values
        mape_test = np.mean(np.abs((y_test - predictions) / np.abs(y_test))) * 100
        
        print(f"  Moyenne Mobile MAPE Test: {mape_test:.2f}%")
        
        self.models['moving_avg'] = {
            'window': window,
            'mape_test': mape_test
        }
    
    def _forecast_moving_average(self, last_known_data, horizon):
        """Forecasting Moyenne Mobile"""
        window = self.models['moving_avg']['window']
        
        predictions = []
        history = last_known_data[self.component_name].tail(window).values.tolist()
        
        for _ in range(horizon):
            pred = np.mean(history[-window:])
            predictions.append(pred)
            history.append(pred)
        
        return np.array(predictions)
    
    def train(self, train_data, test_data):
        """
        Entrainer tous les modeles et calculer poids optimaux
        
        Strategie de ponderation:
        1. Inverse du MAPE (meilleure performance = plus de poids)
        2. Normalisation pour somme = 1
        3. Seuil minimum pour eviter poids nuls
        """
        print(f"\n{'='*70}")
        print(f"ENTRAINEMENT ENSEMBLE - {self.component_name.upper()}")
        print(f"{'='*70}")
        
        self._train_xgboost(train_data, test_data)
        self._train_prophet(train_data, test_data)
        self._train_arima(train_data, test_data)
        self._train_moving_average(train_data, test_data)
        
        print(f"\n{'='*70}")
        print(f"CALCUL POIDS ENSEMBLISTE")
        print(f"{'='*70}")
        
        mapes = {}
        for name, model_info in self.models.items():
            mapes[name] = model_info['mape_test']
        
        inverse_mapes = {name: 1.0 / mape for name, mape in mapes.items()}
        total = sum(inverse_mapes.values())
        
        self.weights = {name: inv / total for name, inv in inverse_mapes.items()}
        
        print(f"\nPoids calcules (bases sur performance test):")
        for name, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            print(f"  {name:15s} : {weight:.4f} (MAPE: {mapes[name]:.2f}%)")
        
        best_model = min(mapes.items(), key=lambda x: x[1])
        print(f"\nMeilleur modele individuel: {best_model[0]} ({best_model[1]:.2f}% MAPE)")
    
    def forecast(self, last_known_data, horizon):
        """
        Forecasting ensembliste avec moyenne ponderee
        
        Args:
            last_known_data: Donnees historiques connues
            horizon: Nombre de jours a predire
            
        Returns:
            dict avec predictions de chaque modele et ensemble
        """
        print(f"\n{'='*70}")
        print(f"FORECASTING ENSEMBLISTE - {self.component_name.upper()}")
        print(f"{'='*70}")
        
        predictions = {}
        
        print("\nGeneration predictions individuelles...")
        predictions['xgboost'] = self._forecast_xgboost(last_known_data, horizon)
        print("  XGBoost: OK")
        
        predictions['prophet'] = self._forecast_prophet(last_known_data, horizon)
        print("  Prophet: OK")
        
        predictions['arima'] = self._forecast_arima(last_known_data, horizon)
        print("  ARIMA: OK")
        
        predictions['moving_avg'] = self._forecast_moving_average(last_known_data, horizon)
        print("  Moyenne Mobile: OK")
        
        ensemble_pred = np.zeros(horizon)
        
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        predictions['ensemble'] = ensemble_pred
        
        print(f"\nEnsemble cree avec succes")
        
        return predictions
    
    def evaluate(self, predictions, actual_values):
        """
        Evaluer toutes les predictions (individuelles + ensemble)
        
        Args:
            predictions: Dict des predictions de chaque modele
            actual_values: Vraies valeurs
            
        Returns:
            dict avec metriques pour chaque modele
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION ENSEMBLE - {self.component_name.upper()}")
        print(f"{'='*70}")
        
        results = {}
        
        for model_name, pred in predictions.items():
            common_len = min(len(pred), len(actual_values))
            pred_subset = pred[:common_len]
            actual_subset = actual_values[:common_len]
            
            mae = mean_absolute_error(actual_subset, pred_subset)
            rmse = np.sqrt(mean_squared_error(actual_subset, pred_subset))
            mape = np.mean(np.abs((actual_subset - pred_subset) / np.abs(actual_subset))) * 100
            r2 = r2_score(actual_subset, pred_subset)
            
            results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
            
            print(f"\n{model_name.upper()}:")
            print(f"  MAE:  {mae:,.0f}")
            print(f"  RMSE: {rmse:,.0f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R2:   {r2:.4f}")
        
        best = min(results.items(), key=lambda x: x[1]['mape'])
        print(f"\n{'='*70}")
        print(f"MEILLEUR: {best[0].upper()} avec {best[1]['mape']:.2f}% MAPE")
        print(f"{'='*70}")
        
        return results


def plot_ensemble_results(predictions, actual_values, component_name, code_agence, output_dir):
    """Visualiser resultats ensemblistes avec 2 graphiques"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    days = range(len(actual_values))
    
    colors = {
        'xgboost': 'blue',
        'prophet': 'green',
        'arima': 'orange',
        'moving_avg': 'purple',
        'ensemble': 'red'
    }
    
    ax1 = axes[0]
    
    ax1.plot(days, actual_values, 'o-', label='Reel', 
             linewidth=3, markersize=8, color='black', zorder=10)
    
    for model_name in ['xgboost', 'prophet', 'arima', 'moving_avg', 'ensemble']:
        pred = predictions[model_name][:len(actual_values)]
        linestyle = '-' if model_name == 'ensemble' else '--'
        linewidth = 2.5 if model_name == 'ensemble' else 1.5
        ax1.plot(days, pred, linestyle, label=model_name.upper(), 
                alpha=0.7, color=colors[model_name], linewidth=linewidth)
    
    ax1.set_title(f'Comparaison Tous les Modeles - {component_name} - Agence {code_agence}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Jour')
    ax1.set_ylabel('Montant')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    
    mapes = {name: np.mean(np.abs((actual_values - predictions[name][:len(actual_values)]) / 
                                   np.abs(actual_values))) * 100 
             for name in predictions.keys()}
    best_model_name = min(mapes.items(), key=lambda x: x[1])[0]
    
    ax2.plot(days, actual_values, 'o-', label='Reel', 
             linewidth=3, markersize=8, color='black')
    ax2.plot(days, predictions[best_model_name][:len(actual_values)], 's-', 
             label=f'Meilleur: {best_model_name.upper()} ({mapes[best_model_name]:.2f}% MAPE)', 
             linewidth=3, markersize=8, color=colors[best_model_name])
    
    ax2.fill_between(days,
                     predictions[best_model_name][:len(actual_values)] * 0.9,
                     predictions[best_model_name][:len(actual_values)] * 1.1,
                     alpha=0.2, color=colors[best_model_name], label='Intervalle +/-10%')
    
    ax2.set_title(f'Meilleur Modele vs Reel - {component_name} - Agence {code_agence}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Jour')
    ax2.set_ylabel('Montant')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'Agence_{code_agence}_{component_name}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_agency(file_path, output_dir, horizon=5):
    """Traiter une seule agence - reproduction exacte du code original"""
    
    filename = os.path.basename(file_path)
    code_agence = filename.replace('.csv', '').replace('Agence_', '')
    
    print(f"\n{'='*80}")
    print(f"TRAITEMENT AGENCE {code_agence}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(file_path, parse_dates=['Date Opération'])
        df = df.sort_values('Date Opération').set_index('Date Opération')
        
        print(f"Donnees: {len(df)} observations")
        print(f"Periode: {df.index.min().date()} a {df.index.max().date()}")
        
        if len(df) < 50:
            print(f"ATTENTION: Donnees insuffisantes ({len(df)} lignes)")
            return None
        
        train = df[df.index <= '2025-02-28']
        test = df[(df.index > '2025-02-28') & (df.index <= '2025-06-30')]
        validate = df[df.index > '2025-06-30']
        
        print(f"Train: {len(train)} | Test: {len(test)} | Validate: {len(validate)}")
        
        horizon_actual = min(horizon, len(validate))
        
        if horizon_actual == 0:
            print(f"ERREUR: Pas de donnees de validation")
            return None
        
        results_list = []
        
        for component in ['encaissements', 'decaissements', 'Besoin']:
            print(f"\n{'#'*80}")
            print(f"TRAITEMENT {component.upper()}")
            print(f"{'#'*80}")
            
            ensemble = EnsembleForecaster(component)
            ensemble.train(train, test)
            
            last_known = df[df.index <= '2025-06-30']
            predictions = ensemble.forecast(last_known, horizon_actual)
            actual_values = validate[component].head(horizon_actual).values
            
            evaluation = ensemble.evaluate(predictions, actual_values)
            
            plot_ensemble_results(predictions, actual_values, component, 
                                code_agence, output_dir)
            
            for model_name in ['xgboost', 'prophet', 'arima', 'moving_avg', 'ensemble']:
                result_row = {
                    'Code_Agence': code_agence,
                    'Composante': component,
                    'Modele': model_name,
                    'MAE': evaluation[model_name]['mae'],
                    'RMSE': evaluation[model_name]['rmse'],
                    'MAPE': evaluation[model_name]['mape'],
                    'R2': evaluation[model_name]['r2'],
                    'Poids': ensemble.weights.get(model_name, 1.0) if model_name != 'ensemble' else 1.0,
                    'Nb_Jours_Train': len(train),
                    'Nb_Jours_Test': len(test),
                    'Nb_Jours_Validation': horizon_actual,
                    'Statut': 'OK'
                }
                results_list.append(result_row)
        
        print(f"\n{'='*80}")
        print(f"AGENCE {code_agence}: TERMINE AVEC SUCCES")
        print(f"{'='*80}")
        
        return results_list
        
    except Exception as e:
        print(f"\nERREUR lors du traitement de l'agence {code_agence}:")
        print(traceback.format_exc())
        
        error_row = {
            'Code_Agence': code_agence,
            'Composante': 'N/A',
            'Modele': 'N/A',
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan,
            'Poids': np.nan,
            'Nb_Jours_Train': np.nan,
            'Nb_Jours_Test': np.nan,
            'Nb_Jours_Validation': np.nan,
            'Statut': f'ERREUR: {str(e)[:100]}'
        }
        return [error_row]


def main_multi_agency_analysis(input_dir='agences_split/', 
                                output_dir='resultats/', 
                                horizon=5):
    """
    Pipeline complet pour toutes les agences
    """
    
    print("\n" + "="*80)
    print("ANALYSE ENSEMBLISTE MULTI-AGENCES")
    print("ENCAISSEMENTS, DECAISSEMENTS, BESOIN")
    print("="*80)
    print(f"Repertoire entree: {input_dir}")
    print(f"Repertoire sortie: {output_dir}")
    print(f"Horizon prediction: {horizon} jours")
    
    os.makedirs(output_dir, exist_ok=True)
    graphiques_dir = os.path.join(output_dir, 'graphiques')
    os.makedirs(graphiques_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    print(f"\nNombre de fichiers detectes: {len(csv_files)}")
    
    log_file = os.path.join(output_dir, 'logs_execution.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Execution demarree: {datetime.now()}\n")
        f.write(f"Nombre d'agences: {len(csv_files)}\n\n")
    
    all_results = []
    
    for idx, csv_file in enumerate(csv_files, 1):
        file_path = os.path.join(input_dir, csv_file)
        
        print(f"\n[{idx}/{len(csv_files)}] Traitement: {csv_file}")
        
        agency_results = process_single_agency(file_path, graphiques_dir, horizon)
        
        if agency_results:
            all_results.extend(agency_results)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{idx}/{len(csv_files)}] {csv_file}: ")
            if agency_results and agency_results[0]['Statut'] == 'OK':
                f.write("OK\n")
            else:
                f.write(f"ERREUR\n")
    
    df_results = pd.DataFrame(all_results)
    
    synthese_file = os.path.join(output_dir, 'synthese_globale.csv')
    df_results.to_csv(synthese_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("TRAITEMENT TERMINE")
    print("="*80)
    print(f"Resultats sauvegardes dans: {synthese_file}")
    print(f"Graphiques sauvegardes dans: {graphiques_dir}")
    print(f"Logs disponibles dans: {log_file}")
    
    nb_success = len([r for r in all_results if r['Statut'] == 'OK'])
    nb_errors = len([r for r in all_results if r['Statut'] != 'OK'])
    
    print(f"\nBilan: {nb_success} succes, {nb_errors} erreurs")
    
    if nb_success > 0:
        print("\nApercu des resultats (premieres lignes):")
        print(df_results.head(10))
        
        print("\n" + "="*80)
        print("RESUME STATISTIQUE PAR AGENCE")
        print("="*80)
        
        df_ok = df_results[df_results['Statut'] == 'OK']
        if len(df_ok) > 0:
            summary = df_ok.groupby(['Code_Agence', 'Composante']).agg({
                'MAPE': 'mean',
                'MAE': 'mean',
                'RMSE': 'mean',
                'R2': 'mean'
            }).round(2)
            print(summary)
    
    return df_results


if __name__ == "__main__":
    df_synthesis = main_multi_agency_analysis(
        input_dir='agences_split/',
        output_dir='resultats/',
        horizon=5
    )