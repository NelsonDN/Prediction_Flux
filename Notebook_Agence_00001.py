# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Chargement des données
df = pd.read_csv('Agence_00001.csv', parse_dates=['Date Opération'])
df = df[['Date Opération', 'Besoin']].set_index('Date Opération').sort_index()

# Vérification des données manquantes
print(f"Données manquantes: {df.isnull().sum().sum()}")
print(f"Plage de dates: {df.index.min()} to {df.index.max()}")
print(f"Nombre total de jours: {len(df)}")

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
df = pd.read_csv('Agence_00001.csv', parse_dates=['Date Opération'])
df = df.sort_values('Date Opération').set_index('Date Opération')


# Ajouter jour de la semaine (pour graphe 5)
df['jour_semaine'] = df.index.day_name()
jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
df['jour_semaine'] = pd.Categorical(df['jour_semaine'], categories=jours_ordre, ordered=True)


# %%

# 1. Graphe : Besoin vs Encaissements
plt.figure(figsize=(8, 5))
plt.scatter(df['encaissements'], df['Besoin'], alpha=0.6, color='steelblue', edgecolor='k', linewidth=0.5)
plt.title("1. Corrélation : Besoin vs Encaissements", fontsize=12, fontweight='bold')
plt.xlabel("Encaissements")
plt.ylabel("Besoin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%

# 2. Graphe : Besoin vs Décaissements
plt.figure(figsize=(8, 5))
plt.scatter(df['decaissements'], df['Besoin'], alpha=0.6, color='indianred', edgecolor='k', linewidth=0.5)
plt.title("2. Corrélation : Besoin vs Décaissements", fontsize=12, fontweight='bold')
plt.xlabel("Décaissements")
plt.ylabel("Besoin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%

# 3. Graphe : Encaissements vs Décaissements
plt.figure(figsize=(8, 5))
plt.scatter(df['encaissements'], df['decaissements'], alpha=0.6, color='forestgreen', edgecolor='k', linewidth=0.5)
plt.title("3. Corrélation : Encaissements vs Décaissements", fontsize=12, fontweight='bold')
plt.xlabel("Encaissements")
plt.ylabel("Décaissements")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%

# 4. Graphe : Besoin(t) vs Besoin(t-1) — Auto-corrélation lag=1
df_lag = df.copy()
df_lag['Besoin_lag1'] = df_lag['Besoin'].shift(1)
df_lag = df_lag.dropna(subset=['Besoin_lag1'])

plt.figure(figsize=(8, 5))
plt.scatter(df_lag['Besoin_lag1'], df_lag['Besoin'], alpha=0.6, color='goldenrod', edgecolor='k', linewidth=0.5)
plt.title("4. Auto-corrélation : Besoin(t) vs Besoin(t-1)", fontsize=12, fontweight='bold')
plt.xlabel("Besoin (Jour précédent)")
plt.ylabel("Besoin (Jour courant)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%

# 5. Graphe : Besoin moyen par jour de la semaine
besoin_jour = df.groupby('jour_semaine')['Besoin'].mean()
jours_labels = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi']

plt.figure(figsize=(8, 5))
plt.bar(jours_labels, besoin_jour, color='slateblue', edgecolor='k', linewidth=0.8)
plt.title("5. Besoin moyen par jour de la semaine", fontsize=12, fontweight='bold')
plt.xlabel("Jour de la semaine")
plt.ylabel("Besoin moyen")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# VISUALISATION SERIE TEMPORELLE (TRAIN)

# %%
import matplotlib.pyplot as plt

train_end = '2025-02-28'
train_data = df[df.index <= train_end]['Besoin']
test_data = df[df.index > train_end]['Besoin'].head(7)  # 7 premiers jours de mars 2025

plt.figure(figsize=(14, 5))
plt.plot(train_data.index.to_numpy(), train_data.values, label='Besoin (train)', color='blue')
plt.title("Série temporelle du Besoin (Sept 2022 - Fév 2025)")
plt.xlabel("Date")
plt.ylabel("Besoin (Flux net client)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# TEST ADF

# %%
from statsmodels.tsa.stattools import adfuller

result = adfuller(train_data.dropna())
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

# %% [markdown]
# DECOMPOSITION ADDITIVE

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(train_data.dropna(), model='additive', period=5, extrapolate_trend='freq')
fig = decomp.plot()
plt.show()

# %% [markdown]
# CORRELELOGRAMME SIMPLE ET PARTIEL (ACF/PACF)

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train_data.dropna(), lags=30, ax=ax[0])
plot_pacf(train_data.dropna(), lags=30, ax=ax[1], method='ywm')
plt.show()

# %% [markdown]
# CELLULE POUR MODELE PROPHET

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Configuration des avertissements et logs
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def load_and_prepare_data(file_path):
    """Charger et préparer les données pour Prophet"""
    try:
        # Charger le fichier CSV
        df = pd.read_csv(file_path, parse_dates=['Date Opération'])
        
        # Trier par date et extraire la colonne cible
        df = df.sort_values('Date Opération')
        
        # Préparer pour Prophet (colonnes 'ds' et 'y')
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(df['Date Opération']),
            'y': pd.to_numeric(df['Besoin'], errors='coerce')
        })
        
        # Supprimer les valeurs manquantes
        df_prophet = df_prophet.dropna().reset_index(drop=True)
        
        # Supprimer timezone proprement
        if hasattr(df_prophet['ds'].dtype, 'tz') and df_prophet['ds'].dtype.tz is not None:
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        
        # Assurer que les types sont corrects
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = df_prophet['y'].astype(float)
        
        print(f"Données chargées : {len(df_prophet)} observations")
        print(f"Période complète : du {df_prophet['ds'].min().date()} au {df_prophet['ds'].max().date()}")
        
        return df_prophet
        
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        raise

def split_train_test(df_prophet, train_end_date='2025-02-28', test_days=7):
    """Diviser les données en jeu d'entraînement et de test"""
    train_end = pd.to_datetime(train_end_date)
    
    # Données d'entraînement
    train_data = df_prophet[df_prophet['ds'] <= train_end].copy()
    
    # Données de test (premiers jours après train_end_date)
    test_data = df_prophet[df_prophet['ds'] > train_end].head(test_days).copy()
    
    print(f"Taille jeu d'entraînement : {len(train_data)} observations")
    print(f"Taille jeu de test : {len(test_data)} observations")
    
    if len(test_data) < test_days:
        print(f"Attention : Seulement {len(test_data)} jours disponibles pour le test au lieu de {test_days}")
    
    if len(test_data) > 0:
        print(f"Période test : du {test_data['ds'].min().date()} au {test_data['ds'].max().date()}")
    
    return train_data, test_data















def create_and_train_model(train_data):
    """Créer et entraîner le modèle Prophet optimisé pour données bancaires"""
    print("\nInitialisation du modèle Prophet...")
    
    # Configuration spécifique aux flux bancaires
    model = Prophet(
        seasonality_mode='additive',  # Additif pour flux bancaires
        weekly_seasonality=True,
        yearly_seasonality=False,  # Désactivé car seulement 3 ans
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # Réduit pour stabilité
        seasonality_prior_scale=15,    # Augmenté pour patterns bancaires
        holidays_prior_scale=10,
        interval_width=0.95,  # IC 95%
        growth='linear'
    )
    
    # Saisonnalités bancaires
    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    model.add_seasonality(name='biweekly', period=14, fourier_order=5)
    
    # IMPORTANT: Créer une copie pour ne pas modifier l'original
    train_data_copy = train_data.copy()
    
    # Ajouter les regressors bancaires
    train_data_copy['is_month_start'] = (train_data_copy['ds'].dt.day <= 5).astype(float)
    train_data_copy['is_month_end'] = (train_data_copy['ds'].dt.day >= 25).astype(float)
    train_data_copy['is_mid_month'] = ((train_data_copy['ds'].dt.day >= 14) & 
                                       (train_data_copy['ds'].dt.day <= 16)).astype(float)
    
    # Ajouter les regressors au modèle
    model.add_regressor('is_month_start')
    model.add_regressor('is_month_end')
    model.add_regressor('is_mid_month')
    
    print("Entraînement du modèle...")
    model.fit(train_data_copy)
    print("Entraînement terminé avec succès")
    
    return model

def make_predictions(model, train_data, test_data):
    """Générer les prédictions avec alignement correct des dates"""
    print("\nGénération des prédictions...")
    
    # Fonction helper pour ajouter les regressors
    def add_regressor_columns(df):
        df_copy = df.copy()
        df_copy['is_month_start'] = (df_copy['ds'].dt.day <= 5).astype(float)
        df_copy['is_month_end'] = (df_copy['ds'].dt.day >= 25).astype(float)
        df_copy['is_mid_month'] = ((df_copy['ds'].dt.day >= 14) & 
                                   (df_copy['ds'].dt.day <= 16)).astype(float)
        return df_copy
    
    if len(test_data) > 0:
        # Créer future avec dates exactes de test
        future_train = pd.DataFrame({'ds': train_data['ds']})
        future_test = pd.DataFrame({'ds': test_data['ds']})
        future = pd.concat([future_train, future_test], ignore_index=True)
        
        # CRUCIAL: Ajouter les colonnes regressors au future dataframe
        future = add_regressor_columns(future)
        
    else:
        # Si pas de données test, créer 7 jours ouvrables après
        last_train_date = train_data['ds'].max()
        # Générer dates ouvrables uniquement (exclure weekends)
        future_dates = pd.bdate_range(start=last_train_date + pd.Timedelta(days=1), 
                                      periods=7)
        future_train = pd.DataFrame({'ds': train_data['ds']})
        future_test = pd.DataFrame({'ds': future_dates})
        future = pd.concat([future_train, future_test], ignore_index=True)
        
        # Ajouter les regressors
        future = add_regressor_columns(future)
    
    # Générer les prédictions
    forecast = model.predict(future)
    
    # Extraire les prédictions pour la période de test
    if len(test_data) > 0:
        # S'assurer de l'alignement parfait avec les dates de test
        forecast_test = forecast[forecast['ds'].isin(test_data['ds'])].copy()
        forecast_test = forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
        
        # Vérification de l'alignement
        print(f"\nVérification alignement des dates:")
        print(f"Dates test attendues : {test_data['ds'].dt.date.values}")
        print(f"Dates prédictions    : {forecast_test['ds'].dt.date.values}")
        
        if len(forecast_test) != len(test_data):
            print(f"ATTENTION: Nombre de prédictions ({len(forecast_test)}) != données test ({len(test_data)})")
        
        print("\nPrédictions pour la période de test :")
        for i, row in forecast_test.iterrows():
            print(f"{row['ds'].date()} : {row['yhat']:,.0f}")
            
    else:
        forecast_test = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        print("\nPrédictions futures (7 prochains jours ouvrables) :")
        for i, row in forecast_test.iterrows():
            print(f"{row['ds'].date()} : {row['yhat']:,.0f}")
    
    return forecast, forecast_test

def evaluate_model(test_data, forecast_test):
    """Évaluer les performances du modèle"""
    if len(test_data) == 0:
        print("Pas de données de test disponibles pour l'évaluation")
        return None, None, None
    
    print(f"\nEVALUATION DU MODELE :")
    print("="*50)
    
    # Aligner les prédictions avec les vraies valeurs
    y_true = test_data['y'].values
    y_pred = forecast_test['yhat'].values
    
    # Calculer les métriques
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE avec gestion des valeurs nulles
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    print(f"MAE  (Erreur absolue moyenne)      : {mae:,.0f}")
    print(f"RMSE (Racine erreur quadratique)   : {rmse:,.0f}")
    print(f"MAPE (Erreur absolue % moyenne)    : {mape:.2f}%")
    
    return mae, rmse, mape

def create_comprehensive_visualizations(model, forecast, test_data, forecast_test):
    """Créer toutes les visualisations nécessaires"""
    print("\nCréation des graphiques...")
    
    # Configuration matplotlib
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 10
    
    # Graphique 1 : Comparaison prédictions vs réalité
    if len(test_data) > 0:
        plt.figure(figsize=(14, 8))
        
        # Conversion sécurisée
        test_dates = pd.to_datetime(test_data['ds']).dt.tz_localize(None)
        test_values = test_data['y'].astype(float)
        forecast_dates = pd.to_datetime(forecast_test['ds']).dt.tz_localize(None) 
        forecast_values = forecast_test['yhat'].astype(float)
        forecast_lower = forecast_test['yhat_lower'].astype(float)
        forecast_upper = forecast_test['yhat_upper'].astype(float)
        
        # Graphique principal
        plt.plot(test_dates.values, test_values.values, 'o-', 
                color='#2E86AB', linewidth=2.5, label='Valeurs réelles', markersize=8)
        
        plt.plot(forecast_dates.values, forecast_values.values, 's-', 
                color='#A23B72', linewidth=2.5, label='Prédictions Prophet', markersize=6)
        
        plt.fill_between(forecast_dates.values, forecast_lower.values, forecast_upper.values, 
                        color='#A23B72', alpha=0.2, label='Intervalle de confiance (80%)')
        
        plt.title("Comparaison : Prédictions Prophet vs Valeurs Réelles", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Besoin (Flux Net Client)", fontsize=12)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Graphique 2 : Vue d'ensemble historique et prévisions
    plt.figure(figsize=(16, 10))
    
    try:
        # Données historiques
        if hasattr(model, 'history') and len(model.history) > 0:
            hist_dates = pd.to_datetime(model.history['ds']).dt.tz_localize(None)
            hist_values = model.history['y'].astype(float)
            plt.plot(hist_dates.values, hist_values.values, '.', 
                    color='#404040', markersize=1.5, label='Données historiques', alpha=0.8)
        
        # Prédictions complètes
        forecast_dates_all = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
        forecast_values_all = forecast['yhat'].astype(float)
        forecast_lower_all = forecast['yhat_lower'].astype(float)
        forecast_upper_all = forecast['yhat_upper'].astype(float)
        
        plt.plot(forecast_dates_all.values, forecast_values_all.values, 
                color='#0072B2', linewidth=2, label='Prédictions Prophet')
        
        plt.fill_between(forecast_dates_all.values, 
                        forecast_lower_all.values, 
                        forecast_upper_all.values,
                        alpha=0.15, color='#0072B2', label='Intervalle de confiance')
        
        # Période de test en évidence
        if len(test_data) > 0:
            test_dates = pd.to_datetime(test_data['ds']).dt.tz_localize(None)
            test_values = test_data['y'].astype(float)
            plt.plot(test_dates.values, test_values.values, 'o', 
                    color='#D55E00', markersize=6, label='Période de test')
        
        plt.title("Vue d'ensemble : Données Historiques et Prévisions Prophet", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Valeur", fontsize=12)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur graphique d'ensemble : {e}")
    
    # Graphique 3 : Analyse des résidus
    if len(test_data) > 0:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            residuals = test_data['y'].astype(float) - forecast_test['yhat'].astype(float)
            dates_test = pd.to_datetime(test_data['ds']).dt.tz_localize(None)
            
            # Résidus dans le temps
            ax1.plot(dates_test.values, residuals.values, 'o-', color='#CC79A7', linewidth=2, markersize=6)
            ax1.axhline(y=0, color='#E69F00', linestyle='--', alpha=0.8, linewidth=1.5)
            ax1.set_title("Évolution des Résidus", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Date", fontsize=12)
            ax1.set_ylabel("Erreur de Prédiction", fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Distribution des résidus
            ax2.hist(residuals.values, bins=max(3, len(residuals)//2), 
                    color='#56B4E9', alpha=0.7, edgecolor='black', linewidth=1)
            ax2.axvline(x=0, color='#E69F00', linestyle='--', alpha=0.8, linewidth=1.5)
            ax2.set_title("Distribution des Erreurs", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Erreur de Prédiction", fontsize=12)
            ax2.set_ylabel("Fréquence", fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Erreur analyse résidus : {e}")
    
    # Graphique 4 : Métriques de performance (si données de test disponibles)
    if len(test_data) > 0:
        try:
            mae = mean_absolute_error(test_data['y'].values, forecast_test['yhat'].values)
            rmse = np.sqrt(mean_squared_error(test_data['y'].values, forecast_test['yhat'].values))
            mape = np.mean(np.abs((test_data['y'].values - forecast_test['yhat'].values) / 
                                np.where(test_data['y'].values != 0, test_data['y'].values, 1))) * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['MAE', 'RMSE', 'MAPE (%)']
            values = [mae, rmse, mape]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:,.0f}' if value > 1 else f'{value:.2f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_title("Métriques de Performance du Modèle", fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel("Valeur", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Erreur graphique métriques : {e}")

def save_results(forecast_test, mae=None, rmse=None, mape=None):
    """Sauvegarder les résultats"""
    print("\nSauvegarde des résultats...")
    
    # Sauvegarder les prédictions
    forecast_test.to_csv('predictions_prophet_professional.csv', index=False)
    print("Prédictions sauvegardées dans 'predictions_prophet_professional.csv'")
    
    # Sauvegarder les métriques si disponibles
    if mae is not None:
        with open('performance_metrics_prophet.txt', 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE PERFORMANCE - MODELE PROPHET\n")
            f.write("="*50 + "\n\n")
            f.write("METRIQUES DE PERFORMANCE:\n")
            f.write("-" * 25 + "\n")
            f.write(f"MAE  (Mean Absolute Error)     : {mae:,.0f}\n")
            f.write(f"RMSE (Root Mean Square Error)  : {rmse:,.0f}\n")
            f.write(f"MAPE (Mean Absolute Percentage): {mape:.2f}%\n\n")
            f.write("INTERPRETATION DES METRIQUES:\n")
            f.write("-" * 30 + "\n")
            f.write("MAE  : Erreur moyenne absolue en unités originales\n")
            f.write("RMSE : Penalise davantage les grandes erreurs\n")
            f.write("MAPE : Erreur relative en pourcentage\n\n")
            f.write("QUALITE DU MODELE:\n")
            f.write("-" * 18 + "\n")
            if mape <= 10:
                f.write("Excellent : MAPE <= 10%\n")
            elif mape <= 20:
                f.write("Bon : 10% < MAPE <= 20%\n")
            elif mape <= 50:
                f.write("Acceptable : 20% < MAPE <= 50%\n")
            else:
                f.write("Amélioration nécessaire : MAPE > 50%\n")
        
        print("Métriques sauvegardées dans 'performance_metrics_prophet.txt'")

def main():
    """Fonction principale d'exécution"""
    try:
        print("ANALYSE PROPHET - VERSION PROFESSIONNELLE")
        print("="*60)
        
        # 1. Chargement et préparation des données
        df_prophet = load_and_prepare_data('Agence_00001.csv')
        
        # 2. Division train/test
        train_data, test_data = split_train_test(df_prophet)
        
        # 3. Création et entraînement du modèle
        model = create_and_train_model(train_data)
        
        # 4. Génération des prédictions
        forecast, forecast_test = make_predictions(model, train_data, test_data)
        
        # 5. Évaluation du modèle
        mae, rmse, mape = evaluate_model(test_data, forecast_test)
        
        # 6. Visualisations complètes
        create_comprehensive_visualizations(model, forecast, test_data, forecast_test)
        
        # 7. Sauvegarde des résultats
        save_results(forecast_test, mae, rmse, mape)
        
        print("\nANALYSE TERMINEE AVEC SUCCES")
        print("="*60)
        print("FICHIERS GENERES:")
        print("- predictions_prophet_professional.csv")
        print("- performance_metrics_prophet.txt")
        print("- Graphiques d'analyse affichés")
        
    except Exception as e:
        print(f"\nERREUR CRITIQUE : {e}")
        import traceback
        traceback.print_exc()

# Exécution du script
if __name__ == "__main__":
    main()

# %% [markdown]
# TEST MODELE XGBOOST

# %%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_csv('Agence_00001.csv', parse_dates=['Date Opération'])
df = df.sort_values('Date Opération').set_index('Date Opération')

# Créer features
df_feat = df[['Besoin']].copy()
for i in range(1, 8):
    df_feat[f'lag_{i}'] = df_feat['Besoin'].shift(i)
df_feat['rolling_mean_7'] = df_feat['Besoin'].rolling(7).mean()
df_feat['rolling_std_7'] = df_feat['Besoin'].rolling(7).std()
df_feat['jour_semaine'] = df_feat.index.dayofweek
df_feat['mois'] = df_feat.index.month
df_feat['jour_mois'] = df_feat.index.day

# Supprimer les NaN
df_feat = df_feat.dropna()

# Split train/test
train_end = '2025-02-28'
train = df_feat[df_feat.index <= train_end]
test = df_feat[df_feat.index > train_end].head(7)

X_train = train.drop('Besoin', axis=1)
y_train = train['Besoin']
X_test = test.drop('Besoin', axis=1)
y_test = test['Besoin']

# Entraîner
model = XGBRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prédire
preds = model.predict(X_test)

# Évaluer
mae = mean_absolute_error(y_test, preds)
print(f"MAE XGBoost : {mae:,.0f}")  # Devrait être bien inférieur à 335M

# %% [markdown]
# MODELE POUR XGBOOST

# %%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ============= PARTIE 1: PRÉPARATION DES DONNÉES (votre code conservé) =============
def prepare_xgboost_data(file_path='Agence_00001.csv'):
    """Préparer les données pour XGBoost avec features temporelles"""
    print("="*60)
    print("ANALYSE XGBOOST - PRÉDICTION DE FLUX DE LIQUIDITÉ")
    print("="*60)
    
    # Charger les données
    df = pd.read_csv(file_path, parse_dates=['Date Opération'])
    df = df.sort_values('Date Opération').set_index('Date Opération')
    print(f"Données chargées : {len(df)} observations")
    print(f"Période : {df.index.min().date()} à {df.index.max().date()}")
    
    # Créer features (votre code exact)
    df_feat = df[['Besoin']].copy()
    
    # Features de lag
    for i in range(1, 8):
        df_feat[f'lag_{i}'] = df_feat['Besoin'].shift(i)
    
    # Rolling statistics
    df_feat['rolling_mean_7'] = df_feat['Besoin'].rolling(7).mean()
    df_feat['rolling_std_7'] = df_feat['Besoin'].rolling(7).std()
    df_feat['rolling_min_7'] = df_feat['Besoin'].rolling(7).min()
    df_feat['rolling_max_7'] = df_feat['Besoin'].rolling(7).max()
    
    # Features temporelles
    df_feat['jour_semaine'] = df_feat.index.dayofweek
    df_feat['mois'] = df_feat.index.month
    df_feat['jour_mois'] = df_feat.index.day
    df_feat['trimestre'] = df_feat.index.quarter
    
    # Features bancaires spécifiques
    df_feat['debut_mois'] = (df_feat.index.day <= 5).astype(int)
    df_feat['fin_mois'] = (df_feat.index.day >= 25).astype(int)
    df_feat['milieu_mois'] = ((df_feat.index.day >= 14) & 
                              (df_feat.index.day <= 16)).astype(int)
    
    # Variations
    df_feat['diff_1'] = df_feat['Besoin'].diff(1)
    df_feat['diff_5'] = df_feat['Besoin'].diff(5)
    
    # Supprimer les NaN
    df_feat = df_feat.dropna()
    print(f"Après création des features : {len(df_feat)} observations")
    
    return df_feat

# ============= PARTIE 2: ENTRAÎNEMENT ET PRÉDICTION (CORRIGÉE) =============
def train_and_predict_xgboost(df_feat, train_end='2025-02-28', test_days=7):
    """Entraîner XGBoost et faire les prédictions"""
    
    # Split train/test
    train = df_feat[df_feat.index <= train_end]
    test = df_feat[df_feat.index > train_end].head(test_days)
    
    print(f"\nDivision des données:")
    print(f"Train : {len(train)} observations (jusqu'au {train.index.max().date()})")
    print(f"Test  : {len(test)} observations (du {test.index.min().date()} au {test.index.max().date()})")
    
    X_train = train.drop('Besoin', axis=1)
    y_train = train['Besoin']
    X_test = test.drop('Besoin', axis=1)
    y_test = test['Besoin']
    
    # Entraîner avec paramètres optimisés
    print("\nEntraînement du modèle XGBoost...")
    # model = XGBRegressor(
    #     n_estimators=200,
    #     max_depth=6,
    #     learning_rate=0.05,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     objective='reg:squarederror',
    #     eval_metric='rmse',  # Déplacer eval_metric ici
    #     early_stopping_rounds=50
    # )






    model = XGBRegressor(
        # PARAMÈTRES ANTI-OVERFITTING
        n_estimators=100,        # Réduire (était 200) - moins d'arbres = moins de mémorisation
        max_depth=4,             # Réduire (était 6) - arbres moins profonds = moins complexes
        learning_rate=0.1,       # Augmenter (était 0.05) - convergence plus rapide, moins d'itérations
        
        # RÉGULARISATION (les plus importants!)
        reg_alpha=0.1,           # L1 regularization - force certains poids à 0
        reg_lambda=1.0,          # L2 regularization - pénalise les poids élevés
        gamma=0.1,               # Pénalité minimale pour créer une nouvelle branche
        
        # ÉCHANTILLONNAGE
        subsample=0.7,           # Utiliser 70% des données à chaque arbre
        colsample_bytree=0.7,    # Utiliser 70% des features à chaque arbre
        
        # AUTRES
        min_child_weight=5,      # Augmenter pour éviter les branches sur peu de données
        random_state=42
    )






    
    # Entraînement avec eval_set seulement
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Prédire
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    return model, X_train, y_train, X_test, y_test, preds_train, preds_test, train, test


# ============= PARTIE 3: MÉTRIQUES D'ÉVALUATION =============
def calculate_metrics(y_true, y_pred, dataset_name=""):
    """Calculer toutes les métriques de performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE avec gestion des zéros
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    # Incertitude (écart moyen en %)
    incertitude = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    
    print(f"\n{'='*50}")
    print(f"MÉTRIQUES {dataset_name}")
    print(f"{'='*50}")
    print(f"MAE  : {mae:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")
    print(f"Incertitude moyenne : {incertitude:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'incertitude': incertitude}

# ============= PARTIE 4: VISUALISATIONS COMPLÈTES =============
def create_xgboost_visualizations(model, X_train, y_train, X_test, y_test, 
                                  preds_train, preds_test, train, test):
    """Créer tous les graphiques d'analyse"""
    
    # Configuration du style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. GRAPHIQUE: Prédictions vs Réel (Test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Série temporelle
    ax1.plot(y_test.index, y_test.values, 'o-', color=colors[0], 
             label='Valeurs réelles', linewidth=2, markersize=8)
    ax1.plot(y_test.index, preds_test, 's-', color=colors[1], 
             label='Prédictions XGBoost', linewidth=2, markersize=6)
    ax1.fill_between(y_test.index, preds_test * 0.9, preds_test * 1.1, 
                     alpha=0.2, color=colors[1], label='±10% intervalle')
    ax1.set_title('Prédictions XGBoost vs Valeurs Réelles (7 jours)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Besoin (Flux Net)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Formater l'axe y
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if abs(x) >= 1e9 else f'{x/1e6:.0f}M'
    ))
    
    # Subplot 2: Scatter plot
    ax2.scatter(y_test.values, preds_test, color=colors[2], alpha=0.6, s=100, edgecolors='black')
    
    # Ligne parfaite
    min_val = min(y_test.min(), preds_test.min())
    max_val = max(y_test.max(), preds_test.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Prédiction parfaite')
    
    # Régression linéaire
    z = np.polyfit(y_test.values, preds_test, 1)
    p = np.poly1d(z)
    ax2.plot([min_val, max_val], p([min_val, max_val]), 'b-', alpha=0.5, label='Régression')
    
    ax2.set_title('Corrélation Prédictions vs Réel', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Valeurs Réelles')
    ax2.set_ylabel('Prédictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Formater les axes
    for ax in [ax2]:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e9:.1f}B' if abs(x) >= 1e9 else f'{x/1e6:.0f}M'
        ))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e9:.1f}B' if abs(x) >= 1e9 else f'{x/1e6:.0f}M'
        ))
    
    plt.tight_layout()
    plt.show()
    
    # 2. GRAPHIQUE: Vue d'ensemble avec historique
    plt.figure(figsize=(18, 8))
    
    # Afficher les 60 derniers jours d'entraînement + test
    last_60_train = train.tail(60)
    
    plt.plot(last_60_train.index, last_60_train['Besoin'].values, 
            'o-', color='gray', alpha=0.5, label='Historique (60 derniers jours)', markersize=3)
    plt.plot(y_test.index, y_test.values, 
            'o-', color=colors[0], linewidth=2.5, markersize=10, label='Test réel')
    plt.plot(y_test.index, preds_test, 
            's-', color=colors[1], linewidth=2.5, markersize=8, label='Prédictions XGBoost')
    
    # Ligne verticale de séparation
    plt.axvline(x=y_test.index[0], color='red', linestyle='--', alpha=0.5, label='Début prédictions')
    
    plt.title('Vue d\'Ensemble: Historique et Prédictions XGBoost', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Besoin (Flux Net)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Formater l'axe y
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}B' if abs(x) >= 1e9 else f'{x/1e6:.0f}M'
    ))
    
    plt.tight_layout()
    plt.show()
    
    # 3. GRAPHIQUE: Analyse des résidus
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Résidus test
    residuals_test = y_test.values - preds_test
    
    # 3.1: Résidus dans le temps
    axes[0,0].plot(y_test.index, residuals_test, 'o-', color=colors[3], markersize=8)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Résidus au fil du temps (Test)', fontweight='bold')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Résidu')
    axes[0,0].grid(True, alpha=0.3)
    for tick in axes[0,0].get_xticklabels():
        tick.set_rotation(45)
    
    # 3.2: Distribution des résidus
    axes[0,1].hist(residuals_test, bins=min(7, len(residuals_test)), 
                   color=colors[2], alpha=0.7, edgecolor='black')
    axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0,1].set_title('Distribution des Résidus (Test)', fontweight='bold')
    axes[0,1].set_xlabel('Résidu')
    axes[0,1].set_ylabel('Fréquence')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # 3.3: Q-Q Plot
    from scipy import stats
    stats.probplot(residuals_test, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot des Résidus', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 3.4: Résidus vs Prédictions
    axes[1,1].scatter(preds_test, residuals_test, color=colors[0], alpha=0.6, s=100)
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Résidus vs Prédictions', fontweight='bold')
    axes[1,1].set_xlabel('Valeurs Prédites')
    axes[1,1].set_ylabel('Résidus')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. GRAPHIQUE: Importance des features
    plt.figure(figsize=(10, 8))
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'], 
            color=colors[0], alpha=0.7)
    plt.title('Importance des Features dans XGBoost', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs
    for i, (feat, imp) in enumerate(zip(feature_importance['feature'], 
                                        feature_importance['importance'])):
        plt.text(imp, i, f'{imp:.3f}', va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 5. GRAPHIQUE: Courbes d'apprentissage
    if hasattr(model, 'evals_result_'):
        results = model.evals_result()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['validation_0']['rmse'], label='Train', color=colors[0])
        plt.plot(results['validation_1']['rmse'], label='Test', color=colors[1])
        plt.title('Évolution RMSE durant l\'entraînement', fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Calculer le ratio pour détecter l'overfitting
        ratio = np.array(results['validation_1']['rmse']) / np.array(results['validation_0']['rmse'])
        plt.plot(ratio, color=colors[2])
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Pas d\'overfitting')
        plt.title('Détection Overfitting (Test/Train RMSE)', fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio Test/Train')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============= PARTIE 5: FORECASTING MULTI-HORIZONS =============
def forecast_multiple_horizons(model, df_feat, train_end='2025-02-28'):
    """Faire des prédictions pour 7, 14 et 30 jours"""
    
    horizons = [7, 14, 30]
    results = {}
    
    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"FORECASTING HORIZON: {horizon} JOURS")
        print(f"{'='*50}")
        
        # Préparer les données
        train = df_feat[df_feat.index <= train_end]
        test = df_feat[df_feat.index > train_end].head(horizon)
        
        if len(test) < horizon:
            print(f"Attention: Seulement {len(test)} jours disponibles pour horizon {horizon}")
        
        X_test = test.drop('Besoin', axis=1)
        y_test = test['Besoin']
        
        # Prédire
        preds = model.predict(X_test)
        
        # Calculer les métriques
        metrics = calculate_metrics(y_test, preds, f"HORIZON {horizon} JOURS")
        
        results[horizon] = {
            'y_test': y_test,
            'predictions': preds,
            'metrics': metrics
        }
    
    # Graphique comparatif des horizons
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (horizon, data) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(data['y_test'].index, data['y_test'].values, 
               'o-', label='Réel', color='#2E86AB', linewidth=2)
        ax.plot(data['y_test'].index, data['predictions'], 
               's-', label='Prédit', color='#A23B72', linewidth=2)
        ax.set_title(f'Horizon {horizon} jours\nMAPE: {data["metrics"]["mape"]:.1f}%', 
                    fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Besoin')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Formater l'axe y
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e9:.1f}B' if abs(x) >= 1e9 else f'{x/1e6:.0f}M'
        ))
    
    plt.suptitle('Comparaison Multi-Horizons XGBoost', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()
    
    return results

# ============= PARTIE 6: TABLEAU DE SYNTHÈSE =============
def create_summary_table(results_horizons):
    """Créer un tableau récapitulatif des performances"""
    
    summary = []
    for horizon, data in results_horizons.items():
        metrics = data['metrics']
        summary.append({
            'Horizon (jours)': horizon,
            'MAE': f"{metrics['mae']/1e6:.0f}M",
            'RMSE': f"{metrics['rmse']/1e6:.0f}M",
            'MAPE (%)': f"{metrics['mape']:.2f}",
            'R²': f"{metrics['r2']:.4f}",
            'Incertitude (%)': f"{metrics['incertitude']:.2f}"
        })
    
    df_summary = pd.DataFrame(summary)
    
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF - PERFORMANCE XGBOOST PAR HORIZON")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)
    
    return df_summary






# ============= EXÉCUTION PRINCIPALE =============
def main_xgboost_analysis():
    """Fonction principale pour l'analyse XGBoost complète"""
    
    # 1. Préparer les données
    df_feat = prepare_xgboost_data('Agence_00001.csv')
    
    # 2. Entraîner et prédire
    model, X_train, y_train, X_test, y_test, preds_train, preds_test, train, test = \
        train_and_predict_xgboost(df_feat)
    
    # 3. Calculer les métriques
    metrics_train = calculate_metrics(y_train, preds_train, "TRAIN")
    metrics_test = calculate_metrics(y_test, preds_test, "TEST (7 JOURS)")
    
    # 4. Créer les visualisations
    create_xgboost_visualizations(model, X_train, y_train, X_test, y_test, 
                                  preds_train, preds_test, train, test)
    
    # 5. Forecasting multi-horizons
    results_horizons = forecast_multiple_horizons(model, df_feat)
    
    # 6. Tableau de synthèse
    df_summary = create_summary_table(results_horizons)
    
    # 7. Sauvegarder les résultats
    df_summary.to_csv('xgboost_performance_summary.csv', index=False)
    print("\nRésultats sauvegardés dans 'xgboost_performance_summary.csv'")
    
    return model, results_horizons, df_summary

# Lancer l'analyse
if __name__ == "__main__":

    
    model, results, summary = main_xgboost_analysis()

# %% [markdown]
# CELLULE POUR MODELE  --  PROPHET
# Fonctions Prophet (préparation + modèle + CV + forecast)

# %%
# CELLULE PROPHET - corrigée (structure XGBoost, alignement et métriques robustes)
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============= PARTIE 1: PRÉPARATION DES DONNÉES ============
def prepare_prophet_data(file_path='Agence_00001.csv', date_col='Date Opération', target_col='Besoin'):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    print("="*60)
    print("ANALYSE PROPHET - PRÉDICTION DE FLUX DE LIQUIDITÉ")
    print("="*60)
    print(f"Données chargées : {len(df)} observations")
    print(f"Période : {df.index.min().date()} à {df.index.max().date()}")
    df_prophet = df[[target_col]].rename(columns={target_col: 'y'}).reset_index().rename(columns={df.index.name or date_col: 'ds'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_ts = df[[target_col]].copy()
    return df_prophet, df_ts

# ============= PARTIE 2: ENTRAÎNEMENT ET PRÉDICTION ============
def train_and_predict_prophet(df_prophet, df_ts, train_end='2025-02-28', test_days=7):
    train_mask = df_prophet['ds'] <= pd.to_datetime(train_end)
    train_df = df_prophet[train_mask].copy()
    test_df = df_prophet[~train_mask].head(test_days).copy()

    print(f"\nDivision des données:")
    print(f"Train : {len(train_df)} observations (jusqu'au {train_df['ds'].max().date() if len(train_df)>0 else 'N/A'})")
    print(f"Test  : {len(test_df)} observations")

    model = Prophet(
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=5,
        mcmc_samples=0,
        interval_width=0.95,
        uncertainty_samples=500,
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=15,
        changepoint_range=0.8
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='payday', period=15, fourier_order=3)

    if len(train_df)==0:
        raise ValueError("Train vide — vérifie train_end ou le fichier.")

    model.fit(train_df)

    # Construire future jusqu'au max test date afin d'aligner exactement par ds
    if len(test_df)>0:
        last_train = train_df['ds'].max()
        final_test_date = test_df['ds'].max()
        periods = (final_test_date - last_train).days
        if periods < 0:
            # cas improbable: train_end après test dates
            periods = 0
        future = model.make_future_dataframe(periods=periods, freq='D')
        fcst = model.predict(future)
        # reindex predictions by ds for exact alignment with test_df['ds']
        preds_series = fcst.set_index('ds')['yhat'].reindex(test_df['ds'])
        preds_test = preds_series.values  # will include NaN where no prediction
    else:
        preds_test = np.array([])

    # preds_train = prediction for train dates (aligned)
    preds_train = model.predict(train_df)[['ds','yhat']].set_index('ds')['yhat'].reindex(train_df['ds']).values

    # X_train/X_test placeholders to keep signature (Prophet doesn't use X)
    X_train = pd.DataFrame(index=train_df['ds'])
    X_test = pd.DataFrame(index=test_df['ds']) if len(test_df)>0 else pd.DataFrame()
    y_train = train_df['y'].values
    y_test = test_df['y'] if len(test_df)>0 else pd.Series(dtype=float)

    return model, X_train, y_train, X_test, y_test, preds_train, preds_test, train_df, test_df

# ============= PARTIE 3: MÉTRIQUES ROBUSTES (alignement + NaN safe) ============
def calculate_metrics(y_true, y_pred, dataset_name=""):
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # If empty or all nan -> return NaNs and message
    if y_true.size == 0 or y_pred.size == 0:
        print(f"\nAucune donnée valide pour calculer les métriques {dataset_name}")
        return {'mae':np.nan,'rmse':np.nan,'mape':np.nan,'r2':np.nan,'incertitude':np.nan}

    # If lengths differ allow aligning by taking overlapping positions where both not NaN.
    min_len = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        # align by index if pandas passed earlier; else take element-wise up to min_len
        # Prefer to keep entries where both are finite:
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        if mask.sum() == 0:
            # fallback: trim to min_len and drop nans
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        else:
            y_true = y_true[mask]; y_pred = y_pred[mask]
    else:
        # same length: filter NaNs
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true = y_true[mask]; y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"\nAprès alignement il n'y a pas d'échantillon commun pour {dataset_name}")
        return {'mae':np.nan,'rmse':np.nan,'mape':np.nan,'r2':np.nan,'incertitude':np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    denom = np.where(np.abs(y_true) != 0, np.abs(y_true), 1)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    incertitude = np.mean(np.abs(y_true - y_pred) / denom) * 100

    print(f"\n{'='*50}")
    print(f"MÉTRIQUES {dataset_name}")
    print(f"{'='*50}")
    print(f"MAE  : {mae:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2 if not np.isnan(r2) else 'N/A'}")
    print(f"Incertitude moyenne : {incertitude:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'incertitude': incertitude}

# ============= PARTIE 4: VISUALISATIONS (mêmes plots que XGBoost) ============
def create_prophet_visualizations(model, train_df, test_df, y_test, preds_test, preds_train):
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # 1. Préd vs Réel (Test)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
    if len(test_df) > 0:
        # preds_test might contain NaN -> plot only where not NaN for preds line
        ax1.plot(test_df['ds'], y_test.values, 'o-', color=colors[0], label='Valeurs réelles', linewidth=2)
        # plot preds with mask on non-nan
        mask = ~np.isnan(preds_test)
        if mask.any():
            ax1.plot(test_df['ds'][mask], preds_test[mask], 's-', color=colors[1], label='Prédictions Prophet', linewidth=2)
            ax1.fill_between(test_df['ds'][mask], preds_test[mask]*0.9, preds_test[mask]*1.1, alpha=0.2, color=colors[1], label='±10% intervalle')
    else:
        ax1.text(0.5,0.5,"Pas de données test pour afficher Préd vs Réel", ha='center')
    ax1.set_title('Prédictions Prophet vs Valeurs Réelles (Test)')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Besoin'); ax1.legend(); ax1.grid(True); ax1.tick_params(axis='x', rotation=45)

    # scatter
    if len(test_df) > 0 and (~np.isnan(preds_test)).any():
        valid_mask = ~np.isnan(preds_test)
        ax2.scatter(y_test.values[valid_mask], preds_test[valid_mask], alpha=0.6, s=100, edgecolors='black')
        min_val = min(y_test.values[valid_mask].min(), preds_test[valid_mask].min())
        max_val = max(y_test.values[valid_mask].max(), preds_test[valid_mask].max())
        ax2.plot([min_val, max_val],[min_val, max_val],'r--', alpha=0.5)
        z = np.polyfit(y_test.values[valid_mask], preds_test[valid_mask],1); p=np.poly1d(z)
        ax2.plot([min_val,max_val], p([min_val,max_val]), 'b-', alpha=0.5)
    else:
        ax2.text(0.5,0.5,"Pas de points valides pour scatter", ha='center')

    ax2.set_title('Corrélation Prédictions vs Réel'); ax2.set_xlabel('Valeurs Réelles'); ax2.set_ylabel('Prédictions'); ax2.grid(True)
    plt.tight_layout(); plt.show()

    # 2. Vue d'ensemble historique
    plt.figure(figsize=(18,6))
    if len(train_df)>0:
        last_60 = train_df.tail(60)
        plt.plot(last_60['ds'], last_60['y'], 'o-', color='gray', alpha=0.5, markersize=3, label='Historique (60 derniers jours)')
    if len(test_df)>0:
        plt.plot(test_df['ds'], y_test.values, 'o-', color=colors[0], linewidth=2.5, markersize=8, label='Test réel')
        mask = ~np.isnan(preds_test)
        if mask.any():
            plt.plot(test_df['ds'][mask], preds_test[mask], 's-', color=colors[1], linewidth=2.5, markersize=8, label='Prédictions Prophet')
            plt.axvline(x=test_df['ds'].iloc[0], color='red', linestyle='--', alpha=0.5, label='Début prédictions')
    plt.title("Vue d'ensemble: Historique et Prédictions Prophet"); plt.xlabel('Date'); plt.ylabel('Besoin'); plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # 3. Résidus
    if len(test_df)>0 and (~np.isnan(preds_test)).any():
        residuals_test = y_test.values[~np.isnan(preds_test)] - preds_test[~np.isnan(preds_test)]
        fig, axes = plt.subplots(2,2, figsize=(15,12))
        axes[0,0].plot(test_df['ds'][~np.isnan(preds_test)], residuals_test, 'o-'); axes[0,0].axhline(0, linestyle='--', color='black', alpha=0.5); axes[0,0].set_title('Résidus au fil du temps (Test)')
        axes[0,1].hist(residuals_test, bins=min(7,len(residuals_test)), alpha=0.7, edgecolor='black'); axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.5); axes[0,1].set_title('Distribution des Résidus (Test)')
        stats.probplot(residuals_test, dist="norm", plot=axes[1,0]); axes[1,0].set_title('Q-Q Plot des Résidus')
        axes[1,1].scatter(preds_test[~np.isnan(preds_test)], residuals_test, s=100, alpha=0.6); axes[1,1].axhline(0, color='red', linestyle='--', alpha=0.5); axes[1,1].set_title('Résidus vs Prédictions')
        plt.tight_layout(); plt.show()
    else:
        print("Pas d'analyse de résidus (pas assez de prédictions valides).")

    # 4. Components (remplace feature importance)
    try:
        model.plot_components(model.history)
        plt.show()
    except Exception:
        # fallback: predict on full history and plot components
        try:
            hist_fcst = model.predict(model.history)
            model.plot_components(hist_fcst)
            plt.show()
        except Exception:
            print("Impossible d'afficher les composants Prophet.")

# ============= PARTIE 5: FORECASTING MULTI-HORIZONS ============
def forecast_multiple_horizons_prophet(model, df_prophet, train_end='2025-02-28'):
    horizons = [7,14,30]
    results = {}
    for horizon in horizons:
        print(f"\n{'='*50}\nFORECAST HORIZON: {horizon} JOURS\n{'='*50}")
        train_df = df_prophet[df_prophet['ds'] <= pd.to_datetime(train_end)].copy()
        test_df = df_prophet[df_prophet['ds'] > pd.to_datetime(train_end)].head(horizon).copy()
        if len(test_df) < horizon:
            print(f"Attention: Seulement {len(test_df)} jours disponibles pour horizon {horizon}")
        # predict by making future until final test date, then reindex to test ds
        if len(train_df)==0:
            results[horizon] = {'y_test': test_df['y'] if len(test_df)>0 else pd.Series(dtype=float),
                                'predictions': np.array([]), 'metrics': {}}
            continue
        if len(test_df)==0:
            results[horizon] = {'y_test': pd.Series(dtype=float), 'predictions': np.array([]), 'metrics': {}}
            continue
        last_train = train_df['ds'].max()
        final_test_date = test_df['ds'].max()
        periods = (final_test_date - last_train).days
        future = model.make_future_dataframe(periods=periods, freq='D')
        fcst = model.predict(future)
        preds_series = fcst.set_index('ds')['yhat'].reindex(test_df['ds'])
        preds = preds_series.values
        y_test = test_df['y'].values
        metrics = calculate_metrics(y_test, preds, f"HORIZON {horizon} JOURS")
        results[horizon] = {'y_test': pd.Series(y_test, index=test_df['ds']), 'predictions': preds, 'metrics': metrics}

    # Graphique comparatif (1x3)
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    for i, (h, data) in enumerate(results.items()):
        ax = axes[i]
        if len(data['y_test'])>0:
            ax.plot(data['y_test'].index, data['y_test'].values, 'o-', label='Réel')
            # predictions might have NaN -> plot only valid
            preds = data['predictions']
            mask = ~np.isnan(preds)
            if mask.any():
                ax.plot(data['y_test'].index[mask], preds[mask], 's-', label='Prédit')
            ax.set_title(f'Horizon {h} jours\nMAPE: {data["metrics"]["mape"]:.1f}%' if 'mape' in data['metrics'] and not np.isnan(data['metrics']['mape']) else f'Horizon {h} jours')
        else:
            ax.text(0.5,0.5,'No data', ha='center')
        ax.set_xlabel('Date'); ax.set_ylabel('Besoin'); ax.legend(); ax.grid(True); ax.tick_params(axis='x', rotation=45)
    plt.suptitle('Comparaison Multi-Horizons Prophet', fontsize=16, y=1.05); plt.tight_layout(); plt.show()
    return results

# ============= PARTIE 6: TABLEAU DE SYNTHÈSE ============
def create_summary_table(results_horizons):
    summary = []
    for horizon, data in results_horizons.items():
        metrics = data['metrics']
        mae = metrics.get('mae', np.nan)
        rmse = metrics.get('rmse', np.nan)
        mape = metrics.get('mape', np.nan)
        r2 = metrics.get('r2', np.nan)
        inc = metrics.get('incertitude', np.nan)
        summary.append({
            'Horizon (jours)': horizon,
            'MAE': f"{mae/1e6:.0f}M" if not np.isnan(mae) else 'N/A',
            'RMSE': f"{rmse/1e6:.0f}M" if not np.isnan(rmse) else 'N/A',
            'MAPE (%)': f"{mape:.2f}" if not np.isnan(mape) else 'N/A',
            'R²': f"{r2:.4f}" if not np.isnan(r2) else 'N/A',
            'Incertitude (%)': f"{inc:.2f}" if not np.isnan(inc) else 'N/A'
        })
    df_summary = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF - PERFORMANCE PROPHET PAR HORIZON")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)
    return df_summary

# ============= MAIN (identique à XGBoost) ============
def main_prophet_analysis(file_path='Agence_00001.csv', train_end='2025-02-28'):
    df_prophet, df_ts = prepare_prophet_data(file_path)
    model, X_train, y_train, X_test, y_test, preds_train, preds_test, train_df, test_df = train_and_predict_prophet(df_prophet, df_ts, train_end=train_end, test_days=7)
    metrics_train = calculate_metrics(y_train, preds_train, "TRAIN")
    metrics_test = calculate_metrics(y_test.values if len(y_test)>0 else np.array([]), preds_test, "TEST (7 JOURS)")
    create_prophet_visualizations(model, train_df, test_df, y_test, preds_test, preds_train)
    results_horizons = forecast_multiple_horizons_prophet(model, df_prophet, train_end=train_end)
    df_summary = create_summary_table(results_horizons)
    df_summary.to_csv('prophet_performance_summary.csv', index=False)
    joblib.dump(model, 'prophet_model.joblib')
    print("\nRésultats sauvegardés dans 'prophet_performance_summary.csv' et modèle dans 'prophet_model.joblib'")
    return model, results_horizons, df_summary

# Exécution directe
if __name__ == "__main__":
    model_prophet, results_prophet, summary_prophet = main_prophet_analysis('Agence_00001.csv', train_end='2025-02-28')


# %% [markdown]
# CELLULE POUR MODELE LIGHTGBM

# %%
# CELLULE LIGHTGBM - corrigée (utilise callbacks pour early stopping, robuste aux versions)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============= PARTIE 1: PREP ============
def prepare_features_for_lgb(file_path='Agence_00001.csv', date_col='Date Opération', target_col='Besoin'):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    print("="*60)
    print("ANALYSE LIGHTGBM - PRÉDICTION DE FLUX DE LIQUIDITÉ")
    print("="*60)
    print(f"Données chargées : {len(df)} observations")
    print(f"Période : {df.index.min().date()} à {df.index.max().date()}")
    df_feat = df[[target_col]].copy()
    for i in range(1,8):
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
    df_feat['rolling_mean_7'] = df_feat[target_col].rolling(7).mean()
    df_feat['rolling_std_7'] = df_feat[target_col].rolling(7).std()
    df_feat['rolling_min_7'] = df_feat[target_col].rolling(7).min()
    df_feat['rolling_max_7'] = df_feat[target_col].rolling(7).max()
    df_feat['jour_semaine'] = df_feat.index.dayofweek
    df_feat['mois'] = df_feat.index.month
    df_feat['jour_mois'] = df_feat.index.day
    df_feat['trimestre'] = df_feat.index.quarter
    df_feat['debut_mois'] = (df_feat.index.day <= 5).astype(int)
    df_feat['fin_mois'] = (df_feat.index.day >= 25).astype(int)
    df_feat['milieu_mois'] = ((df_feat.index.day >= 14) & (df_feat.index.day <= 16)).astype(int)
    df_feat['diff_1'] = df_feat[target_col].diff(1)
    df_feat['diff_5'] = df_feat[target_col].diff(5)
    df_feat = df_feat.dropna()
    print(f"Après création des features : {len(df_feat)} observations")
    return df_feat

# ============= PARTIE 2: TRAIN&PREDICT ============
def train_and_predict_lgb(df_feat, train_end='2025-02-28', test_days=7):
    train = df_feat[df_feat.index <= pd.to_datetime(train_end)]
    test = df_feat[df_feat.index > pd.to_datetime(train_end)].head(test_days)
    print(f"\nDivision des données:")
    print(f"Train : {len(train)} observations (jusqu'au {train.index.max().date() if len(train)>0 else 'N/A'})")
    print(f"Test  : {len(test)} observations")
    if len(train)==0:
        raise ValueError("Train vide — vérifie train_end ou données.")
    X_train = train.drop('Besoin', axis=1)
    y_train = train['Besoin']
    X_test = test.drop('Besoin', axis=1) if len(test)>0 else pd.DataFrame()
    y_test = test['Besoin'] if len(test)>0 else pd.Series(dtype=float)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.1,
        'min_child_samples': 30,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'path_smooth': 10,
        'max_bin': 127,
        'verbosity': -1,
        'random_state': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data) if len(X_test)>0 else None

    # Use callbacks for early stopping / logging to be compatible across LightGBM versions
    callbacks = [lgb.log_evaluation(period=0)]
    if valid_data is not None:
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=20))

    

    # model = lgb.train(
    #     params,
    #     train_data,
    #     num_boost_round=1000,
    #     valid_sets=[valid_data] if valid_data is not None else None,
    #     callbacks=callbacks if callbacks else None
    # )



    # prepare a dict to record evals
    evals_result = {}

    # build the callbacks: early stopping + record evaluation + silence logs
    callbacks = [
        lgb.early_stopping(stopping_rounds=20),    # compatible wrapper
        lgb.log_evaluation(period=0),              # disable printing per-iter
        lgb.record_evaluation(evals_result)        # fill evals_result
    ]

    # provide both training and validation sets so we can visualize both curves
    valid_sets = [train_data]
    valid_names = ['train']
    if valid_data is not None:
        valid_sets.append(valid_data)
        valid_names.append('valid_0')

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    # --- After training: evals_result will contain the recorded metrics ---
    # Print keys to help debugging
    print("Recorded eval keys:", list(evals_result.keys()))
    # Typical structure: evals_result['train']['rmse'] and evals_result['valid_0']['rmse']

    # Extract RMSE arrays robustly (check names)
    train_key = None
    valid_key = None
    if 'train' in evals_result and 'rmse' in evals_result['train']:
        train_key = ('train','rmse')
    elif valid_names and valid_names[0] in evals_result and 'rmse' in evals_result[valid_names[0]]:
        train_key = (valid_names[0],'rmse')

    if len(valid_names) > 1 and valid_names[1] in evals_result and 'rmse' in evals_result[valid_names[1]]:
        valid_key = (valid_names[1],'rmse')
    elif len(valid_names) > 0 and valid_names[-1] in evals_result and 'rmse' in evals_result[valid_names[-1]]:
        valid_key = (valid_names[-1],'rmse')

    # Fallback search
    if train_key is None or valid_key is None:
        for ds_name, metrics in evals_result.items():
            if train_key is None and 'rmse' in metrics:
                train_key = (ds_name, 'rmse')
            elif valid_key is None and 'rmse' in metrics:
                valid_key = (ds_name, 'rmse')

    # Get arrays for plotting (if present)
    train_rmse = evals_result[train_key[0]][train_key[1]] if train_key is not None else None
    valid_rmse = evals_result[valid_key[0]][valid_key[1]] if valid_key is not None else None

    # Plot training curves if available
    if train_rmse is not None or valid_rmse is not None:
        plt.figure(figsize=(10,5))
        if train_rmse is not None:
            plt.plot(train_rmse, label='Train RMSE')
        if valid_rmse is not None:
            plt.plot(valid_rmse, label='Valid RMSE')
        # show best iteration line if available
        best_it = getattr(model, 'best_iteration', None)
        if best_it is not None and best_it not in (0, None):
            plt.axvline(best_it-1, color='red', linestyle='--', label=f'best_iteration = {best_it}')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('LightGBM Training Curve (RMSE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # zoom: last 100 iters or around best_iter
        if valid_rmse is not None:
            n = len(valid_rmse)
            if best_it not in (0, None):
                start = max(0, best_it - 50)
                end = min(n, best_it + 10)
            else:
                start = max(0, n - 100)
                end = n
            plt.figure(figsize=(10,4))
            if train_rmse is not None:
                plt.plot(range(start, end), train_rmse[start:end], label='Train RMSE (zoom)')
            plt.plot(range(start, end), valid_rmse[start:end], label='Valid RMSE (zoom)')
            if best_it not in (0, None):
                plt.axvline(best_it-1, color='red', linestyle='--', label=f'best_iteration = {best_it}')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Zoom Training Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        print("No eval history recorded — ensure valid_sets and record_evaluation callback are used.")






    # predictions (use best_iteration if present)
    best_iter = getattr(model, 'best_iteration', None)
    num_iter = best_iter if best_iter not in (0, None) else None
    preds_train = model.predict(X_train, num_iteration=num_iter)
    preds_test = model.predict(X_test, num_iteration=num_iter) if len(X_test)>0 else np.array([])

    return model, X_train, y_train, X_test, y_test, preds_train, preds_test, train, test

# ============= PARTIE 3: MÉTRIQUES ROBUSTES (comme Prophet) ============
def calculate_metrics(y_true, y_pred, dataset_name=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        print(f"\nAucune donnée valide pour calculer les métriques {dataset_name}")
        return {'mae':np.nan,'rmse':np.nan,'mape':np.nan,'r2':np.nan,'incertitude':np.nan}
    # align/keep only positions where both finite
    min_len = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        if mask.sum() == 0:
            y_true = y_true[:min_len]; y_pred = y_pred[:min_len]
            mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        else:
            y_true = y_true[mask]; y_pred = y_pred[mask]
    else:
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true = y_true[mask]; y_pred = y_pred[mask]
    if len(y_true) == 0:
        print(f"\nAprès alignement il n'y a pas d'échantillon commun pour {dataset_name}")
        return {'mae':np.nan,'rmse':np.nan,'mape':np.nan,'r2':np.nan,'incertitude':np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true)>1 else np.nan
    denom = np.where(np.abs(y_true) != 0, np.abs(y_true), 1)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    incertitude = np.mean(np.abs(y_true - y_pred) / denom) * 100
    print(f"\n{'='*50}")
    print(f"MÉTRIQUES {dataset_name}")
    print(f"{'='*50}")
    print(f"MAE  : {mae:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2 if not np.isnan(r2) else 'N/A'}")
    print(f"Incertitude moyenne : {incertitude:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'incertitude': incertitude}

# ============= PARTIE 4: VISUALISATIONS (identiques) ============
def create_lgb_visualizations(model, X_train, y_train, X_test, y_test, preds_train, preds_test, train, test):
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
    if len(test)>0 and len(preds_test)>0:
        ax1.plot(test.index, y_test.values, 'o-', color=colors[0], label='Valeurs réelles', linewidth=2)
        ax1.plot(test.index, preds_test, 's-', color=colors[1], label='Prédictions LightGBM', linewidth=2)
        ax1.fill_between(test.index, preds_test*0.9, preds_test*1.1, alpha=0.2, color=colors[1], label='±10% intervalle')
    else:
        ax1.text(0.5,0.5,"Pas assez de points test pour afficher Préd vs Réel", ha='center')
    ax1.set_title('Prédictions LightGBM vs Valeurs Réelles (Test)'); ax1.set_xlabel('Date'); ax1.set_ylabel('Besoin'); ax1.legend(); ax1.grid(True); ax1.tick_params(axis='x', rotation=45)
    if len(test)>0 and len(preds_test)>0:
        ax2.scatter(y_test.values, preds_test, color=colors[2], alpha=0.6, s=100, edgecolors='black')
        min_val = min(y_test.min(), np.min(preds_test)); max_val = max(y_test.max(), np.max(preds_test))
        ax2.plot([min_val, max_val],[min_val, max_val],'r--', alpha=0.5)
        z = np.polyfit(y_test.values, preds_test,1); p=np.poly1d(z)
        ax2.plot([min_val,max_val], p([min_val,max_val]), 'b-', alpha=0.5)
    else:
        ax2.text(0.5,0.5,"Pas assez de points pour scatter", ha='center')
    ax2.set_title('Corrélation Prédictions vs Réel'); ax2.set_xlabel('Valeurs Réelles'); ax2.set_ylabel('Prédictions'); ax2.grid(True)
    plt.tight_layout(); plt.show()

    # 2. Vue d'ensemble historique
    plt.figure(figsize=(18,6))
    last_60 = train.tail(60)
    plt.plot(last_60.index, last_60['Besoin'].values, 'o-', color='gray', alpha=0.5, markersize=3, label='Historique (60 derniers jours)')
    if len(test)>0 and len(preds_test)>0:
        plt.plot(test.index, y_test.values, 'o-', color=colors[0], linewidth=2.5, markersize=8, label='Test réel')
        plt.plot(test.index, preds_test, 's-', color=colors[1], linewidth=2.5, markersize=8, label='Prédictions LightGBM')
        plt.axvline(x=test.index[0], color='red', linestyle='--', alpha=0.5, label='Début prédictions')
    plt.title("Vue d'ensemble: Historique et Prédictions LightGBM"); plt.xlabel('Date'); plt.ylabel('Besoin'); plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # 3. Résidus
    if len(test)>0 and len(preds_test)>0:
        residuals_test = y_test.values - preds_test
        fig, axes = plt.subplots(2,2, figsize=(15,12))
        axes[0,0].plot(test.index, residuals_test, 'o-'); axes[0,0].axhline(0, linestyle='--', color='black', alpha=0.5); axes[0,0].set_title('Résidus au fil du temps (Test)')
        axes[0,1].hist(residuals_test, bins=min(7,len(residuals_test)), alpha=0.7, edgecolor='black'); axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.5); axes[0,1].set_title('Distribution des Résidus (Test)')
        stats.probplot(residuals_test, dist="norm", plot=axes[1,0]); axes[1,0].set_title('Q-Q Plot des Résidus')
        axes[1,1].scatter(preds_test, residuals_test, s=100, alpha=0.6); axes[1,1].axhline(0, color='red', linestyle='--', alpha=0.5); axes[1,1].set_title('Résidus vs Prédictions')
        plt.tight_layout(); plt.show()
    else:
        print("Pas d'analyse de résidus (pas assez de prédictions valides).")

    # 4. Feature importance
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importance(importance_type='gain')}).sort_values('importance', ascending=True)
    plt.figure(figsize=(10,8)); plt.barh(importance['feature'], importance['importance']); plt.title('Importance des Features (gain)'); plt.xlabel('Importance'); plt.grid(True, axis='x', alpha=0.3)
    for i,(feat,imp) in enumerate(zip(importance['feature'], importance['importance'])):
        plt.text(imp, i, f'{imp:.0f}', va='center', ha='left', fontsize=9)
    plt.tight_layout(); plt.show()

# ============= PARTIE 5: FORECASTING MULTI-HORIZONS ============
def forecast_multiple_horizons_lgb(model, df_feat, train_end='2025-02-28'):
    horizons = [7,14,30]
    results = {}
    for horizon in horizons:
        print(f"\n{'='*50}\nFORECAST HORIZON: {horizon} JOURS\n{'='*50}")
        train = df_feat[df_feat.index <= pd.to_datetime(train_end)]
        test = df_feat[df_feat.index > pd.to_datetime(train_end)].head(horizon)
        if len(test) < horizon:
            print(f"Attention: Seulement {len(test)} jours disponibles pour horizon {horizon}")
        if len(test)==0:
            results[horizon] = {'y_test': pd.Series(dtype=float), 'predictions': np.array([]), 'metrics': {}}
            continue
        X_test = test.drop('Besoin', axis=1)
        y_test = test['Besoin'].values
        best_iter = getattr(model, 'best_iteration', None)
        preds = model.predict(X_test, num_iteration=best_iter if best_iter not in (0, None) else None)
        metrics = calculate_metrics(y_test, preds, f"HORIZON {horizon} JOURS")
        results[horizon] = {'y_test': pd.Series(y_test, index=test.index), 'predictions': preds, 'metrics': metrics}
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    for i,(h,data) in enumerate(results.items()):
        ax = axes[i]
        if len(data['y_test'])>0:
            ax.plot(data['y_test'].index, data['y_test'].values, 'o-', label='Réel')
            if len(data['predictions'])>0:
                ax.plot(data['y_test'].index, data['predictions'], 's-', label='Prédit')
            ax.set_title(f'Horizon {h} jours\nMAPE: {data["metrics"]["mape"]:.1f}%' if 'mape' in data['metrics'] and not np.isnan(data['metrics']['mape']) else f'Horizon {h} jours')
        else:
            ax.text(0.5,0.5,'No data', ha='center')
        ax.set_xlabel('Date'); ax.set_ylabel('Besoin'); ax.legend(); ax.grid(True); ax.tick_params(axis='x', rotation=45)
    plt.suptitle('Comparaison Multi-Horizons LightGBM', fontsize=16, y=1.05); plt.tight_layout(); plt.show()
    return results

# ============= PARTIE 6: TABLEAU DE SYNTHÈSE ============
def create_summary_table(results_horizons):
    summary = []
    for horizon, data in results_horizons.items():
        metrics = data['metrics']
        mae = metrics.get('mae', np.nan); rmse = metrics.get('rmse', np.nan); mape = metrics.get('mape', np.nan); r2 = metrics.get('r2', np.nan); inc = metrics.get('incertitude', np.nan)
        summary.append({'Horizon (jours)': horizon, 'MAE': f"{mae/1e6:.0f}M" if not np.isnan(mae) else 'N/A', 'RMSE': f"{rmse/1e6:.0f}M" if not np.isnan(rmse) else 'N/A', 'MAPE (%)': f"{mape:.2f}" if not np.isnan(mape) else 'N/A', 'R²': f"{r2:.4f}" if not np.isnan(r2) else 'N/A', 'Incertitude (%)': f"{inc:.2f}" if not np.isnan(inc) else 'N/A'})
    df_summary = pd.DataFrame(summary)
    print("\n" + "="*80); print("TABLEAU RÉCAPITULATIF - PERFORMANCE LIGHTGBM PAR HORIZON"); print("="*80); print(df_summary.to_string(index=False)); print("="*80)
    return df_summary

# ============= MAIN (identique) ============
def main_lgb_analysis(file_path='Agence_00001.csv', train_end='2025-02-28'):
    df_feat = prepare_features_for_lgb(file_path)
    model, X_train, y_train, X_test, y_test, preds_train, preds_test, train, test = train_and_predict_lgb(df_feat, train_end=train_end, test_days=7)
    metrics_train = calculate_metrics(y_train.values, preds_train, "TRAIN")
    metrics_test = calculate_metrics(y_test.values if len(y_test)>0 else np.array([]), preds_test, "TEST (7 JOURS)")
    create_lgb_visualizations(model, X_train, y_train, X_test, y_test, preds_train, preds_test, train, test)
    results_horizons = forecast_multiple_horizons_lgb(model, df_feat, train_end=train_end)
    df_summary = create_summary_table(results_horizons)
    df_summary.to_csv('lgbm_performance_summary.csv', index=False)
    try:
        model.save_model('lgbm_model.txt'); print("LightGBM model saved to lgbm_model.txt")
    except Exception:
        joblib.dump(model, 'lgbm_model.joblib'); print("LightGBM model saved to lgbm_model.joblib")
    return model, results_horizons, df_summary

# Exécution directe
if __name__ == "__main__":
    model_lgb, results_lgb, summary_lgb = main_lgb_analysis('Agence_00001.csv', train_end='2025-02-28')


# %% [markdown]
# CELLULE POUR MODELE RNN / LSTM

# %%



