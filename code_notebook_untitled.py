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
df = pd.read_csv('Agence_00046.csv', parse_dates=['Date Opération'])
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

# %%
from statsmodels.tsa.stattools import adfuller

result = adfuller(train_data.dropna())
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(train_data.dropna(), model='additive', period=5, extrapolate_trend='freq')
fig = decomp.plot()
plt.show()

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train_data.dropna(), lags=30, ax=ax[0])
plot_pacf(train_data.dropna(), lags=30, ax=ax[1], method='ywm')
plt.show()

# %%
import matplotlib.pyplot as plt

# Exemple : créer un forecast factice (à remplacer par tes vraies prédictions !)
forecast = test_data.values * 1.05  # juste +5% pour l'exemple — à remplacer !

plt.figure(figsize=(12, 6))
plt.plot(test_data.index.to_numpy(), test_data.values, 'o-', label='Réel (mars 2025)', color='blue', linewidth=2)
plt.plot(test_data.index.to_numpy(), forecast, 'x-', label='Prédit', color='red', linewidth=2)
plt.title("Prédiction vs Réel - 7 premiers jours de mars 2025")
plt.xlabel("Date")
plt.ylabel("Besoin")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

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
    """Créer et entraîner le modèle Prophet optimisé"""
    print("\nInitialisation du modèle Prophet...")
    
    # Configuration optimisée du modèle
    model = Prophet(
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000
    )
    
    # Ajouter saisonnalités personnalisées
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    
    print("Entraînement du modèle...")
    model.fit(train_data)
    print("Entraînement terminé avec succès")
    
    return model

def make_predictions(model, train_data, test_data):
    """Générer les prédictions"""
    print("\nGénération des prédictions...")
    
    # Créer le DataFrame future pour les prédictions
    periods = len(test_data) if len(test_data) > 0 else 7  
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    # Générer les prédictions
    forecast = model.predict(future)
    
    # Extraire les prédictions pour la période de test
    if len(test_data) > 0:
        forecast_test = forecast.tail(len(test_data))[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        print("\nPrédictions pour la période de test :")
        print(forecast_test[['ds', 'yhat']].round(0))
    else:
        forecast_test = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        print("\nPrédictions pour les 7 prochains jours :")
        print(forecast_test[['ds', 'yhat']].round(0))
    
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
        
        plt.title("Comparaison : Prédictions vs Valeurs Réelles", fontsize=16, fontweight='bold', pad=20)
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

# %%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_csv('Agence_00046.csv', parse_dates=['Date Opération'])
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

# %%



