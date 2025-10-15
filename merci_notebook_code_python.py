# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Librairies spécifiques pour les séries temporelles
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA



# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Librairies importées avec succès !")































































# %%
# Import des données de l'agence 00001
df = pd.read_csv('Agence_00046.csv')

print("=== EXPLORATION INITIALE DES DONNÉES ===")
print(f"Forme du dataset : {df.shape}")
print("\n=== PREMIÈRES LIGNES ===")
print(df.head())
print("\n=== DERNIÈRES LIGNES ===")
print(df.tail())
print("\n=== INFORMATIONS GÉNÉRALES ===")
print(df.info())
print("\n=== STATISTIQUES DESCRIPTIVES DE 'BESOIN' ===")
print(df['Besoin'].describe())

# %%
# Nettoyage et préparation des données
print("=== NETTOYAGE DES DONNÉES ===")

# Convertir la colonne Date en datetime
df['Date Opération'] = pd.to_datetime(df['Date Opération'])

# Définir la date comme index
df.set_index('Date Opération', inplace=True)

# Renommer la colonne pour plus de simplicité
df.rename(columns={'Besoin': 'Besoin'}, inplace=True)

print("✓ Dates converties en datetime")
print("✓ Index défini sur les dates")
print(f"✓ Période des données : {df.index.min()} à {df.index.max()}")
print(f"✓ Nombre de jours : {len(df)}")

# Vérifier s'il y a des valeurs manquantes
print(f"\n=== VALEURS MANQUANTES ===")
print(f"Valeurs manquantes : {df['Besoin'].isnull().sum()}")

# Affichage des nouvelles données nettoyées
print("\n=== DONNÉES APRÈS NETTOYAGE ===")
print(df.head(10))

# %%
# Division des données selon le cahier des charges
# Train/Test : jusqu'à février 2025
# Validation : Mars 2025+

print("=== DIVISION TRAIN/TEST/VALIDATION ===")

# Définir les dates de coupure
date_limite_train = '2025-02-28'
date_debut_validation = '2025-03-01'

# Création des datasets
df_train_test = df[df.index <= date_limite_train].copy()
df_validation = df[df.index >= date_debut_validation].copy()

print(f"✓ Train/Test : {df_train_test.index.min()} à {df_train_test.index.max()}")
print(f"  - Nombre d'observations : {len(df_train_test)}")
print(f"✓ Validation : {df_validation.index.min()} à {df_validation.index.max()}")
print(f"  - Nombre d'observations : {len(df_validation)}")

# Statistiques des différentes périodes
print(f"\n=== STATISTIQUES PAR PÉRIODE ===")
print("Train/Test (Sept 2022 - Fév 2025):")
print(df_train_test['Besoin'].describe())
print(f"\nValidation (Mars 2025+):")
print(df_validation['Besoin'].describe())

# %%
# Visualisation de la série temporelle complète
print("=== VISUALISATION DE LA SÉRIE TEMPORELLE ===")

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Graphique 1 : Série complète
axes[0].plot(df.index, df['Besoin'], color='blue', linewidth=1)
axes[0].axvline(x=pd.to_datetime('2025-02-28'), color='red', linestyle='--', 
                label='Limite Train/Test - Validation', linewidth=2)
axes[0].set_title('Série Temporelle Complète - Besoin Journalier Agence 00001', fontsize=16, pad=20)
axes[0].set_ylabel('Besoin (CFA)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique 2 : Focus sur Train/Test (données d'entraînement)
axes[1].plot(df_train_test.index, df_train_test['Besoin'], color='darkblue', linewidth=1)
axes[1].set_title('Série Train/Test (Sept 2022 - Fév 2025)', fontsize=14, pad=15)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Besoin (CFA)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Observations visuelles
print("\n=== OBSERVATIONS VISUELLES INITIALES ===")
print("À partir de cette visualisation, pouvez-vous identifier :")
print("1. Une tendance générale ?")
print("2. Des patterns saisonniers ?")
print("3. La présence de cycles ?")
print("4. Des outliers ou valeurs extrêmes ?")

# %%
# Analyse plus détaillée des patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution des valeurs
axes[0,0].hist(df_train_test['Besoin'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(df_train_test['Besoin'].mean(), color='red', linestyle='--', 
                  label=f'Moyenne: {df_train_test["Besoin"].mean():.0f}')
axes[0,0].set_title('Distribution des Besoins')
axes[0,0].set_xlabel('Besoin (CFA)')
axes[0,0].set_ylabel('Fréquence')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Boxplot par mois pour détecter la saisonnalité
df_train_test_copy = df_train_test.copy()
df_train_test_copy['Mois'] = df_train_test_copy.index.month
monthly_data = [df_train_test_copy[df_train_test_copy['Mois'] == i]['Besoin'].values 
                for i in range(1, 13)]
axes[0,1].boxplot(monthly_data, labels=['Jan','Fév','Mar','Avr','Mai','Jun',
                                        'Jul','Aoû','Sep','Oct','Nov','Déc'])
axes[0,1].set_title('Variation Mensuelle des Besoins')
axes[0,1].set_ylabel('Besoin (CFA)')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# Moyennes mobiles pour voir la tendance
axes[1,0].plot(df_train_test.index, df_train_test['Besoin'], alpha=0.5, label='Données originales')
axes[1,0].plot(df_train_test.index, df_train_test['Besoin'].rolling(window=30).mean(), 
               color='red', linewidth=2, label='Moyenne mobile 30j')
axes[1,0].plot(df_train_test.index, df_train_test['Besoin'].rolling(window=90).mean(), 
               color='green', linewidth=2, label='Moyenne mobile 90j')
axes[1,0].set_title('Tendances avec Moyennes Mobiles')
axes[1,0].set_ylabel('Besoin (CFA)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Evolution année par année
df_train_test_copy['Année'] = df_train_test_copy.index.year
for year in df_train_test_copy['Année'].unique():
    year_data = df_train_test_copy[df_train_test_copy['Année'] == year]
    axes[1,1].plot(year_data.index.dayofyear, year_data['Besoin'], 
                   alpha=0.7, linewidth=1, label=f'{year}')
axes[1,1].set_title('Comparaison Annuelle (Jour de l\'année)')
axes[1,1].set_xlabel('Jour de l\'année')
axes[1,1].set_ylabel('Besoin (CFA)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Test de stationnarité - Augmented Dickey-Fuller
print("=== TEST DE STATIONNARITÉ (AUGMENTED DICKEY-FULLER) ===")

def adfuller_test(series, title):
    """Fonction pour effectuer le test ADF avec interprétation complète"""
    result = adfuller(series.dropna())
    
    print(f'\n--- {title} ---')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    # Interprétation
    if result[1] <= 0.05:
        print("✅ RÉSULTAT: Série STATIONNAIRE")
        print("   → Forte évidence contre H0 (présence de racine unitaire)")
        print("   → Rejet de H0 : la série est stationnaire")
    else:
        print("❌ RÉSULTAT: Série NON-STATIONNAIRE") 
        print("   → Évidence faible contre H0")
        print("   → Acceptance de H0 : la série a une racine unitaire (non-stationnaire)")
    
    return result

# Test sur la série originale
adf_original = adfuller_test(df_train_test['Besoin'], "SÉRIE ORIGINALE")

# %%
# Décomposition de la série temporelle
from statsmodels.tsa.seasonal import seasonal_decompose

print("=== DÉCOMPOSITION DE LA SÉRIE TEMPORELLE ===")

# Décomposition additive et multiplicative
fig, axes = plt.subplots(4, 2, figsize=(18, 16))

# DÉCOMPOSITION ADDITIVE
print("Calcul de la décomposition ADDITIVE...")
decomposition_add = seasonal_decompose(df_train_test['Besoin'], 
                                       model='additive', 
                                       period=252)  # ~252 jours ouvrés par an

decomposition_add.observed.plot(ax=axes[0,0], title='Série Observée (Additive)')
decomposition_add.trend.plot(ax=axes[1,0], title='Tendance (Additive)', color='red')
decomposition_add.seasonal.plot(ax=axes[2,0], title='Saisonnalité (Additive)', color='green')
decomposition_add.resid.plot(ax=axes[3,0], title='Résidus (Additive)', color='purple')

# DÉCOMPOSITION MULTIPLICATIVE
print("Calcul de la décomposition MULTIPLICATIVE...")
# Pour éviter les valeurs négatives, on ajuste les données
series_positive = df_train_test['Besoin'] - df_train_test['Besoin'].min() + 1

decomposition_mult = seasonal_decompose(series_positive, 
                                        model='multiplicative', 
                                        period=252)

decomposition_mult.observed.plot(ax=axes[0,1], title='Série Observée (Multiplicative)')
decomposition_mult.trend.plot(ax=axes[1,1], title='Tendance (Multiplicative)', color='red')
decomposition_mult.seasonal.plot(ax=axes[2,1], title='Saisonnalité (Multiplicative)', color='green')
decomposition_mult.resid.plot(ax=axes[3,1], title='Résidus (Multiplicative)', color='purple')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyse des résidus des deux décompositions
print("\n=== ANALYSE DES RÉSIDUS ===")
print("Résidus Additifs:")
print(f"  Moyenne: {decomposition_add.resid.mean():.2f}")
print(f"  Écart-type: {decomposition_add.resid.std():.2f}")
print(f"  Min: {decomposition_add.resid.min():.2f}")
print(f"  Max: {decomposition_add.resid.max():.2f}")

print("\nRésidus Multiplicatifs:")
print(f"  Moyenne: {decomposition_mult.resid.mean():.2f}")
print(f"  Écart-type: {decomposition_mult.resid.std():.2f}")
print(f"  Min: {decomposition_mult.resid.min():.2f}")
print(f"  Max: {decomposition_mult.resid.max():.2f}")

# %%
# Analyse approfondie des résidus pour choisir le meilleur modèle
print("=== COMPARAISON DES MODÈLES DE DÉCOMPOSITION ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Test de normalité des résidus
from scipy import stats

# Résidus additifs
resid_add = decomposition_add.resid.dropna()
resid_mult = decomposition_mult.resid.dropna()

# Histogrammes des résidus
axes[0,0].hist(resid_add, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[0,0].set_title('Distribution Résidus Additifs')
axes[0,0].axvline(resid_add.mean(), color='red', linestyle='--', label=f'Moyenne: {resid_add.mean():.0f}')
axes[0,0].legend()

axes[1,0].hist(resid_mult, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1,0].set_title('Distribution Résidus Multiplicatifs')
axes[1,0].axvline(resid_mult.mean(), color='red', linestyle='--', label=f'Moyenne: {resid_mult.mean():.2f}')
axes[1,0].legend()

# Q-Q plots pour la normalité
stats.probplot(resid_add, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot Résidus Additifs')

stats.probplot(resid_mult, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot Résidus Multiplicatifs')

# Évolution temporelle des résidus
axes[0,2].plot(resid_add, color='purple', alpha=0.7)
axes[0,2].set_title('Évolution Résidus Additifs')
axes[0,2].axhline(0, color='red', linestyle='--')

axes[1,2].plot(resid_mult, color='orange', alpha=0.7)
axes[1,2].set_title('Évolution Résidus Multiplicatifs')
axes[1,2].axhline(1, color='red', linestyle='--')

plt.tight_layout()
plt.show()

# Tests statistiques sur les résidus
print("\n=== TESTS STATISTIQUES SUR LES RÉSIDUS ===")
print("Résidus Additifs:")
shapiro_add = stats.shapiro(resid_add[:5000] if len(resid_add) > 5000 else resid_add)
print(f"  Test de normalité (Shapiro): p-value = {shapiro_add[1]:.6f}")
ljung_add = sm.stats.diagnostic.acorr_ljungbox(resid_add, lags=10, return_df=True)
print(f"  Test d'autocorrélation (Ljung-Box): p-value = {ljung_add['lb_pvalue'].iloc[-1]:.6f}")

print("\nRésidus Multiplicatifs:")
shapiro_mult = stats.shapiro(resid_mult[:5000] if len(resid_mult) > 5000 else resid_mult)
print(f"  Test de normalité (Shapiro): p-value = {shapiro_mult[1]:.6f}")
ljung_mult = sm.stats.diagnostic.acorr_ljungbox(resid_mult, lags=10, return_df=True)
print(f"  Test d'autocorrélation (Ljung-Box): p-value = {ljung_mult['lb_pvalue'].iloc[-1]:.6f}")

# %%
# Analyse de la volatilité pour justifier GARCH
print("=== ANALYSE DE LA VOLATILITÉ ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Série des rendements (différences premières)
df_train_test['rendements'] = df_train_test['Besoin'].diff()
df_train_test['rendements_pct'] = df_train_test['Besoin'].pct_change()

axes[0,0].plot(df_train_test.index, df_train_test['rendements'], alpha=0.7, color='blue')
axes[0,0].set_title('Rendements Absolus (Différences Premières)')
axes[0,0].set_ylabel('Rendements')

# 2. Rendements au carré (proxy de volatilité)
df_train_test['rendements_carre'] = df_train_test['rendements']**2
axes[0,1].plot(df_train_test.index, df_train_test['rendements_carre'], alpha=0.7, color='red')
axes[0,1].set_title('Rendements au Carré (Volatilité)')
axes[0,1].set_ylabel('Rendements²')

# 3. Volatilité mobile
window = 21  # ~1 mois bancaire
df_train_test['volatilite_mobile'] = df_train_test['rendements'].rolling(window=window).std()
axes[1,0].plot(df_train_test.index, df_train_test['volatilite_mobile'], color='purple', linewidth=2)
axes[1,0].set_title(f'Volatilité Mobile ({window} jours)')
axes[1,0].set_ylabel('Écart-type mobile')

# 4. Autocorrélation des rendements au carré
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_train_test['rendements_carre'].dropna(), lags=40, ax=axes[1,1], title='ACF Rendements²')

plt.tight_layout()
plt.show()

# Tests statistiques pour la présence d'effets ARCH
from statsmodels.stats.diagnostic import het_arch
print("\n=== TESTS POUR EFFETS ARCH ===")
rendements_clean = df_train_test['rendements'].dropna()

# Test de Ljung-Box sur les rendements au carré
ljung_box_r2 = sm.stats.diagnostic.acorr_ljungbox(rendements_clean**2, lags=10, return_df=True)
print("Test Ljung-Box sur rendements² (H0: pas d'autocorrélation):")
print(f"  p-value: {ljung_box_r2['lb_pvalue'].iloc[-1]:.6f}")

# Test ARCH de Engle
try:
    arch_test = het_arch(rendements_clean, nlags=5)
    print(f"\nTest ARCH d'Engle (H0: pas d'effets ARCH):")
    print(f"  Statistique: {arch_test[0]:.4f}")
    print(f"  p-value: {arch_test[1]:.6f}")
    if arch_test[1] < 0.05:
        print("  RÉSULTAT: Effets ARCH détectés - GARCH recommandé")
    else:
        print("  RÉSULTAT: Pas d'effets ARCH détectés")
except:
    print("Test ARCH non disponible avec cette version")

# %%
# Analyse ACF/PACF pour déterminer les ordres ARIMA
print("=== ANALYSE ACF/PACF POUR ORDRES ARIMA ===")

# Utilisons les résidus de décomposition ou la série différenciée
serie_a_analyser = df_train_test['rendements'].dropna()

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# ACF et PACF de la série originale
plot_acf(df_train_test['Besoin'].dropna(), lags=40, ax=axes[0,0], title='ACF - Série Originale')
plot_pacf(df_train_test['Besoin'].dropna(), lags=40, ax=axes[0,1], title='PACF - Série Originale')

# ACF et PACF des rendements (différence première)
plot_acf(serie_a_analyser, lags=40, ax=axes[1,0], title='ACF - Rendements (diff=1)')
plot_pacf(serie_a_analyser, lags=40, ax=axes[1,1], title='PACF - Rendements (diff=1)')

# ACF et PACF des rendements au carré
plot_acf(serie_a_analyser**2, lags=40, ax=axes[2,0], title='ACF - Rendements² (pour GARCH)')
plot_pacf(serie_a_analyser**2, lags=40, ax=axes[2,1], title='PACF - Rendements² (pour GARCH)')

plt.tight_layout()
plt.show()

print("\n=== INTERPRÉTATION POUR ORDRES ARIMA ===")
print("Regardez les graphiques ACF/PACF des rendements:")
print("- PACF: Coupure nette après lag p → ordre AR(p)")
print("- ACF: Coupure nette après lag q → ordre MA(q)")
print("- Si décroissance exponentielle → mélange ARMA")
print("\nPour GARCH, observez l'ACF des rendements² :")
print("- Décroissance lente = effets GARCH présents")

# %%
# Entraînement des modèles ARIMA sur série originale
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

print("=== ENTRAÎNEMENT MODÈLES ARIMA ===")

# Série d'entraînement (originale, pas différenciée)
y_train = df_train_test['Besoin'].dropna()

# Test de plusieurs ordres ARIMA basés sur l'analyse ACF/PACF
ordres_a_tester = [
    (1, 0, 0),  # AR(1) - basé sur PACF
    (1, 0, 1),  # ARMA(1,1) - combinaison
    (2, 0, 1),  # ARMA(2,1) - basé sur PACF lag 2
    (1, 0, 2),  # ARMA(1,2) - alternative
    (1, 0, 2),  # ARMA(1,2) - alternative
    (1, 1, 1),  # ARMA(1,2) - alternative
    (3, 0, 3),  # ARMA(1,2) - alternative
    (2, 0, 2),  # ARMA(1,2) - alternative
    (3, 0, 2),  # ARMA(1,2) - alternative
    (2, 0, 3),  # ARMA(1,2) - alternative
    (2, 1, 2),  # ARMA(1,2) - alternative
    (2, 1, 0),  # ARMA(1,2) - alternative
    (2, 1, 1),  # ARMA(1,2) - alternative
]

resultats_arima = {}

for ordre in ordres_a_tester:
    try:
        print(f"\nEntraînement ARIMA{ordre}...")
        
        model = ARIMA(y_train, order=ordre)
        fitted_model = model.fit()
        
        # Métriques
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # Test des résidus
        residus = fitted_model.resid
        ljung_test = sm.stats.diagnostic.acorr_ljungbox(residus, lags=10, return_df=True)
        ljung_pvalue = ljung_test['lb_pvalue'].iloc[-1]
        
        resultats_arima[ordre] = {
            'model': fitted_model,
            'aic': aic,
            'bic': bic,
            'ljung_pvalue': ljung_pvalue
        }
        
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Ljung-Box p-value: {ljung_pvalue:.4f}")
        
    except Exception as e:
        print(f"  ERREUR: {e}")

# Sélection du meilleur modèle
meilleur_ordre = min(resultats_arima.keys(), key=lambda x: resultats_arima[x]['aic'])
meilleur_modele = resultats_arima[meilleur_ordre]

print(f"\n=== MEILLEUR MODÈLE ARIMA{meilleur_ordre} ===")
print(meilleur_modele['model'].summary())

# %%
# Entraînement du modèle GARCH
from arch import arch_model
import numpy as np

print("=== ENTRAÎNEMENT MODÈLE GARCH ===")

# Préparation des données pour GARCH (centrer les rendements)
rendements = df_train_test['Besoin'].diff().dropna()
rendements_centres = rendements - rendements.mean()

# Test de différents ordres GARCH
ordres_garch = [
    (1, 1),  # GARCH(1,1) - standard
    (1, 2),  # GARCH(1,2) 
    (2, 1),  # GARCH(2,1)
]

resultats_garch = {}

for p, q in ordres_garch:
    try:
        print(f"\nEntraînement GARCH({p},{q})...")
        
        # Modèle GARCH
        garch_model = arch_model(rendements_centres, 
                                vol='Garch', 
                                p=p, q=q,
                                rescale=False)
        
        garch_fitted = garch_model.fit(disp='off')
        
        resultats_garch[(p,q)] = {
            'model': garch_fitted,
            'aic': garch_fitted.aic,
            'bic': garch_fitted.bic
        }
        
        print(f"  AIC: {garch_fitted.aic:.2f}")
        print(f"  BIC: {garch_fitted.bic:.2f}")
        
    except Exception as e:
        print(f"  ERREUR: {e}")

# Sélection du meilleur GARCH
if resultats_garch:
    meilleur_garch_ordre = min(resultats_garch.keys(), key=lambda x: resultats_garch[x]['aic'])
    meilleur_garch = resultats_garch[meilleur_garch_ordre]
    
    print(f"\n=== MEILLEUR MODÈLE GARCH{meilleur_garch_ordre} ===")
    print(meilleur_garch['model'].summary())
else:
    print("Aucun modèle GARCH n'a convergé")

# %%
# %%
# ÉTAPE 1: AMÉLIORATION DU MODÈLE ARIMA - Gestion problème de convergence
print("=== AMÉLIORATION MODÈLE ARIMA ===")

# Normalisation des données pour éviter les problèmes numériques
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardisation de la série
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(df_train_test[['Besoin']])
y_train_scaled = pd.Series(y_train_scaled.flatten(), index=df_train_test.index)

print(" Données standardisées pour améliorer la convergence")

# Re-test des modèles ARIMA avec données standardisées
ordres_arima_ameliores = [
    (1, 0, 1),  # Notre meilleur précédent
    (2, 0, 2),  # Plus complexe basé sur ACF/PACF
    (1, 1, 1),  # Avec différenciation
    (2, 1, 2),  # ARIMA plus complet
    (1, 1, 1),  # ARMA(1,2) - alternative
    (3, 0, 3),  # ARMA(1,2) - alternative
    (2, 0, 2),  # ARMA(1,2) - alternative
    (3, 0, 2),  # ARMA(1,2) - alternative
    (2, 0, 3),  # ARMA(1,2) - alternative
    (2, 1, 2),  # ARMA(1,2) - alternative
    (2, 1, 0),  # ARMA(1,2) - alternative
    (2, 1, 1),  # ARMA(1,2) - alternative
]

resultats_arima_ameliores = {}

for ordre in ordres_arima_ameliores:
    try:
        print(f"\nEntraînement ARIMA{ordre} (données standardisées)...")
        
        model = ARIMA(y_train_scaled, order=ordre)
        fitted_model = model.fit()
        
        # Métriques
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # Test des résidus
        residus = fitted_model.resid
        ljung_test = sm.stats.diagnostic.acorr_ljungbox(residus, lags=15, return_df=True)
        ljung_pvalue = ljung_test['lb_pvalue'].iloc[-1]
        
        # Test de normalité des résidus
        shapiro_test = stats.shapiro(residus[:5000] if len(residus) > 5000 else residus)
        shapiro_pvalue = shapiro_test[1]
        
        resultats_arima_ameliores[ordre] = {
            'model': fitted_model,
            'aic': aic,
            'bic': bic,
            'ljung_pvalue': ljung_pvalue,
            'shapiro_pvalue': shapiro_pvalue,
            'residus_std': residus.std()
        }
        
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Ljung-Box p-value: {ljung_pvalue:.4f}")
        print(f"  Shapiro p-value: {shapiro_pvalue:.4f}")
        
    except Exception as e:
        print(f"  ERREUR: {e}")

# %%
# ÉTAPE 2: TEST MODÈLES SARIMA (avec saisonnalité)
print("\n=== TEST MODÈLES SARIMA ===")

# Paramètres saisonniers à tester (période = 5 jours ouvrés ou 21 jours ~1 mois)
ordres_sarima = [
    ((1, 0, 1), (1, 0, 1, 5)),   # Saisonnalité hebdomadaire
    ((1, 0, 1), (1, 0, 1, 21)),  # Saisonnalité mensuelle
    ((1, 1, 1), (1, 1, 1, 5)),   # Avec différenciation
    ((2, 0, 2), (1, 0, 1, 5)),   # Plus complexe 
    ((2, 1, 2), (1, 0, 1, 365)),   # Plus complexe 
    ((2, 1, 2), (1, 0, 1, 12)),   # Plus complexe 
    ((2, 1, 2), (1, 0, 1, 12)),   # Plus complexe 
    ((2, 1, 2), (1, 1, 1, 12)),   # Plus complexe 
    ((2, 1, 2), (1, 1, 1, 365)),   # Plus complexe 
    ((2, 1, 2), (1, 1, 1, 5)),   # Plus complexe 
    
]

resultats_sarima = {}

for (p,d,q), (P,D,Q,s) in ordres_sarima:
    try:
        print(f"\nEntraînement SARIMA({p},{d},{q})({P},{D},{Q},{s})...")
        
        model = ARIMA(y_train_scaled, 
                     order=(p,d,q), 
                     seasonal_order=(P,D,Q,s))
        fitted_model = model.fit()
        
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # Test des résidus
        residus = fitted_model.resid
        ljung_test = sm.stats.diagnostic.acorr_ljungbox(residus, lags=15, return_df=True)
        ljung_pvalue = ljung_test['lb_pvalue'].iloc[-1]
        
        resultats_sarima[(p,d,q,P,D,Q,s)] = {
            'model': fitted_model,
            'aic': aic,
            'bic': bic,
            'ljung_pvalue': ljung_pvalue
        }
        
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Ljung-Box p-value: {ljung_pvalue:.4f}")
        
    except Exception as e:
        print(f"  ERREUR: {e}")

# %%
# ÉTAPE 3: OPTIMISATION FINE DU MODÈLE GARCH
print("\n=== OPTIMISATION MODÈLE GARCH ===")

# Test avec différentes distributions et ordres
from arch import arch_model

# Préparation des rendements standardisés
rendements_scaled = y_train_scaled.diff().dropna()

configurations_garch = [
    # (p, q, distribution)
    (1, 1, 'normal'),
    (1, 2, 'normal'),
    (2, 1, 'normal'),
    (1, 1, 't'),        # Distribution t-Student
    (1, 2, 't'),        # Plus robuste aux outliers
    (1, 1, 'skewt'),    # Distribution t-Student asymétrique
]

resultats_garch_optimises = {}

for p, q, dist in configurations_garch:
    try:
        print(f"\nEntraînement GARCH({p},{q}) - {dist}...")
        
        garch_model = arch_model(rendements_scaled * 100,  # Scaling pour convergence
                                vol='Garch', 
                                p=p, q=q,
                                dist=dist)
        
        garch_fitted = garch_model.fit(disp='off')
        
        # Calcul du critère d'information
        aic = garch_fitted.aic
        bic = garch_fitted.bic
        
        # Test des résidus standardisés
        residus_std = garch_fitted.std_resid
        ljung_test_garch = sm.stats.diagnostic.acorr_ljungbox(residus_std**2, lags=10, return_df=True)
        ljung_pvalue_garch = ljung_test_garch['lb_pvalue'].iloc[-1]
        
        resultats_garch_optimises[(p, q, dist)] = {
            'model': garch_fitted,
            'aic': aic,
            'bic': bic,
            'ljung_pvalue': ljung_pvalue_garch,
            'log_likelihood': garch_fitted.loglikelihood
        }
        
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Log-Likelihood: {garch_fitted.loglikelihood:.2f}")
        print(f"  Ljung-Box résidus² p-value: {ljung_pvalue_garch:.4f}")
        
    except Exception as e:
        print(f"  ERREUR: {e}")

# %%
# ÉTAPE 4: SÉLECTION DES MEILLEURS MODÈLES
print("\n=== SÉLECTION DES MEILLEURS MODÈLES ===")

# Meilleur ARIMA/SARIMA
print(" CLASSEMENT ARIMA/SARIMA (par AIC):")
tous_arima = {**resultats_arima_ameliores, **resultats_sarima}
arima_tries = sorted(tous_arima.items(), key=lambda x: x[1]['aic'])

for i, (ordre, resultats) in enumerate(arima_tries[:5], 1):
    ljung_status = "" if resultats['ljung_pvalue'] > 0.05 else "❌"
    print(f"{i}. ARIMA{ordre}: AIC={resultats['aic']:.2f}, Ljung-Box={ljung_status}")

meilleur_arima_ordre = arima_tries[0][0]
meilleur_arima_model = arima_tries[0][1]['model']
print(f"\n MEILLEUR: ARIMA{meilleur_arima_ordre}")

# Meilleur GARCH
print("\n CLASSEMENT GARCH (par AIC):")
garch_tries = sorted(resultats_garch_optimises.items(), key=lambda x: x[1]['aic'])

for i, (config, resultats) in enumerate(garch_tries[:5], 1):
    ljung_status = "" if resultats['ljung_pvalue'] > 0.05 else "❌"
    print(f"{i}. GARCH{config}: AIC={resultats['aic']:.2f}, Ljung-Box={ljung_status}")

if garch_tries:
    meilleur_garch_config = garch_tries[0][0]
    meilleur_garch_model = garch_tries[0][1]['model']
    print(f"\n MEILLEUR: GARCH{meilleur_garch_config}")

# %%
# ÉTAPE 5: DIAGNOSTIC APPROFONDI DES MEILLEURS MODÈLES
print("\n=== DIAGNOSTIC APPROFONDI ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# DIAGNOSTIC ARIMA
residus_arima = meilleur_arima_model.resid

# 1. Résidus dans le temps
axes[0,0].plot(residus_arima, alpha=0.7, color='blue')
axes[0,0].set_title('Résidus ARIMA dans le Temps')
axes[0,0].axhline(0, color='red', linestyle='--')
axes[0,0].grid(True, alpha=0.3)

# 2. Distribution des résidus
axes[0,1].hist(residus_arima, bins=50, alpha=0.7, color='skyblue', density=True)
# Superposition normale théorique
x = np.linspace(residus_arima.min(), residus_arima.max(), 100)
axes[0,1].plot(x, stats.norm.pdf(x, residus_arima.mean(), residus_arima.std()), 
               'r-', linewidth=2, label='Normale théorique')
axes[0,1].set_title('Distribution Résidus ARIMA')
axes[0,1].legend()

# 3. Q-Q Plot
stats.probplot(residus_arima, dist="norm", plot=axes[0,2])
axes[0,2].set_title('Q-Q Plot Résidus ARIMA')

# DIAGNOSTIC GARCH (si disponible)
if garch_tries:
    residus_garch = meilleur_garch_model.std_resid
    
    # 4. Résidus standardisés GARCH
    axes[1,0].plot(residus_garch, alpha=0.7, color='purple')
    axes[1,0].set_title('Résidus Standardisés GARCH')
    axes[1,0].axhline(0, color='red', linestyle='--')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Volatilité conditionnelle
    volatilite = meilleur_garch_model.conditional_volatility
    axes[1,1].plot(volatilite, color='orange', linewidth=1)
    axes[1,1].set_title('Volatilité Conditionnelle GARCH')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. ACF résidus² standardisés
    plot_acf(residus_garch**2, lags=20, ax=axes[1,2], title='ACF Résidus² Standardisés')

plt.tight_layout()
plt.show()

# %%
# ÉTAPE 6: PRÉPARATION DU FORECASTING
print("\n=== PRÉPARATION FORECASTING ===")

# Fonction de prédiction combinée ARIMA + GARCH
def forecast_arima_garch(model_arima, model_garch, scaler, steps=7):
    """
    Forecasting combiné ARIMA + GARCH
    """
    # Prédiction moyenne avec ARIMA
    forecast_arima = model_arima.forecast(steps=steps)
    
    # Prédiction volatilité avec GARCH (si disponible)
    if model_garch is not None:
        forecast_garch = model_garch.forecast(horizon=steps)
        volatilite_pred = forecast_garch.variance.values[-1]  # Dernière prédiction
    else:
        volatilite_pred = None
    
    # Retransformation à l'échelle originale
    forecast_original = scaler.inverse_transform(forecast_arima.values.reshape(-1, 1)).flatten()
    
    return {
        'mean_forecast': forecast_original,
        'volatility_forecast': volatilite_pred,
        'forecast_scaled': forecast_arima.values
    }

# Test de prédiction sur les 7 premiers jours de mars 2025
print(" TEST DE PRÉDICTION - 7 PREMIERS JOURS MARS 2025")

# Dates cibles (7 premiers jours ouvrés de mars 2025)
dates_forecast = pd.date_range(start='2025-03-03', periods=7, freq='B')  # 'B' = Business days
print(f"Dates de prédiction: {dates_forecast[0]} à {dates_forecast[-1]}")

# Valeurs réelles pour comparaison
valeurs_reelles = df_validation.loc[dates_forecast[0]:dates_forecast[-1], 'Besoin']
print(f"Nombre de valeurs réelles disponibles: {len(valeurs_reelles)}")

# Prédiction
if garch_tries:
    predictions = forecast_arima_garch(meilleur_arima_model, meilleur_garch_model, scaler, steps=7)
else:
    predictions = forecast_arima_garch(meilleur_arima_model, None, scaler, steps=7)

print("\n RÉSULTATS PRÉDICTION:")
print(f"Prédictions moyennes: {predictions['mean_forecast']}")
if len(valeurs_reelles) > 0:
    print(f"Valeurs réelles: {valeurs_reelles.values}")
    
    # Calcul métriques de performance
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    n_compare = min(len(predictions['mean_forecast']), len(valeurs_reelles))
    if n_compare > 0:
        mae = mean_absolute_error(valeurs_reelles.values[:n_compare], 
                                predictions['mean_forecast'][:n_compare])
        rmse = np.sqrt(mean_squared_error(valeurs_reelles.values[:n_compare], 
                                        predictions['mean_forecast'][:n_compare]))
        
        print(f"\n MÉTRIQUES PERFORMANCE:")
        print(f"MAE: {mae:,.0f} CFA")
        print(f"RMSE: {rmse:,.0f} CFA")
        print(f"MAPE: {100 * mae / np.mean(valeurs_reelles.values[:n_compare]):.2f}%")

print("\n OPTIMISATION TERMINÉE - PRÊT POUR FORECASTING COMPLET!")

# %%
# %%
# VISUALISATION COMPLÈTE DU FORECASTING
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

print("=== ANALYSE DÉTAILLÉE DU FORECASTING ===")

# Récupération des modèles optimaux
meilleur_sarima = meilleur_arima_model  # SARIMA(1,0,1)(1,0,1,5)
meilleur_garch_model = meilleur_garch_model  # GARCH(1,1,'skewt')

# Dates de prédiction (7 premiers jours ouvrés de mars 2025)
dates_forecast = pd.date_range(start='2025-03-03', periods=7, freq='B')
print(f"Période de prédiction: {dates_forecast[0].strftime('%Y-%m-%d')} à {dates_forecast[-1].strftime('%Y-%m-%d')}")

# Valeurs réelles pour comparaison
valeurs_reelles = df_validation.loc[dates_forecast, 'Besoin'].values
print(f"Valeurs réelles disponibles: {len(valeurs_reelles)} jours")

# %%
# PRÉDICTIONS DÉTAILLÉES JOUR PAR JOUR
print("\n=== PRÉDICTIONS JOUR PAR JOUR ===")

# Prédictions SARIMA avec intervalles de confiance
forecast_sarima = meilleur_sarima.get_forecast(steps=7)
predictions_mean = scaler.inverse_transform(forecast_sarima.predicted_mean.values.reshape(-1, 1)).flatten()
conf_int = forecast_sarima.conf_int()
conf_int_scaled = scaler.inverse_transform(conf_int.values)

# Prédictions GARCH (volatilité)
forecast_garch = meilleur_garch_model.forecast(horizon=7)
volatilite_predictions = forecast_garch.variance.values[-1]

# Création du DataFrame de résultats
resultats_forecast = pd.DataFrame({
    'Date': dates_forecast,
    'Jour_Semaine': [d.strftime('%A') for d in dates_forecast],
    'Prediction_Besoin': predictions_mean,
    'Borne_Inf_95%': conf_int_scaled[:, 0],
    'Borne_Sup_95%': conf_int_scaled[:, 1],
    'Valeur_Reelle': valeurs_reelles,
    'Volatilite_GARCH': [volatilite_predictions] * 7
})

# Calcul des erreurs
resultats_forecast['Erreur_Absolue'] = np.abs(resultats_forecast['Prediction_Besoin'] - resultats_forecast['Valeur_Reelle'])
resultats_forecast['Erreur_Relative_%'] = 100 * resultats_forecast['Erreur_Absolue'] / np.abs(resultats_forecast['Valeur_Reelle'])

# Affichage détaillé
print("\nTABLEAU DÉTAILLÉ DES PRÉDICTIONS:")
print("=" * 100)
for i, row in resultats_forecast.iterrows():
    print(f" {row['Date'].strftime('%Y-%m-%d')} ({row['Jour_Semaine']}):")
    print(f"   Prédiction: {row['Prediction_Besoin']:>15,.0f} CFA")
    print(f"   Réel:       {row['Valeur_Reelle']:>15,.0f} CFA")
    print(f"   Erreur:     {row['Erreur_Absolue']:>15,.0f} CFA ({row['Erreur_Relative_%']:>5.1f}%)")
    print(f"   I.C. 95%:  [{row['Borne_Inf_95%']:>13,.0f} ; {row['Borne_Sup_95%']:>13,.0f}]")
    
    # Interprétation métier
    if row['Prediction_Besoin'] < 0:
        print(f"    INTERPRÉTATION: Jour de RETRAIT net ({abs(row['Prediction_Besoin']):.0f} CFA)")
        print(f"      → Plus de clients retirent que déposent")
    else:
        print(f"    INTERPRÉTATION: Jour de DÉPÔT net ({row['Prediction_Besoin']:.0f} CFA)")
        print(f"      → Plus de clients déposent que retirent")
    print("-" * 80)

# %%
# VISUALISATION GRAPHIQUE COMPLÈTE
fig, axes = plt.subplots(3, 1, figsize=(15, 14))

# GRAPHIQUE 1: Série historique + Prédictions
axes[0].plot(df_train_test.index[-60:], df_train_test['Besoin'].iloc[-60:], 
             color='blue', linewidth=1.5, label='Données Historiques')
axes[0].plot(dates_forecast, predictions_mean, 
             color='red', linewidth=2, marker='o', markersize=6, label='Prédictions SARIMA')
axes[0].plot(dates_forecast, valeurs_reelles, 
             color='green', linewidth=2, marker='s', markersize=6, label='Valeurs Réelles')

# Intervalle de confiance
axes[0].fill_between(dates_forecast, conf_int_scaled[:, 0], conf_int_scaled[:, 1], 
                     color='red', alpha=0.2, label='Intervalle Confiance 95%')

axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Seuil Équilibre')
axes[0].set_title('Prédiction vs Réalité - Besoin Journalier Agence 00001', fontsize=14, pad=15)
axes[0].set_ylabel('Besoin (CFA)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Rotation des dates pour lisibilité
axes[0].tick_params(axis='x', rotation=45)

# GRAPHIQUE 2: Erreurs de prédiction
erreurs = predictions_mean - valeurs_reelles
axes[1].bar(range(7), erreurs, color=['red' if e > 0 else 'blue' for e in erreurs], 
            alpha=0.7, edgecolor='black')
axes[1].set_title('Erreurs de Prédiction par Jour', fontsize=14, pad=15)
axes[1].set_ylabel('Erreur (CFA)', fontsize=12)
axes[1].set_xlabel('Jours de Prédiction', fontsize=12)
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels([f"J{i+1}" for i in range(7)])
axes[1].grid(True, alpha=0.3, axis='y')

# Ajout des valeurs sur les barres
for i, v in enumerate(erreurs):
    axes[1].text(i, v + (max(erreurs) * 0.02 if v > 0 else min(erreurs) * 0.02), 
                f'{v/1e6:.0f}M', ha='center', va='bottom' if v > 0 else 'top', fontsize=9)

# GRAPHIQUE 3: Comparaison valeurs absolues
x_pos = np.arange(7)
width = 0.35

bars1 = axes[2].bar(x_pos - width/2, np.abs(predictions_mean)/1e6, width, 
                    label='Prédictions', color='red', alpha=0.7)
bars2 = axes[2].bar(x_pos + width/2, np.abs(valeurs_reelles)/1e6, width, 
                    label='Valeurs Réelles', color='green', alpha=0.7)

axes[2].set_title('Comparaison Valeurs Absolues (en Millions CFA)', fontsize=14, pad=15)
axes[2].set_ylabel('Valeur Absolue (Millions CFA)', fontsize=12)
axes[2].set_xlabel('Jours de Mars 2025', fontsize=12)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels([d.strftime('%m-%d') for d in dates_forecast])
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

# Ajout des valeurs sur les barres
for bar in bars1:
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}M', ha='center', va='bottom', fontsize=8)
                
for bar in bars2:
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}M', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# %%
# MÉTRIQUES DE PERFORMANCE DÉTAILLÉES
print("\n=== MÉTRIQUES DE PERFORMANCE GLOBALES ===")

# Métriques classiques
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

mae = mean_absolute_error(valeurs_reelles, predictions_mean)
rmse = np.sqrt(mean_squared_error(valeurs_reelles, predictions_mean))
try:
    mape = mean_absolute_percentage_error(valeurs_reelles, predictions_mean)
except:
    mape = np.mean(np.abs((valeurs_reelles - predictions_mean) / valeurs_reelles))

print(f"MAE (Erreur Absolue Moyenne):     {mae:>15,.0f} CFA")
print(f"RMSE (Racine Erreur Quadratique): {rmse:>15,.0f} CFA")  
print(f"MAPE (Erreur Absolue Relative):   {mape*100:>15.2f} %")

# Métriques personnalisées pour le contexte bancaire
correct_direction = sum((predictions_mean > 0) == (valeurs_reelles > 0))
accuracy_direction = correct_direction / len(predictions_mean) * 100

print(f"\nMÉTRIQUES MÉTIER:")
print(f"Précision Direction (Retrait/Dépôt): {accuracy_direction:>10.1f} %")
print(f"Prédictions dans I.C. 95%:        {sum((valeurs_reelles >= conf_int_scaled[:, 0]) & (valeurs_reelles <= conf_int_scaled[:, 1]))}/7")

# Analyse par jour de la semaine
print(f"\nANALYSE PAR JOUR DE LA SEMAINE:")
for jour in resultats_forecast['Jour_Semaine'].unique():
    mask = resultats_forecast['Jour_Semaine'] == jour
    erreur_moy = resultats_forecast[mask]['Erreur_Relative_%'].mean()
    print(f"{jour:>10}: Erreur moyenne = {erreur_moy:>6.1f}%")

# %%
# RECOMMANDATIONS MÉTIER
print("\n=== RECOMMANDATIONS POUR L'AGENCE ===")

# Analyse des patterns détectés
jours_retrait_net = resultats_forecast[resultats_forecast['Prediction_Besoin'] < 0]
jours_depot_net = resultats_forecast[resultats_forecast['Prediction_Besoin'] > 0]

print(f"\nPATTERNS DÉTECTÉS:")
print(f"• Jours de retrait net prédits: {len(jours_retrait_net)}/7")
print(f"• Jours de dépôt net prédits:   {len(jours_depot_net)}/7")

if len(jours_retrait_net) > 0:
    retrait_max = jours_retrait_net.loc[jours_retrait_net['Prediction_Besoin'].idxmin()]
    print(f"• Plus fort retrait prédit: {retrait_max['Date'].strftime('%A %d/%m')} ({abs(retrait_max['Prediction_Besoin']):.0f} CFA)")

if len(jours_depot_net) > 0:
    depot_max = jours_depot_net.loc[jours_depot_net['Prediction_Besoin'].idxmax()]
    print(f"• Plus fort dépôt prédit: {depot_max['Date'].strftime('%A %d/%m')} ({depot_max['Prediction_Besoin']:.0f} CFA)")

print(f"\nRECOMMANDATIONS OPÉRATIONNELLES:")
print(f"1. GESTION LIQUIDITÉ:")
for i, row in resultats_forecast.iterrows():
    if row['Prediction_Besoin'] < -500e6:  # Retrait > 500M
        print(f"     {row['Date'].strftime('%A %d/%m')}: Prévoir liquidités supplémentaires ({abs(row['Prediction_Besoin'])/1e6:.0f}M CFA)")
    elif row['Prediction_Besoin'] > 500e6:  # Dépôt > 500M  
        print(f"    {row['Date'].strftime('%A %d/%m')}: Excédent prévu - envisager placement ({row['Prediction_Besoin']/1e6:.0f}M CFA)")

print(f"\n2. FIABILITÉ DU MODÈLE:")
if mape < 0.3:  # 30%
    print(f"  Modèle TRÈS FIABLE (MAPE = {mape*100:.1f}%)")
elif mape < 0.5:  # 50%
    print(f"  Modèle MOYENNEMENT FIABLE (MAPE = {mape*100:.1f}%)")
else:
    print(f"  Modèle PEU FIABLE (MAPE = {mape*100:.1f}%) - Révision nécessaire")

print(f"\n3. INTERVALLES DE CONFIANCE:")
largeur_ic_moyenne = np.mean(conf_int_scaled[:, 1] - conf_int_scaled[:, 0])
print(f"   Largeur moyenne I.C. 95%: {largeur_ic_moyenne/1e6:.0f}M CFA")
if largeur_ic_moyenne > 1e9:  # > 1 milliard
    print(f"   Intervalles larges - Incertitude élevée")
else:
    print(f"   Intervalles acceptables")

print(f"\n=== FORECASTING TERMINÉ ===")

# %%
# %%
# VALIDATION COMPLÈTE DES MODÈLES - TESTS RIGOUREUX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from arch.unitroot import ADF
import warnings
warnings.filterwarnings('ignore')

print("=== VALIDATION RIGOUREUSE DES MODÈLES ===")

# %%
# TEST 1: VALIDATION DES RÉSIDUS SARIMA
print("\n1. VALIDATION RÉSIDUS SARIMA")
print("=" * 50)

# Récupération des résidus du meilleur modèle SARIMA
residus_sarima = meilleur_arima_model.resid
residus_sarima_clean = residus_sarima.dropna()

# Test 1.1: Stationnarité des résidus
print("\n1.1 TEST DE STATIONNARITÉ (ADF)")
adf_residus = ADF(residus_sarima_clean)
print(f"ADF Statistic: {adf_residus.stat:.4f}")
print(f"p-value: {adf_residus.pvalue:.6f}")
print(f"Résidus stationnaires: {'OUI' if adf_residus.pvalue < 0.05 else 'NON'}")

# Test 1.2: Autocorrélation des résidus (Ljung-Box)
print("\n1.2 TEST D'AUTOCORRÉLATION (Ljung-Box)")
ljung_result = acorr_ljungbox(residus_sarima_clean, lags=20, return_df=True)
ljung_pvalues = ljung_result['lb_pvalue']
autocorr_detected = (ljung_pvalues < 0.05).sum()
print(f"Nombre de lags avec autocorrélation significative: {autocorr_detected}/20")
print(f"p-value globale (lag 20): {ljung_pvalues.iloc[-1]:.4f}")
print(f"Résidus non-autocorrélés: {'OUI' if ljung_pvalues.iloc[-1] > 0.05 else 'NON'}")

# Test 1.3: Normalité des résidus
print("\n1.3 TEST DE NORMALITÉ")
# Jarque-Bera
jb_stat, jb_pvalue = jarque_bera(residus_sarima_clean)
print(f"Jarque-Bera p-value: {jb_pvalue:.6f}")
print(f"Résidus normaux (JB): {'OUI' if jb_pvalue > 0.05 else 'NON'}")

# Shapiro-Wilk (échantillon)
sample_size = min(5000, len(residus_sarima_clean))
shapiro_stat, shapiro_pvalue = stats.shapiro(residus_sarima_clean.iloc[-sample_size:])
print(f"Shapiro-Wilk p-value: {shapiro_pvalue:.6f}")
print(f"Résidus normaux (SW): {'OUI' if shapiro_pvalue > 0.05 else 'NON'}")

# Test 1.4: Homoscédasticité (ARCH effects)
print("\n1.4 TEST D'HOMOSCÉDASTICITÉ (ARCH)")
try:
    arch_stat, arch_pvalue, _, _ = het_arch(residus_sarima_clean, nlags=5)
    print(f"ARCH test p-value: {arch_pvalue:.6f}")
    print(f"Homoscédasticité: {'OUI' if arch_pvalue > 0.05 else 'NON (effets ARCH présents)'}")
    arch_effects = arch_pvalue < 0.05
except:
    print("Test ARCH impossible - utilisation alternative")
    ljung_r2 = acorr_ljungbox(residus_sarima_clean**2, lags=10, return_df=True)
    arch_pvalue = ljung_r2['lb_pvalue'].iloc[-1]
    print(f"Ljung-Box sur résidus² p-value: {arch_pvalue:.6f}")
    arch_effects = arch_pvalue < 0.05
    print(f"Effets ARCH: {'OUI' if arch_effects else 'NON'}")

# %%
# TEST 2: VALIDATION MODÈLE GARCH
print("\n\n2. VALIDATION MODÈLE GARCH")
print("=" * 50)

if 'meilleur_garch_model' in locals():
    # Résidus standardisés GARCH
    residus_garch = meilleur_garch_model.std_resid
    
    # Test 2.1: Autocorrélation des résidus standardisés
    print("\n2.1 AUTOCORRÉLATION RÉSIDUS STANDARDISÉS")
    ljung_garch = acorr_ljungbox(residus_garch, lags=15, return_df=True)
    print(f"Ljung-Box résidus std p-value: {ljung_garch['lb_pvalue'].iloc[-1]:.4f}")
    print(f"Pas d'autocorrélation: {'OUI' if ljung_garch['lb_pvalue'].iloc[-1] > 0.05 else 'NON'}")
    
    # Test 2.2: Autocorrélation des résidus² standardisés (test homoscédasticité)
    print("\n2.2 HOMOSCÉDASTICITÉ RÉSIDUS GARCH")
    ljung_garch_r2 = acorr_ljungbox(residus_garch**2, lags=15, return_df=True)
    print(f"Ljung-Box résidus²std p-value: {ljung_garch_r2['lb_pvalue'].iloc[-1]:.4f}")
    print(f"Homoscédasticité atteinte: {'OUI' if ljung_garch_r2['lb_pvalue'].iloc[-1] > 0.05 else 'NON'}")
    
    # Test 2.3: Normalité des résidus standardisés
    print("\n2.3 NORMALITÉ RÉSIDUS GARCH")
    shapiro_garch = stats.shapiro(residus_garch[:5000] if len(residus_garch) > 5000 else residus_garch)
    print(f"Shapiro-Wilk p-value: {shapiro_garch[1]:.6f}")
    print(f"Résidus normaux: {'OUI' if shapiro_garch[1] > 0.05 else 'NON'}")
    
    garch_valide = (ljung_garch['lb_pvalue'].iloc[-1] > 0.05 and 
                   ljung_garch_r2['lb_pvalue'].iloc[-1] > 0.05)
else:
    print("Modèle GARCH non disponible")
    garch_valide = False

# %%
# TEST 3: ANALYSE DE LA SÉRIE TEMPORELLE ORIGINALE
print("\n\n3. ANALYSE APPROFONDIE DE LA SÉRIE")
print("=" * 50)

# Test de ruptures structurelles
from statsmodels.stats.diagnostic import breaks_cusumolsresid, breaks_hansen
print("\n3.1 TESTS DE RUPTURES STRUCTURELLES")

# Utilisation de la série train/test
serie_analyse = df_train_test['Besoin'].dropna()

# Test CUSUM pour détecter les ruptures
try:
    cusum_stat, cusum_pvalue = breaks_cusumolsresid(serie_analyse.values)
    print(f"CUSUM test p-value: {cusum_pvalue:.4f}")
    print(f"Stabilité structurelle: {'OUI' if cusum_pvalue > 0.05 else 'NON - Ruptures détectées'}")
except:
    print("Test CUSUM non applicable - série complexe")

# Analyse de la variance par période
print("\n3.2 ANALYSE DE LA VARIANCE PAR PÉRIODE")
# Division en quartiles temporels
n_points = len(serie_analyse)
quartile_size = n_points // 4

variances_quartiles = []
for i in range(4):
    start_idx = i * quartile_size
    end_idx = (i + 1) * quartile_size if i < 3 else n_points
    quartile_data = serie_analyse.iloc[start_idx:end_idx]
    var_quartile = quartile_data.var()
    variances_quartiles.append(var_quartile)
    print(f"Quartile {i+1}: Variance = {var_quartile:.2e}")

# Test d'homogénéité des variances (Levene)
from scipy.stats import levene
levene_stat, levene_pvalue = levene(*[serie_analyse.iloc[i*quartile_size:(i+1)*quartile_size if i<3 else n_points] 
                                     for i in range(4)])
print(f"Test de Levene p-value: {levene_pvalue:.4f}")
print(f"Homogénéité des variances: {'OUI' if levene_pvalue > 0.05 else 'NON'}")

# %%
# TEST 4: ALTERNATIVES AU GARCH CLASSIQUE
print("\n\n4. ÉVALUATION ALTERNATIVES GARCH")
print("=" * 50)

from arch import arch_model

# Préparation des données
rendements = df_train_test['Besoin'].diff().dropna()
rendements_centres = (rendements - rendements.mean())

# Test de différentes variantes GARCH
variantes_arch = [
    ('GARCH', {'vol': 'Garch', 'p': 1, 'q': 1, 'dist': 'skewt'}),
    ('EGARCH', {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewt'}),
    ('GJR-GARCH', {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewt'}),
    ('TARCH', {'vol': 'GARCH', 'p': 2, 'q': 1, 'dist': 't'}),
]

resultats_variantes = {}

for nom, params in variantes_arch:
    try:
        print(f"\nTest {nom}...")
        model = arch_model(rendements_centres, **params, rescale=False)
        fitted = model.fit(disp='off')
        
        # Métriques
        aic = fitted.aic
        bic = fitted.bic
        
        # Test des résidus
        std_resid = fitted.std_resid
        ljung_resid = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
        ljung_pval = ljung_resid['lb_pvalue'].iloc[-1]
        
        resultats_variantes[nom] = {
            'aic': aic,
            'bic': bic,
            'ljung_pvalue': ljung_pval,
            'model': fitted
        }
        
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Ljung-Box p-value: {ljung_pval:.4f}")
        print(f"  Homoscédasticité: {'OUI' if ljung_pval > 0.05 else 'NON'}")
        
    except Exception as e:
        print(f"  Erreur: {str(e)[:50]}...")

# %%
# TEST 5: VALIDATION CROISÉE TEMPORELLE
print("\n\n5. VALIDATION CROISÉE TEMPORELLE")
print("=" * 50)

def validation_croisee_temporelle(serie, model_func, horizons=[1, 3, 5, 7]):
    """
    Validation croisée avec horizon de prédiction variable
    """
    resultats = {}
    
    # Utilisation des 6 derniers mois pour validation
    split_point = len(serie) - 120  # ~6 mois
    train_data = serie.iloc[:split_point]
    test_data = serie.iloc[split_point:]
    
    for h in horizons:
        erreurs = []
        
        # Fenêtre glissante
        for i in range(0, len(test_data) - h, 5):  # Pas de 5 jours
            train_subset = pd.concat([train_data, test_data.iloc[:i]])
            
            try:
                # Entraînement du modèle
                model = model_func(train_subset)
                
                # Prédiction
                if hasattr(model, 'get_forecast'):
                    forecast = model.get_forecast(steps=h)
                    pred = forecast.predicted_mean.iloc[-1]
                else:
                    pred = model.forecast(steps=h)[-1]
                
                # Valeur réelle
                if i + h < len(test_data):
                    reel = test_data.iloc[i + h]
                    erreur = abs(pred - reel) / abs(reel)
                    erreurs.append(erreur)
                    
            except Exception as e:
                continue
        
        if erreurs:
            resultats[h] = {
                'mape_moyen': np.mean(erreurs),
                'mape_std': np.std(erreurs),
                'n_predictions': len(erreurs)
            }
            
            print(f"Horizon {h} jours:")
            print(f"  MAPE moyen: {np.mean(erreurs)*100:.2f}%")
            print(f"  Écart-type: {np.std(erreurs)*100:.2f}%")
            print(f"  Nb prédictions: {len(erreurs)}")
    
    return resultats

# Fonction simplifiée pour ARIMA
def fit_simple_arima(data):
    from statsmodels.tsa.arima.model import ARIMA
    # Standardisation
    scaled_data = (data - data.mean()) / data.std()
    model = ARIMA(scaled_data, order=(1, 0, 1))
    return model.fit()

print("Lancement validation croisée...")
validation_results = validation_croisee_temporelle(df_train_test['Besoin'], fit_simple_arima)

# %%
# SYNTHÈSE ET RECOMMANDATIONS
print("\n\n=== SYNTHÈSE CRITIQUE ===")
print("=" * 50)

# Score de validation global
score_sarima = 0
if ljung_pvalues.iloc[-1] > 0.05: score_sarima += 25
if shapiro_pvalue > 0.001: score_sarima += 25  # Critère moins strict pour normalité
if not arch_effects: score_sarima += 25
if 'cusum_pvalue' in locals() and cusum_pvalue > 0.05: score_sarima += 25

print(f"\nSCORE DE VALIDATION SARIMA: {score_sarima}/100")

if score_sarima < 50:
    print(" MODÈLE SARIMA NON VALIDÉ - Hypothèses violées")
elif score_sarima < 75:
    print(" MODÈLE SARIMA PARTIELLEMENT VALIDÉ - Améliorations nécessaires")
else:
    print(" MODÈLE SARIMA VALIDÉ")

if 'garch_valide' in locals():
    print(f"MODÈLE GARCH: {' VALIDÉ' if garch_valide else '❌ NON VALIDÉ'}")

print(f"\nRECOMMANDATIONS:")

if score_sarima < 75 or not garch_valide:
    print("1. MODÈLES ALTERNATIFS À TESTER:")
    print("   - LSTM/GRU pour capturer les non-linéarités")
    print("   - Prophet pour saisonnalités multiples") 
    print("   - VAR pour relations multi-variables")
    print("   - Modèles hybrides ARIMA-ML")
    
    print("\n2. PRÉTRAITEMENT AVANCÉ:")
    print("   - Décomposition STL plus fine")
    print("   - Détection/traitement outliers robuste")
    print("   - Transformation Box-Cox")
    print("   - Variables exogènes (jours fériés, événements)")

print(f"\n3. HORIZON DE PRÉDICTION OPTIMAL:")
if validation_results:
    best_horizon = min(validation_results.keys(), key=lambda k: validation_results[k]['mape_moyen'])
    print(f"   Horizon recommandé: {best_horizon} jour(s)")
    print(f"   MAPE attendu: {validation_results[best_horizon]['mape_moyen']*100:.1f}%")
else:
    print("   Recommandation: Limiter à 1-3 jours maximum")

print(f"\n4. SEUILS D'ACCEPTABILITÉ:")
print(f"   - MAPE < 15% : Excellent")
print(f"   - MAPE 15-25% : Acceptable")
print(f"   - MAPE > 25% : Inacceptable (votre cas: 28.1%)")


