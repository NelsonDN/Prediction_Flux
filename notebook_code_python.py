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

%matplotlib inline

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Librairies importées avec succès !")

# %%
# Import des données de l'agence 00001
df = pd.read_csv('Agence_00001.csv')

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


