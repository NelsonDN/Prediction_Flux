# %% [markdown]
# L'objectif de ce notebook est de vous présenter une application pratique de modélisation d'une série temporelle avec le modèle ARIMA. Les données utilisées sont les données du package statsmodels de python.
# 

# %% [markdown]
# ## Importer les librairies

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# %% [markdown]
# Charger les données:

# %%
data = sm.datasets.get_rdataset("AirPassengers", package="datasets")
df = data.data
df["Annee_mois"] = pd.date_range(start="1949-01", periods=len(df), freq="ME")  # Conversion en série temporelle
df["Annee"]=df["Annee_mois"].dt.year
df.set_index("Annee_mois", inplace=True)

# %% [markdown]
# ## Visualisation

# %%
df.head()

# %%
df.tail(6)

# %%
df.isnull().sum()

# %%
df_passager = df[['value']]

# %%
df_passager.tail(2)

# %%
plt.figure(figsize=(10,5))
plt.plot(df_passager, label="Nombre de Passagers", marker='o', linestyle='-', color='grey')
plt.title("Evolution du nombre de passager aériens")
plt.xlabel("Temps")
plt.ylabel("Nombre de passager")
plt.show()

# %% [markdown]
# On constate clairement que notre série a une tendance linéaire et saisonnière. Ce modèle est-il additif ou multiplicatif?
# 
# ## On va le vérifier en utilisant la méthode de buys-ballot:

# %%
df_buys_ballot= df.groupby("Annee")["value"].agg(Moyenne='mean',Ecart_type='std').reset_index() # moyenne et variance sur une période

# %%
df_buys_ballot

# %% [markdown]
# Nuage de points:

# %%
plt.figure(figsize=(10,5))
sns.scatterplot(x=df_buys_ballot["Moyenne"],y=df_buys_ballot["Ecart_type"], color="grey")
plt.xlabel("Moyenne annuelle du nombre de passager")
plt.ylabel("Ecart-type sur une période")
plt.title("Relation entre écart-type et moyenne sur une période")
plt.show()

# %% [markdown]
# Estimer le modèle de régression linéaire:

# %%
X=df_buys_ballot['Moyenne']
y=df_buys_ballot['Ecart_type']
X=sm.add_constant(X) # modèle avec constante
model = sm.OLS(y,X).fit()

# %%
print(model.summary())

# %% [markdown]
# Ajouter au graphe précedent la droite de régression:

# %%
df_buys_ballot['Prediction_ecart_type']=model.predict(X)

# %%
plt.figure(figsize=(10,5))
sns.scatterplot(x=df_buys_ballot["Moyenne"],y=df_buys_ballot["Ecart_type"], color="grey")
plt.xlabel("Moyenne annuelle du nombre de passager")
plt.ylabel("Ecart-type sur une période")
plt.title("Relation entre écart-type et moyenne sur une période")

plt.plot(df_buys_ballot["Moyenne"],df_buys_ballot["Prediction_ecart_type"], color="red", label="regression_linéaire")
plt.legend()
plt.show()

# %% [markdown]
# ## Décomposer la série temporelle

# %%


# Décomposer la série temporelle
decomposition = seasonal_decompose(df_passager, model="multiplicatif", period=12)  #  'period' est le nombre de données dans l'année

# Extraire les composantes
tendance = decomposition.trend
saison = decomposition.seasonal
Bruit = decomposition.resid

# Afficher les résultats
plt.figure(figsize=(8, 8))

plt.subplot(411)
plt.plot(df_passager, label="Série Originale", color="grey")
plt.legend()

plt.subplot(412)
plt.plot(tendance, label="Tendance", color="red")
plt.legend()

plt.subplot(413)
plt.plot(saison, label="Saisonnalité", color="green")
plt.legend()

plt.subplot(414)
plt.plot(Bruit, label="Résidus", color="purple")
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# Vérifier l'autocorrélation:

# %%
bruit = Bruit.dropna()

# %%
# Vérifier l'autocorrélation des résidus avec ACF
fig, ax = plt.subplots(figsize=(8, 4))
sm.graphics.tsa.plot_acf(bruit, lags=20, ax=ax)
plt.title("Autocorrélation des résidus")
plt.show()

# %%
# Vérifier l'autocorrélation des résidus avec PACF
fig, ax = plt.subplots(figsize=(8, 4))
sm.graphics.tsa.plot_pacf(bruit, lags=20, ax=ax)
plt.title("Autocorrélation partielle des résidus")
plt.show()

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox

# Test de Ljung-Box
ljung_box_test = acorr_ljungbox(bruit, lags=[20], return_df=True)

print(ljung_box_test)

# %%
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(bruit)
print(f"Statistique de Durbin-Watson : {dw_stat}")

# %% [markdown]
# ## Transformer la série pour avoir un modèle additif

# %%
def test_stationnarite(series):
    result = adfuller(series)
    print(f"Statistique de test : {result[0]}")
    print(f"P-valeur : {result[1]}")
    print("La série est stationnaire" if result[1] < 0.05 else "La série n'est PAS stationnaire")

# %%
df_passager["passager_ln"] = np.log(df_passager["value"])

# %%
plt.figure(figsize=(10,5))
plt.plot(df_passager['passager_ln'], label="log du Nombre de Passagers", marker='o', linestyle='-', color='grey')
plt.title("Evolution du log du nombre de passager aériens")
plt.xlabel("Temps")
#plt.ylabel("Nombre de passager")
plt.show()

# %% [markdown]
# ACF:

# %%
y = df_passager['passager_ln']
s, nlag = 12, 36
# Autocorrélation simple
fig, axe = plt.subplots(figsize=(12,6))
plot_acf(y,lags=nlag,ax=axe)
axe.set_xlabel("Lag")
axe.set_ylabel("ACF")
plt.show()

# %%
test_stationnarite(y)

# %% [markdown]
# On commence par tédendancialiser la série (premiière différence)

# %%
df_passager["passager_ln_diff"]=df_passager['passager_ln'].diff(1)

# %%
df_passager.head()

# %%
plt.figure(figsize=(10,5))
plt.plot(df_passager['passager_ln_diff'], label="log du Nombre de Passagers sans tendance", marker='o', linestyle='-', color='navy')
plt.title("Evolution du log du nombre de passager aériens différencié")
plt.xlabel("Temps")
plt.show()

# %% [markdown]
# On représente l'ACF

# %%
y = df_passager['passager_ln_diff'].dropna()
s, nlag = 12, 36
# Autocorrélation simple
fig, axe = plt.subplots(figsize=(12,6))
plot_acf(y,lags=nlag,ax=axe)
axe.set_xlabel("Lag")
axe.set_ylabel("ACF")
plt.show()

# %%
test_stationnarite(y)

# %% [markdown]
# Désaisonnaliser la série

# %%
df_passager["passager_ln_diff_sais"]=df_passager['passager_ln_diff'].diff(12)

# %% [markdown]
# On représente l'ACF

# %%
y = df_passager['passager_ln_diff_sais'].dropna()
s, nlag = 12, 36
# Autocorrélation simple
fig, axe = plt.subplots(figsize=(12,6))
plot_acf(y,lags=nlag,ax=axe)
axe.set_xlabel("Lag")
axe.set_ylabel("ACF")
plt.show()

# %%
test_stationnarite(y)

# %% [markdown]
# ## Etape 0: Vérifier la stationnarité:

# %%
test_stationnarite(y)

# %% [markdown]
# ## Identification du modèle

# %% [markdown]
# Se fait à partir de l'ACF et du PACF: de y=(I-B)(I-B^12)ln(Xt)

# %%
s, nlag = 12, 36
# Autocorrélation simple
fig, axe = plt.subplots(figsize=(10,6))
plot_pacf(y,lags=nlag,ax=axe)
axe.set_xlabel("Lag")
axe.set_ylabel("PACF")
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y, lags=36, ax=ax[0])
plot_pacf(y, lags=36, ax=ax[1])
plt.show()

# %% [markdown]
# Identifier les valeur du modèle $SARIMA(p,d,q)(P,D,Q)_{12}$
#  - Partie non saisonnière:
#      - d=1
#      - p=1
#      - q=1 
# 
#  - Partie saisonière:
#     - D=1
#     - P=1
#     - Q=1 

# %% [markdown]
# ## Etape 2: Estimer les paramètres de notre série
# 

# %%
#Estimation modèle 1:
#Estimationd'un SARIMA(1,1,1)(1,1,1)_{12}
model1 = ARIMA(y,order=(1,1,1),seasonal_order=(1, 1, 1,12)).fit()
print(model1.summary())
#model1_params =extractParams(model1,model_type= "arima")

# %%
#Estimation modèle 2:
#Estimationd'un SARIMA(1,1,1)(1,1,0)_{12}
model2 = ARIMA(y,order=(1,1,1),seasonal_order=(1, 1, 0,12)).fit()
print(model2.summary())

# %% [markdown]
# ## Etape 3:Validation du modèle retenu

# %%
residus = model2.resid

# %%
residus.head()

# %%
plt.plot(residus, color='grey')
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residus, lags=36, ax=ax[0])
plot_pacf(residus, lags=36, ax=ax[1])
plt.show()

# %%
#test de ljung_box

ljung_box_test_resid = acorr_ljungbox(residus, lags=[24], return_df=True)
print(ljung_box_test_resid)

# %%
#test de durbin durbin_watson
dw_stat=durbin_watson(residus)
print(f"Statisque du test de durbin watson:{dw_stat}")

# %% [markdown]
# ## Etape 4: Prédiction 

# %%
prediction_1961 = model2.forecast(steps=12)


