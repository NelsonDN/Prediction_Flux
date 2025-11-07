"""
CALCUL DU STOCK MINIMUM PAR AGENCE
À partir du fichier "Evolution des soldes des comptes et des clients"
Méthode : Quantile 5 (Q5) des soldes historiques XAF
AVEC VISUALISATIONS INTÉGRÉES
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def extraire_soldes_agence(filepath_excel, code_agence):
    """
    Extraire tous les soldes XAF d'une agence depuis le fichier Excel
    """
    print(f"\nExtraction soldes pour agence {code_agence}...")
    
    df = pd.read_excel(filepath_excel, sheet_name='Soldes chapitres', header=None)
    
    dates_row = df.iloc[1, 6:]
    
    dates = []
    for i in range(0, len(dates_row), 2):
        if pd.notna(dates_row.iloc[i]):
            dates.append(dates_row.iloc[i])
    
    print(f"  {len(dates)} dates trouvees dans le fichier")
    
    df_data = df.iloc[3:]
    
    ligne_agence = df_data[df_data.iloc[:, 1].astype(str) == str(code_agence)]
    
    if len(ligne_agence) == 0:
        raise ValueError(f"Agence {code_agence} introuvable dans le fichier Excel")
    
    if len(ligne_agence) > 1:
        print(f"  ATTENTION: Plusieurs lignes trouvees pour {code_agence}, prise de la premiere")
    
    ligne = ligne_agence.iloc[0]
    
    nom_agence = ligne.iloc[2]
    print(f"  Nom agence: {nom_agence}")
    
    soldes_xaf = []
    
    for i in range(6, len(ligne), 2):
        valeur = ligne.iloc[i]
        if pd.notna(valeur):
            try:
                soldes_xaf.append(float(valeur))
            except:
                pass
    
    soldes_xaf = np.array(soldes_xaf)
    
    print(f"  {len(soldes_xaf)} observations de soldes XAF extraites")
    
    return {
        'code_agence': code_agence,
        'nom_agence': nom_agence,
        'soldes_xaf': soldes_xaf,
        'dates': dates[:len(soldes_xaf)],
        'nb_observations': len(soldes_xaf)
    }


def calculer_stock_minimum_complet(soldes_xaf):
    """
    Calculer le stock minimum et afficher TOUS les percentiles
    """
    percentiles_valeurs = {
        'P1': np.percentile(soldes_xaf, 1),
        'P5': np.percentile(soldes_xaf, 5),
        'P10': np.percentile(soldes_xaf, 10),
        'P25': np.percentile(soldes_xaf, 25),
        'P50': np.percentile(soldes_xaf, 50),
        'P75': np.percentile(soldes_xaf, 75),
        'P90': np.percentile(soldes_xaf, 90),
        'P95': np.percentile(soldes_xaf, 95),
        'P99': np.percentile(soldes_xaf, 99)
    }
    
    percentiles_millions = {k: v / 1_000_000 for k, v in percentiles_valeurs.items()}
    
    stock_minimum_xaf = percentiles_valeurs['P5']
    stock_minimum_millions = percentiles_millions['P5']
    
    stats = {
        'min': np.min(soldes_xaf) / 1_000_000,
        'max': np.max(soldes_xaf) / 1_000_000,
        'moyenne': np.mean(soldes_xaf) / 1_000_000,
        'ecart_type': np.std(soldes_xaf) / 1_000_000,
        'nb_observations': len(soldes_xaf)
    }
    
    return {
        'stock_minimum': stock_minimum_xaf,
        'stock_minimum_millions': stock_minimum_millions,
        'methode': 'Percentile 5 (P5) - Standard Bale III',
        'percentiles': percentiles_millions,
        'stats': stats
    }


def plot_distribution_agence(data, stock_data, output_dir='graphiques_distributions'):
    """
    Créer un graphique de distribution des soldes pour une agence
    """
    os.makedirs(output_dir, exist_ok=True)
    
    code_agence = data['code_agence']
    soldes_millions = data['soldes_xaf'] / 1_000_000
    percentiles = stock_data['percentiles']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ====================================================================
    # GRAPHIQUE 1 : Histogramme avec percentiles
    # ====================================================================
    
    ax1.hist(soldes_millions, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    colors_p = {
        'P1': 'red', 'P5': 'darkred', 'P10': 'orange', 'P25': 'gold',
        'P50': 'green', 'P75': 'blue', 'P90': 'purple', 'P95': 'magenta', 'P99': 'pink'
    }
    
    for p_name, p_val in percentiles.items():
        ax1.axvline(p_val, color=colors_p[p_name], linestyle='--', linewidth=2, 
                   label=f'{p_name} = {p_val:.1f}M')
    
    ax1.set_xlabel('Solde (Millions FCFA)', fontsize=12)
    ax1.set_ylabel('Frequence', fontsize=12)
    ax1.set_title(f'Distribution des Soldes - Agence {code_agence} ({data["nom_agence"]})', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ====================================================================
    # GRAPHIQUE 2 : Boîte à moustaches avec quartiles
    # ====================================================================
    
    bp = ax2.boxplot([soldes_millions], vert=False, widths=0.5, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='black'),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5))
    
    y_pos = 1
    for p_name in ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']:
        p_val = percentiles[p_name]
        ax2.plot(p_val, y_pos, 'o', color=colors_p[p_name], markersize=10, zorder=10)
        ax2.text(p_val, y_pos + 0.15, p_name, ha='center', fontsize=9, fontweight='bold')
    
    q1 = percentiles['P25']
    q2 = percentiles['P50']
    q3 = percentiles['P75']
    min_val = np.min(soldes_millions)
    max_val = np.max(soldes_millions)
    
    ax2.axvspan(min_val, q1, alpha=0.1, color='red', label='Q1 (25% plus bas)')
    ax2.axvspan(q1, q2, alpha=0.1, color='yellow', label='Q2 (25% suivants)')
    ax2.axvspan(q2, q3, alpha=0.1, color='lightgreen', label='Q3 (25% suivants)')
    ax2.axvspan(q3, max_val, alpha=0.1, color='blue', label='Q4 (25% plus hauts)')
    
    p5_val = percentiles['P5']
    ax2.axvline(p5_val, color='darkred', linestyle='-', linewidth=3, 
               label=f'Stock Minimum (P5) = {p5_val:.1f}M', zorder=5)
    
    ax2.set_xlabel('Solde (Millions FCFA)', fontsize=12)
    ax2.set_title('Boite a Moustaches avec Percentiles et Quartiles', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)
    
    ax2.text(p5_val, 0.5, f'<- STOCK MINIMUM\n(Protection 95%)', 
            fontsize=10, color='darkred', fontweight='bold', ha='right')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'{code_agence}_distribution_percentiles.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Graphique sauvegarde: {filename}")
    plt.close()


def plot_comparaison_toutes_agences(filepath_csv='stocks_minimum_agences.csv', 
                                    output_dir='graphiques_distributions'):
    """
    Créer un graphique comparatif de tous les stocks minimum
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(filepath_csv)
    df = df.sort_values('Stock_Minimum_Millions')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['Stock_Minimum_Millions'].abs(), color='steelblue', edgecolor='black')
    
    for i in range(5):
        bars[i].set_color('darkred')
        bars[-(i+1)].set_color('darkgreen')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Code_Agence']}\n{row['Nom_Agence'][:20]}" 
                        for _, row in df.iterrows()], fontsize=9)
    ax.set_xlabel('Stock Minimum (Millions FCFA, valeur absolue)', fontsize=12)
    ax.set_title('Comparaison Stocks Minimum - Toutes Agences (P5)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    red_patch = mpatches.Patch(color='darkred', label='5 stocks les plus eleves')
    green_patch = mpatches.Patch(color='darkgreen', label='5 stocks les plus bas')
    blue_patch = mpatches.Patch(color='steelblue', label='Autres agences')
    ax.legend(handles=[red_patch, green_patch, blue_patch], loc='lower right')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'comparaison_stocks_minimum_toutes_agences.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegarde: {filename}")
    plt.close()


def calculer_stocks_minimum_toutes_agences(filepath_excel, liste_agences):
    """
    Calculer le stock minimum pour toutes les agences avec visualisations
    """
    print("\n" + "="*80)
    print("CALCUL STOCKS MINIMUM - TOUTES AGENCES")
    print("Methode : Percentile 5 (P5) - Standard international Bale III")
    print("="*80)
    
    resultats = []
    
    for code_agence in liste_agences:
        try:
            data = extraire_soldes_agence(filepath_excel, code_agence)
            stock_data = calculer_stock_minimum_complet(data['soldes_xaf'])
            
            resultats.append({
                'Code_Agence': code_agence,
                'Nom_Agence': data['nom_agence'],
                'Stock_Minimum_XAF': stock_data['stock_minimum'],
                'Stock_Minimum_Millions': round(stock_data['stock_minimum_millions'], 2),
                'Methode': stock_data['methode'],
                'Nb_Observations': stock_data['stats']['nb_observations'],
                'Min_M': round(stock_data['stats']['min'], 2),
                'P1_M': round(stock_data['percentiles']['P1'], 2),
                'P5_M': round(stock_data['percentiles']['P5'], 2),
                'P10_M': round(stock_data['percentiles']['P10'], 2),
                'P25_M': round(stock_data['percentiles']['P25'], 2),
                'P50_Mediane_M': round(stock_data['percentiles']['P50'], 2),
                'P75_M': round(stock_data['percentiles']['P75'], 2),
                'P90_M': round(stock_data['percentiles']['P90'], 2),
                'P95_M': round(stock_data['percentiles']['P95'], 2),
                'P99_M': round(stock_data['percentiles']['P99'], 2),
                'Max_M': round(stock_data['stats']['max'], 2),
                'Moyenne_M': round(stock_data['stats']['moyenne'], 2),
                'Ecart_Type_M': round(stock_data['stats']['ecart_type'], 2)
            })
            
            print(f"\n  {code_agence}: Stock minimum (P5) = {stock_data['stock_minimum_millions']:.2f} M")
            print(f"    Percentiles: P1={stock_data['percentiles']['P1']:.2f}M | P10={stock_data['percentiles']['P10']:.2f}M | P25={stock_data['percentiles']['P25']:.2f}M")
            
            # GÉNÉRATION GRAPHIQUE POUR CETTE AGENCE
            plot_distribution_agence(data, stock_data)
            
        except Exception as e:
            print(f"  {code_agence}: ERREUR - {str(e)}")
    
    df_resultats = pd.DataFrame(resultats)
    
    print("\n" + "="*80)
    print("CALCUL TERMINE")
    print("="*80)
    
    return df_resultats


# UTILISATION
if __name__ == "__main__":
    
    AGENCES = [
        '00001', '00005', '00010', '00021', '00024', '00026', '00031',
        '00033', '00034', '00037', '00038', '00039', '00040', '00042',
        '00043', '00045', '00046', '00055', '00056', '00057', '00062',
        '00063', '00064', '00073', '00075', '00079', '00081', '00085',
        '00087'
    ]
    
    FICHIER_EXCEL = 'Evolution des soldes des comptes et des clients.xlsx'
    
    df_stocks = calculer_stocks_minimum_toutes_agences(FICHIER_EXCEL, AGENCES)
    
    output_file = 'stocks_minimum_agences.csv'
    df_stocks.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nFichier sauvegarde: {output_file}")
    
    # GRAPHIQUE COMPARATIF TOUTES AGENCES
    print("\nGeneration graphique comparatif...")
    plot_comparaison_toutes_agences(output_file)
    
    print("\n" + "="*80)
    print("TOUS LES GRAPHIQUES ONT ETE GENERES")
    print("Dossier: graphiques_distributions/")
    print("="*80)