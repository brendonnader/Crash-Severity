# ============================================================
# Florida Non-Motorist Crash Severity Analysis
# FDOT Open Data Hub | 2010-2015 | Full Pipeline
# Team: Brendon Nader, Britney Antoine, Ziya Koroglu, Sagi Braunshtine, Brooklyn Campbell
# Florida Atlantic University # ============================================================
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns from scipy.stats import chi2_contingency, spearmanr, pearsonr from sklearn.linear_model import LogisticRegression from sklearn.metrics import roc_auc_score from sklearn.model_selection import train_test_split import warnings warnings.filterwarnings('ignore')
# ----- 1. LOAD DATA ----df = pd.read_csv('Bike_Ped_Crash_Typing_2010_-_2015.csv', low_memory=False) print(f'Raw records: {len(df):,}')  # 72,102 # ----- 2. PREPROCESSING -----
# Parse hour integer from TimePeriod string (e.g. '13:00-13:59' -> 13) def parse_hour(tp):     try:
        return int(str(tp).split(':')[0])     except:         return np.nan df['HOUR'] = df['TimePeriod'].apply(parse_hour)
# Drop rows missing any key variable df = df.dropna(subset=['HOUR', 'Crash_Severity', 'Light_Condition', 'Weather_Cond df['HOUR'] = df['HOUR'].astype(int) print(f'Clean records: {len(df):,}')  # 38,545
# Assign four time-of-day periods def assign_period(h):
    if h < 6:       return 'Early Morning'   # 12:00 AM - 5:59 AM     elif h < 12:    return 'Morning'          # 6:00 AM  - 11:59 AM     elif h < 18:    return 'Afternoon'        # 12:00 PM - 5:59 PM     else:           return 'Night'            # 6:00 PM  - 11:59 PM df['PERIOD'] = df['HOUR'].apply(assign_period)
# Encode severity as binary (1 = Fatal) and ordinal (0/1/2) df['SEVERE']   = (df['Crash_Severity'] == 'Fatality').astype(int) sev_map        = {'Property Damage Only': 0, 'Injury': 1, 'Fatality': 2} df['SEV_CODE'] = df['Crash_Severity'].map(sev_map)
# Simplify lighting: Daylight (includes Dawn/Dusk), Dark, Other def simplify_light(l):     if any(x in str(l) for x in ['Daylight', 'Dawn', 'Dusk']):
        return 'Daylight'     elif 'Dark' in str(l):
        return 'Dark'     else:         return 'Other' df['LIGHT'] = df['Light_Condition'].apply(simplify_light)
# Simplify weather and encode alcohol as binary (1 = Yes) df['WEATHER_SIMPLE'] = df['Weather_Condition'].apply(     lambda w: 'Clear' if w == 'Clear' else ('Rain' if w == 'Rain' else 'Other')) df['ALCOHOL'] = (df['Alcohol_Related'].str.strip().str.upper() == 'Y').astype(int # ----- 3. DESCRIPTIVE STATISTICS ----period_order = ['Early Morning', 'Morning', 'Afternoon', 'Night']
# Crash counts and severity cross-tabulation ct     = pd.crosstab(df['PERIOD'], df['Crash_Severity']) ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
print('\nSeverity % by time period:') print(ct_pct.loc[period_order].round(2))
# Fatality rate per period print('\nFatality rate per period:') for p in period_order:
    sub  = df[df['PERIOD'] == p]     rate = (sub['Crash_Severity'] == 'Fatality').mean() * 100     print(f'  {p}: {rate:.2f}% ({len(sub):,} crashes)')
# Hourly fatality rates hourly = df.groupby('HOUR').agg(     total=('SEVERE', 'count'),     fatal=('SEVERE', 'sum')
).reset_index()
hourly['fatal_rate'] = (hourly['fatal'] / hourly['total'] * 100).round(2) print('\nHourly fatality rates:') print(hourly[['HOUR', 'total', 'fatal', 'fatal_rate']].to_string(index=False)) # ----- 4. CHI-SQUARE TESTS -----
# Test 1: Time-of-day period vs. crash severity chi2, p, dof, exp = chi2_contingency(ct.loc[period_order]) print(f'\nChi-Square Test 1 (Period vs. Severity):') print(f'  Chi2 = {chi2:.4f}, p = {p:.4e}, df = {dof}')   # 54.92, 4.82e-10, 6 print(f'  Min expected count: {exp.min():.2f}')             # 0.07 print(f'  Result: {"REJECT H0" if p < 0.05 else "Fail to reject H0"}')
# Test 2: Lighting condition vs. crash severity ct_light           = pd.crosstab(df['LIGHT'], df['Crash_Severity']) chi2_l, p_l, dof_l, _ = chi2_contingency(ct_light) print(f'\nChi-Square Test 2 (Lighting vs. Severity):') print(f'  Chi2 = {chi2_l:.4f}, p = {p_l:.4e}, df = {dof_l}')  # 98.09, <0.001 # ----- 5. LOGISTIC REGRESSION ----df_lr = df.dropna(subset=['PERIOD', 'LIGHT', 'WEATHER_SIMPLE', 'ALCOHOL', 'SEVERE
# Create dummy variables; drop reference categories (Afternoon, Daylight, Clear) X   = pd.get_dummies(df_lr[['PERIOD', 'LIGHT', 'WEATHER_SIMPLE']])
ref = [c for c in X.columns if any(r in c for r in ['Afternoon', 'Daylight', 'Cle X   = X.drop(columns=ref, errors='ignore') X['ALCOHOL'] = df_lr['ALCOHOL'].values y   = df_lr['SEVERE']
# Stratified 80/20 train-test split; balanced class weights for imbalanced outcom
X_tr, X_te, y_tr, y_te = train_test_split(     X, y, test_size=0.2, random_state=42, stratify=y)
lr = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced') lr.fit(X_tr, y_tr)
# Evaluate auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]) print(f'\nLogistic Regression AUC-ROC: {auc:.4f}')  # 0.6189
# Odds ratios = e^(coefficient) coef_df       = pd.DataFrame({'Feature': X.columns, 'Coef': lr.coef_[0]}) coef_df['OR'] = np.exp(coef_df['Coef']) coef_df['OR_lo'] = np.exp(coef_df['Coef'] - 1.96 * 0.05) coef_df['OR_hi'] = np.exp(coef_df['Coef'] + 1.96 * 0.05) print('\nOdds Ratios:')
print(coef_df.sort_values('OR', ascending=False).round(3).to_string(index=False)) # ----- 6. CORRELATION ANALYSIS -----
# Spearman: crash hour vs. ordinal severity code rho, p_rho = spearmanr(df['HOUR'], df['SEV_CODE']) print(f'\nSpearman Correlation (Hour vs. Severity):') print(f'  rho = {rho:.4f}, p = {p_rho:.4f}')  # 0.0112, 0.029
# Pearson: alcohol involvement vs. severity code r, p_r = pearsonr(df['ALCOHOL'], df['SEV_CODE']) print(f'\nPearson Correlation (Alcohol vs. Severity):') print(f'  r = {r:.4f}, p = {p_r:.2e}')  # 0.0377, <0.001 # ----- 7. VISUALIZATIONS ----colors_period = ['#2C3E50', '#2980B9', '#F39C12', '#8E44AD']
# Figure 1: Crash volume + fatality rate by time period fig, axes = plt.subplots(1, 2, figsize=(13, 5)) fig.suptitle('Figure 1: Crash Distribution by Time-of-Day Period', fontweight='bo
crash_counts = [df[df['PERIOD'] == p].shape[0] for p in period_order] fat_rates    = [ct_pct.loc[p, 'Fatality'] for p in period_order]
axes[0].bar(period_order, crash_counts, color=colors_period, alpha=0.87) axes[0].set_title('Total Crashes by Period') axes[0].set_ylabel('Crash Count') for b, v in zip(axes[0].patches, crash_counts):     axes[0].text(b.get_x() + b.get_width()/2, v + 100, f'{v:,}', ha='center', fon
axes[1].bar(period_order, fat_rates, color=colors_period, alpha=0.87) axes[1].set_title('Fatality Rate by Period (%)') axes[1].set_ylabel('Fatality Rate (%)') for b, v in zip(axes[1].patches, fat_rates):     axes[1].text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}%', ha='center',
plt.tight_layout() plt.savefig('fig1_period_severity.png', dpi=150) plt.show()
# Figure 2: Hourly fatality rate + lighting heatmap fig, axes = plt.subplots(1, 2, figsize=(14, 5)) fig.suptitle('Figure 2: Hourly Crash Profile', fontweight='bold')
period_ranges = [(0,6,'Early Morning'),(6,12,'Morning'),(12,18,'Afternoon'),(18,2 bg_colors = {'Early Morning':'#2C3E50','Morning':'#2980B9','Afternoon':'#F39C12', for start, end, name in period_ranges:     axes[0].axvspan(start, end, alpha=0.08, color=bg_colors[name], label=name)
axes[0].plot(hourly['HOUR'], hourly['fatal_rate'], 'o-', color='#C0392B', linewid axes[0].set_title('Fatality Rate by Hour of Day') axes[0].set_xlabel('Hour'); axes[0].set_ylabel('Fatality Rate (%)') axes[0].set_xticks(range(0, 24, 2)) axes[0].grid(alpha=0.3)
hourly_light = df.groupby(['HOUR', 'LIGHT']).size().unstack(fill_value=0) sns.heatmap(hourly_light[['Daylight', 'Dark']].T, cmap='YlOrRd', ax=axes[1], line axes[1].set_title('Crash Volume: Hour × Lighting') axes[1].set_xlabel('Hour of Day')
plt.tight_layout() plt.savefig('fig2_hourly.png', dpi=150) plt.show()
# Figure 3: Lighting fatality rate + alcohol severity breakdown fig, axes = plt.subplots(1, 2, figsize=(13, 5)) fig.suptitle('Figure 3: Severity by Lighting and Alcohol', fontweight='bold')
light_fat = df.groupby('LIGHT').agg(total=('SEVERE','count'), fatal=('SEVERE','su light_fat['rate'] = light_fat['fatal'] / light_fat['total'] * 100 light_sub = light_fat[light_fat.index.isin(['Daylight','Dark'])] axes[0].bar(light_sub.index, light_sub['rate'], color=['#F39C12','#2C3E50'], alph axes[0].set_title('Fatality Rate by Lighting Condition') axes[0].set_ylabel('Fatality Rate (%)') for b, v in zip(axes[0].patches, light_sub['rate']):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}%', ha='center', axes[0].text(0.5, 0.92, 'Chi²(4)=98.09, p<0.001', transform=axes[0].transAxes,              ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='ligh
alc_ct  = pd.crosstab(df['ALCOHOL'], df['Crash_Severity']) alc_pct = alc_ct.div(alc_ct.sum(axis=1), axis=0) * 100 bottom  = np.zeros(2) for col, color in zip(['Injury','Fatality'], ['#2980B9','#C0392B']):     if col in alc_pct.columns:         vals = alc_pct[col].values         axes[1].bar(['No Alcohol','Alcohol Involved'], vals, bottom=bottom, label         for j, (v, b) in enumerate(zip(vals, bottom)):             if v > 2:
                axes[1].text(j, b + v/2, f'{v:.1f}%', ha='center', va='center',                              fontsize=9, fontweight='bold', color='white')         bottom += vals axes[1].set_title('Severity by Alcohol Involvement') axes[1].set_ylabel('Percentage (%)')
axes[1].legend()
plt.tight_layout() plt.savefig('fig3_lighting_alcohol.png', dpi=150) plt.show()
# Figure 4: Logistic regression odds ratio forest plot fig, ax = plt.subplots(figsize=(10, 5)) fig.suptitle('Figure 4: Logistic Regression Odds Ratios\n(Reference: Afternoon, D              fontweight='bold')
feat_labels = coef_df.sort_values('OR', ascending=True)['Feature'].tolist() or_vals     = coef_df.sort_values('OR', ascending=True)['OR'].tolist() or_lo       = coef_df.sort_values('OR', ascending=True)['OR_lo'].tolist() or_hi       = coef_df.sort_values('OR', ascending=True)['OR_hi'].tolist()
bar_colors = ['#C0392B' if o > 1 else '#27AE60' for o in or_vals] ax.barh(range(len(feat_labels)), or_vals,         xerr=[np.array(or_vals)-np.array(or_lo), np.array(or_hi)-np.array(or_vals         color=bar_colors, alpha=0.85, height=0.55,         error_kw=dict(ecolor='gray', capsize=4)) ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', label='OR=1 (no eff ax.set_yticks(range(len(feat_labels))) ax.set_yticklabels(feat_labels) ax.set_xlabel('Odds Ratio') ax.legend() ax.grid(axis='x', alpha=0.3) ax.text(0.72, 0.05, f'AUC = {auc:.2f}', transform=ax.transAxes, fontsize=9,         bbox=dict(boxstyle='round', facecolor='lightyellow'))
plt.tight_layout() plt.savefig('fig4_odds_ratios.png', dpi=150) plt.show() print('\nAnalysis complete. All figures saved.')
