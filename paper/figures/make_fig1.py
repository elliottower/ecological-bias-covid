import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import csv
from collections import defaultdict
from pathlib import Path

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
})

DATA_DIR = Path("../../data")
CDC_CSV = DATA_DIR / "cdc_case_surveillance" / "cdc_full.csv"

with open(DATA_DIR / "cdc_case_surveillance/results/cdc_ecological_fallacy.json") as f:
    cdc = json.load(f)

with open(DATA_DIR / "mexico_covid/mexico_ecological_results.json") as f:
    mexico = json.load(f)


def compute_age_mortality_cdc():
    """Stream CDC CSV to compute mortality rate by age group."""
    cache = DATA_DIR / "cdc_case_surveillance/results/age_group_mortality.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    counts = defaultdict(lambda: {"total": 0, "died": 0})
    with open(CDC_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ag = row.get("age_group", "").strip()
            died = row.get("death_yn", "").strip()
            if ag and ag != "Missing" and ag != "Unknown":
                counts[ag]["total"] += 1
                if died == "Yes":
                    counts[ag]["died"] += 1

    result = {}
    for ag, c in sorted(counts.items()):
        if c["total"] > 0:
            result[ag] = {
                "total": c["total"],
                "died": c["died"],
                "mortality_rate": c["died"] / c["total"],
            }
    with open(cache, "w") as f:
        json.dump(result, f, indent=2)
    return result


def age_group_midpoint(ag):
    """Extract numeric midpoint from CDC age group string."""
    ag = ag.strip()
    if "0 - 17" in ag:
        return 8.5
    parts = ag.replace(" Years", "").replace("+", "").split(" - ")
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    if "80" in ag:
        return 85
    return None


print("Computing age-group mortality from CDC data (this may take a few minutes)...")
age_mort = compute_age_mortality_cdc()
print(f"  Found {len(age_mort)} age groups")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Panel A: CDC ---
ax = axes[0]
sites = cdc["site_data"]
exclude = {"Missing", "Unknown"}
x_cdc = [s["prop_80plus"] for s in sites if s["site"] not in exclude]
y_cdc = [s["prop_died"] for s in sites if s["site"] not in exclude]
names_cdc = [s["site"][:12] for s in sites if s["site"] not in exclude]

ax.scatter(x_cdc, y_cdc, s=60, c='#2171b5', edgecolors='white',
           linewidth=0.5, zorder=3, label="Site-level (pseudo-sites)")

slope, intercept, _, p_val, _ = stats.linregress(x_cdc, y_cdc)
x_fit = np.linspace(min(x_cdc) - 0.01, max(x_cdc) + 0.01, 100)
ax.plot(x_fit, intercept + slope * x_fit, '--', color='#cb181d',
        linewidth=1.5, alpha=0.8)

# CI band for ecological slope (n=7)
n = len(x_cdc)
x_arr = np.array(x_cdc)
y_arr = np.array(y_cdc)
y_pred = intercept + slope * x_arr
residuals = y_arr - y_pred
se_resid = np.sqrt(np.sum(residuals**2) / (n - 2))
x_mean = np.mean(x_arr)
ss_x = np.sum((x_arr - x_mean)**2)
ci_upper = intercept + slope * x_fit + 1.96 * se_resid * np.sqrt(1/n + (x_fit - x_mean)**2 / ss_x)
ci_lower = intercept + slope * x_fit - 1.96 * se_resid * np.sqrt(1/n + (x_fit - x_mean)**2 / ss_x)
ax.fill_between(x_fit, ci_lower, ci_upper, color='#cb181d', alpha=0.08)

for i, name in enumerate(names_cdc):
    dy = -4 if "White" in name else 4
    ax.annotate(name, (x_cdc[i], y_cdc[i]), fontsize=6, alpha=0.7,
                xytext=(4, dy), textcoords='offset points')

ax.set_xlabel("Proportion aged 80+")
ax.set_ylabel("Mortality rate")
ax.set_title(r"A.  CDC: ecological regression")
p_str = f"$p={p_val:.2f}$" if p_val >= 0.01 else f"$p={p_val:.1e}$"
ax.text(0.05, 0.95, r"Ecological: $\beta=" + f"{slope:+.2f}$, {p_str}",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee0d2', alpha=0.8))
ax.text(0.05, 0.87, r"Individual: OR = 9.9, $p < 0.001$",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#deebf7', alpha=0.8))

# --- Panel B: Mexico ---
ax = axes[1]
mx_sites = mexico["ecological_regression"]["site_details"]
x_mx = [v["prop_elderly"] for v in mx_sites.values()]
y_mx = [v["mortality_rate"] for v in mx_sites.values()]

ax.scatter(x_mx, y_mx, s=60, c='#238b45', edgecolors='white',
           linewidth=0.5, zorder=3, label="Site-level (32 states)")

slope_mx = mexico["ecological_regression"]["beta"]
intercept_mx = mexico["ecological_regression"]["intercept"]
x_fit = np.linspace(min(x_mx) - 0.005, max(x_mx) + 0.005, 100)
ax.plot(x_fit, intercept_mx + slope_mx * x_fit, '--', color='#cb181d',
        linewidth=1.5, alpha=0.8, label=r"Ecological: $\beta=+1.31$")

# CI band for Mexico ecological slope (n=32)
n_mx = len(x_mx)
x_arr_mx = np.array(x_mx)
y_arr_mx = np.array(y_mx)
y_pred_mx = intercept_mx + slope_mx * x_arr_mx
resid_mx = y_arr_mx - y_pred_mx
se_resid_mx = np.sqrt(np.sum(resid_mx**2) / (n_mx - 2))
x_mean_mx = np.mean(x_arr_mx)
ss_x_mx = np.sum((x_arr_mx - x_mean_mx)**2)
ci_upper_mx = intercept_mx + slope_mx * x_fit + 1.96 * se_resid_mx * np.sqrt(1/n_mx + (x_fit - x_mean_mx)**2 / ss_x_mx)
ci_lower_mx = intercept_mx + slope_mx * x_fit - 1.96 * se_resid_mx * np.sqrt(1/n_mx + (x_fit - x_mean_mx)**2 / ss_x_mx)
ax.fill_between(x_fit, ci_lower_mx, ci_upper_mx, color='#cb181d', alpha=0.08)

ax.set_xlabel("Proportion aged 70+")
ax.set_ylabel("Mortality rate")
ax.set_title(r"B.  Mexico: ecological regression")
ax.text(0.05, 0.95, r"Ecological: $\beta=+1.31$, $p<0.0001$",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee0d2', alpha=0.8))
ax.text(0.05, 0.87, r"Individual: OR = 11.3, $p < 0.001$",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#deebf7', alpha=0.8))

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.grid(True, alpha=0.2, linewidth=0.5)

plt.tight_layout()
plt.savefig("fig1_ecological_bias.pdf", bbox_inches='tight')
plt.savefig("fig1_ecological_bias.png", bbox_inches='tight', dpi=300)
print("Saved fig1_ecological_bias.pdf and .png")
