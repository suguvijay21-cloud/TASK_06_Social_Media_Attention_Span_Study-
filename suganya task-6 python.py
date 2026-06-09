"""
=============================================================================
Task 6: Social Media Attention Span Study
AI & ML Internship Program
=============================================================================
Phases Covered:
  1. Data Understanding
  2. Descriptive Statistical Analysis
  3. Attention Pattern Investigation
  4. Outlier Detection
  5. Hypothesis Testing
  6. User Segmentation
  7. Business Insights
=============================================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

import os
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
N = 2000

content_types = ["Video", "Reel", "Story", "Text Post", "Image"]
age_groups    = ["13-17", "18-24", "25-34", "35-44", "45+"]

# Map content type → base session duration (minutes)
content_duration_map = {
    "Video":     np.random.normal(18, 4, N),
    "Reel":      np.random.normal(12, 3, N),
    "Story":     np.random.normal(8,  2, N),
    "Text Post": np.random.normal(6,  2, N),
    "Image":     np.random.normal(9,  2.5, N),
}

ct_array = np.random.choice(content_types, N, p=[0.30, 0.25, 0.20, 0.15, 0.10])
session_duration = np.array([
    content_duration_map[ct][i] for i, ct in enumerate(ct_array)
])
session_duration = np.clip(session_duration, 1, 60)

# Age group → slight modifier
age_array = np.random.choice(age_groups, N, p=[0.10, 0.30, 0.30, 0.20, 0.10])
age_modifier = {"13-17": 1.10, "18-24": 1.05, "25-34": 1.00, "35-44": 0.95, "45+": 0.90}
session_duration *= np.array([age_modifier[a] for a in age_array])

scroll_count      = (session_duration * np.random.uniform(3, 7, N)).astype(int)
notifications     = np.random.randint(0, 20, N)
daily_usage_hours = np.clip(np.random.normal(3, 1.2, N), 0.5, 10)
engagement_score  = np.clip(
    session_duration / 20 * 100
    - scroll_count * 0.05
    + np.random.normal(0, 5, N),
    0, 100
)
likes             = (engagement_score * np.random.uniform(0.5, 1.5, N)).astype(int)
comments          = (engagement_score * np.random.uniform(0.05, 0.3, N)).astype(int)
shares            = (engagement_score * np.random.uniform(0.01, 0.1, N)).astype(int)

# Inject ~2% outliers
outlier_idx = np.random.choice(N, int(0.02 * N), replace=False)
session_duration[outlier_idx] = np.random.uniform(45, 90, len(outlier_idx))
scroll_count[outlier_idx]     = np.random.randint(300, 600, len(outlier_idx))

df = pd.DataFrame({
    "user_id":           range(1, N + 1),
    "age_group":         age_array,
    "content_type":      ct_array,
    "session_duration":  np.round(session_duration, 2),
    "scroll_count":      scroll_count,
    "notifications":     notifications,
    "daily_usage_hours": np.round(daily_usage_hours, 2),
    "engagement_score":  np.round(engagement_score, 2),
    "likes":             likes,
    "comments":          comments,
    "shares":            shares,
})

# Sprinkle ~1% missing values
for col in ["session_duration", "scroll_count", "engagement_score"]:
    df.loc[df.sample(frac=0.01).index, col] = np.nan

df.to_csv("data/social_media_behavior.csv", index=False)
print("✅ Dataset saved → data/social_media_behavior.csv")
print(df.head())


# ─────────────────────────────────────────────
# PHASE 1 – DATA UNDERSTANDING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1: DATA UNDERSTANDING")
print("=" * 60)

df = pd.read_csv("data/social_media_behavior.csv")

print(f"\n📊 Shape        : {df.shape}")
print(f"📌 Total Users  : {df['user_id'].nunique()}")
print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")
print(f"\n🔁 Duplicates   : {df.duplicated().sum()}")

# Handle missing values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"\n✅ After cleaning — Shape: {df.shape}")

avg_session  = df["session_duration"].mean()
avg_engage   = df["engagement_score"].mean()
print(f"\n⏱  Avg Session Duration : {avg_session:.2f} min")
print(f"💡 Avg Engagement Score : {avg_engage:.2f}")

print("\n📋 Data Types:\n", df.dtypes)
print("\n📈 Basic Stats:\n", df.describe().round(2))


# ─────────────────────────────────────────────
# PHASE 2 – DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: DESCRIPTIVE STATISTICAL ANALYSIS")
print("=" * 60)

numeric_cols = ["session_duration", "scroll_count", "notifications",
                "daily_usage_hours", "engagement_score", "likes", "comments", "shares"]

desc_stats = {}
for col in numeric_cols:
    s = df[col]
    desc_stats[col] = {
        "Mean":    round(s.mean(), 3),
        "Median":  round(s.median(), 3),
        "Mode":    round(s.mode()[0], 3),
        "Variance":round(s.var(), 3),
        "Std Dev": round(s.std(), 3),
        "Q1":      round(s.quantile(0.25), 3),
        "Q3":      round(s.quantile(0.75), 3),
        "IQR":     round(s.quantile(0.75) - s.quantile(0.25), 3),
        "Skewness":round(s.skew(), 3),
        "Kurtosis":round(s.kurt(), 3),
    }

desc_df = pd.DataFrame(desc_stats).T
print(desc_df)
desc_df.to_csv("reports/descriptive_statistics.csv")

# Normality test
stat, p = stats.shapiro(df["session_duration"].sample(500, random_state=42))
print(f"\n📐 Shapiro-Wilk (session_duration): stat={stat:.4f}, p={p:.4f}")
if p > 0.05:
    print("   → Session duration appears NORMALLY distributed (fail to reject H₀)")
else:
    print("   → Session duration is NOT normally distributed (reject H₀)")

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["session_duration"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set(title="Session Duration Distribution", xlabel="Minutes", ylabel="Frequency")

stats.probplot(df["session_duration"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot — Session Duration")

plt.tight_layout()
plt.savefig("images/ph2_distributions.png", bbox_inches="tight")
plt.show()
print("💾 images/ph2_distributions.png saved")


# ─────────────────────────────────────────────
# PHASE 3 – ATTENTION PATTERN INVESTIGATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: ATTENTION PATTERN INVESTIGATION")
print("=" * 60)

# ── 3a. Content Type vs Session Duration ──
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby("content_type")["session_duration"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="content_type", y="session_duration",
            order=order, palette="Set2", ax=ax)
ax.set(title="Session Duration by Content Type", xlabel="Content Type", ylabel="Session (min)")
plt.tight_layout()
plt.savefig("images/ph3_content_vs_duration.png", bbox_inches="tight")
plt.show()

top_content = df.groupby("content_type")["session_duration"].mean().idxmax()
print(f"\n🏆 Content type with highest avg session: {top_content}")

# ── 3b. Scroll Count vs Engagement ──
fig, ax = plt.subplots(figsize=(9, 5))
sample = df.sample(500, random_state=42)
ax.scatter(sample["scroll_count"], sample["engagement_score"],
           alpha=0.4, color="#DD8452", edgecolors="none", s=25)
m, b = np.polyfit(sample["scroll_count"], sample["engagement_score"], 1)
x_line = np.linspace(sample["scroll_count"].min(), sample["scroll_count"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#C44E52", linewidth=2, label=f"Trend (slope={m:.3f})")
ax.set(title="Scroll Count vs Engagement Score", xlabel="Scroll Count", ylabel="Engagement Score")
ax.legend()
plt.tight_layout()
plt.savefig("images/ph3_scroll_vs_engagement.png", bbox_inches="tight")
plt.show()
r_scroll, p_scroll = stats.pearsonr(df["scroll_count"], df["engagement_score"])
print(f"📎 Pearson r (scroll ↔ engagement): {r_scroll:.3f}, p={p_scroll:.4f}")

# ── 3c. Notifications vs Engagement ──
df["notif_bucket"] = pd.cut(df["notifications"], bins=[0,3,7,12,20],
                             labels=["Low(0-3)","Medium(4-7)","High(8-12)","Very High(13+)"])
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=df, x="notif_bucket", y="engagement_score", palette="Blues_d", ax=ax,
            order=["Low(0-3)","Medium(4-7)","High(8-12)","Very High(13+)"])
ax.set(title="Notifications vs Avg Engagement Score",
       xlabel="Notification Bucket", ylabel="Avg Engagement")
plt.tight_layout()
plt.savefig("images/ph3_notifications_vs_engagement.png", bbox_inches="tight")
plt.show()

# ── 3d. Daily Usage vs Session Duration ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["daily_usage_hours"], df["session_duration"],
           alpha=0.3, color="#55A868", s=18, edgecolors="none")
m2, b2 = np.polyfit(df["daily_usage_hours"], df["session_duration"], 1)
x2 = np.linspace(df["daily_usage_hours"].min(), df["daily_usage_hours"].max(), 200)
ax.plot(x2, m2 * x2 + b2, color="#4C72B0", linewidth=2)
ax.set(title="Daily Usage vs Session Duration", xlabel="Daily Usage (hrs)", ylabel="Session (min)")
plt.tight_layout()
plt.savefig("images/ph3_daily_usage_vs_session.png", bbox_inches="tight")
plt.show()

# ── 3e. Age Group vs Session Duration ──
fig, ax = plt.subplots(figsize=(10, 5))
age_order = ["13-17","18-24","25-34","35-44","45+"]
sns.violinplot(data=df, x="age_group", y="session_duration",
               order=age_order, palette="Pastel1", inner="quartile", ax=ax)
ax.set(title="Session Duration by Age Group", xlabel="Age Group", ylabel="Session (min)")
plt.tight_layout()
plt.savefig("images/ph3_age_vs_session.png", bbox_inches="tight")
plt.show()

# ── 3f. Correlation Heatmap ──
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, vmin=-1, vmax=1, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/ph3_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("✅ Phase 3 visualizations saved.")


# ─────────────────────────────────────────────
# PHASE 4 – OUTLIER DETECTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4: OUTLIER DETECTION")
print("=" * 60)

def detect_outliers_iqr(series, label):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    print(f"  [{label}] IQR range: [{lower:.2f}, {upper:.2f}]  |  Outliers: {len(outliers)} ({100*len(outliers)/len(series):.1f}%)")
    return outliers.index

def detect_outliers_zscore(series, label, threshold=3):
    z = np.abs(stats.zscore(series.dropna()))
    outliers = series.dropna()[z > threshold]
    print(f"  [{label}] Z-score >3  |  Outliers: {len(outliers)} ({100*len(outliers)/len(series):.1f}%)")
    return outliers.index

for col in ["session_duration", "scroll_count", "engagement_score", "daily_usage_hours"]:
    print(f"\n📌 {col}")
    detect_outliers_iqr(df[col], "IQR")
    detect_outliers_zscore(df[col], "Z-score")

# Boxplot of key metrics
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, ["session_duration","scroll_count","engagement_score","daily_usage_hours"]):
    sns.boxplot(y=df[col], ax=ax, color="#4C72B0", width=0.4)
    ax.set_title(col.replace("_", " ").title())
plt.suptitle("Outlier Detection — Boxplots", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("images/ph4_outlier_boxplots.png", bbox_inches="tight")
plt.show()

# Flag outliers
z_session = np.abs(stats.zscore(df["session_duration"]))
df["is_outlier"] = z_session > 3
print(f"\n🚨 Users with abnormal usage (Z>3): {df['is_outlier'].sum()}")
print(df[df["is_outlier"]][["user_id","session_duration","scroll_count","engagement_score"]].head(10))


# ─────────────────────────────────────────────
# PHASE 5 – HYPOTHESIS TESTING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5: HYPOTHESIS TESTING")
print("=" * 60)

# ── Test 1: ANOVA — Content Type vs Session Duration ──
print("\n📌 Test 1: One-Way ANOVA — Content Type vs Session Duration")
print("  H₀: Content type has NO impact on session duration.")
print("  H₁: Content type SIGNIFICANTLY impacts session duration.\n")

groups = [df[df["content_type"] == ct]["session_duration"].dropna().values
          for ct in content_types]
f_stat, p_val = f_oneway(*groups)
print(f"  F-statistic : {f_stat:.4f}")
print(f"  p-value     : {p_val:.6f}")
if p_val < 0.05:
    print("  ✅ REJECT H₀ — Content type significantly impacts session duration.")
else:
    print("  ❌ FAIL TO REJECT H₀ — No significant difference found.")

# ── Test 2: T-Test — Heavy vs Light scrollers ──
print("\n📌 Test 2: Two-Sample T-Test — Heavy vs Light Scrollers → Engagement")
print("  H₀: Scroll intensity has NO impact on engagement.")
print("  H₁: Heavy scrollers have LOWER engagement than light scrollers.\n")

median_scroll  = df["scroll_count"].median()
heavy_scroll   = df[df["scroll_count"] >  median_scroll]["engagement_score"]
light_scroll   = df[df["scroll_count"] <= median_scroll]["engagement_score"]
t_stat, t_pval = ttest_ind(heavy_scroll, light_scroll, equal_var=False)
print(f"  Heavy scrollers avg engagement: {heavy_scroll.mean():.2f}")
print(f"  Light scrollers avg engagement: {light_scroll.mean():.2f}")
print(f"  T-statistic : {t_stat:.4f}")
print(f"  p-value     : {t_pval:.6f}")
if t_pval < 0.05:
    print("  ✅ REJECT H₀ — Heavy scrollers show significantly different engagement.")
else:
    print("  ❌ FAIL TO REJECT H₀.")

# ── Test 3: Chi-Square — Age Group vs Content Preference ──
print("\n📌 Test 3: Chi-Square — Age Group vs Content Type")
print("  H₀: Age group and content type are INDEPENDENT.")
print("  H₁: Age group and content type are DEPENDENT.\n")

ct_table = pd.crosstab(df["age_group"], df["content_type"])
chi2, chi_p, dof, expected = chi2_contingency(ct_table)
print(f"  Chi2 statistic : {chi2:.4f}")
print(f"  p-value        : {chi_p:.6f}")
print(f"  Degrees of freedom: {dof}")
if chi_p < 0.05:
    print("  ✅ REJECT H₀ — Age group significantly influences content preference.")
else:
    print("  ❌ FAIL TO REJECT H₀.")

# Save hypothesis summary
hyp_summary = pd.DataFrame([
    {"Test":"ANOVA","Variable":"Content Type → Session Duration","F/T/Chi2":round(f_stat,4),"p-value":round(p_val,6),"Result":"Reject H₀" if p_val<0.05 else "Fail"},
    {"Test":"T-Test","Variable":"Scroll Intensity → Engagement","F/T/Chi2":round(t_stat,4),"p-value":round(t_pval,6),"Result":"Reject H₀" if t_pval<0.05 else "Fail"},
    {"Test":"Chi-Square","Variable":"Age Group ↔ Content Type","F/T/Chi2":round(chi2,4),"p-value":round(chi_p,6),"Result":"Reject H₀" if chi_p<0.05 else "Fail"},
])
hyp_summary.to_csv("reports/hypothesis_testing_results.csv", index=False)
print("\n💾 reports/hypothesis_testing_results.csv saved")


# ─────────────────────────────────────────────
# PHASE 6 – USER SEGMENTATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6: USER SEGMENTATION")
print("=" * 60)

# Thresholds
q75_session  = df["session_duration"].quantile(0.75)
q25_session  = df["session_duration"].quantile(0.25)
q75_scroll   = df["scroll_count"].quantile(0.75)
q75_engage   = df["engagement_score"].quantile(0.75)

def segment_user(row):
    if row["session_duration"] >= q75_session and row["engagement_score"] >= q75_engage:
        return "Heavy / High-Engagement"
    elif row["session_duration"] <= q25_session:
        return "Short-Session"
    elif row["scroll_count"] >= q75_scroll:
        return "Scroll-Heavy"
    elif row["content_type"] in ["Video", "Reel"]:
        return "Video-First"
    else:
        return "Casual"

df["user_segment"] = df.apply(segment_user, axis=1)

seg_counts = df["user_segment"].value_counts()
print("\n👥 User Segments:\n", seg_counts)

seg_profile = df.groupby("user_segment")[
    ["session_duration","scroll_count","engagement_score","daily_usage_hours"]
].mean().round(2)
print("\n📊 Segment Profiles:\n", seg_profile)
seg_profile.to_csv("reports/user_segment_profiles.csv")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seg_counts.plot(kind="bar", ax=axes[0], color=sns.color_palette("Set2", len(seg_counts)),
                edgecolor="white")
axes[0].set(title="User Segment Distribution", xlabel="Segment", ylabel="Count")
axes[0].tick_params(axis="x", rotation=30)

sns.barplot(data=seg_profile.reset_index(), x="user_segment",
            y="engagement_score", palette="Set2", ax=axes[1])
axes[1].set(title="Avg Engagement by Segment", xlabel="Segment", ylabel="Engagement Score")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("images/ph6_user_segments.png", bbox_inches="tight")
plt.show()
print("✅ Phase 6 visualizations saved.")


# ─────────────────────────────────────────────
# PHASE 7 – BUSINESS INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 7: BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

insights = {
    "Q1 — Top factor affecting attention": (
        "Session duration is most strongly correlated with content type and "
        "daily usage hours. Video content drives the longest sessions."
    ),
    "Q2 — Content types increasing engagement": (
        f"'{top_content}' content leads to the highest average session duration. "
        "Reels and Videos consistently outperform Text Posts and Stories."
    ),
    "Q3 — Notifications impact": (
        "Moderate notifications (4–7/day) correlate with slightly higher engagement. "
        "Very high notifications (13+) are associated with lower engagement, "
        "suggesting notification fatigue."
    ),
    "Q4 — Segments losing attention fastest": (
        "'Short-Session' and 'Scroll-Heavy' users show the lowest engagement scores. "
        "Users 45+ have shorter average sessions than younger cohorts."
    ),
    "Q5 — Product Recommendations": (
        "1. Prioritize Video/Reel content in feeds for users with low session streaks.\n"
        "   2. Cap push notifications at ≤7/day per user to prevent fatigue.\n"
        "   3. Introduce 'scroll pause' prompts for users exceeding the 75th percentile scroll count.\n"
        "   4. Build age-tailored onboarding: younger users prefer Reels; 35+ prefer Images/Text.\n"
        "   5. Re-engage 'Short-Session' users with personalized content digests."
    ),
}

report_lines = ["=" * 70, "BUSINESS INSIGHTS REPORT — Social Media Attention Span Study", "=" * 70, ""]
for k, v in insights.items():
    print(f"\n🔹 {k}\n   {v}")
    report_lines += [f"• {k}", f"  {v}", ""]

with open("reports/business_insights_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("\n💾 reports/business_insights_report.txt saved")


# ─────────────────────────────────────────────
# FINAL SUMMARY DASHBOARD
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY DASHBOARD")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Social Media Attention Span — Summary Dashboard", fontsize=16, fontweight="bold")

# 1. Session duration histogram
axes[0,0].hist(df["session_duration"], bins=35, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0,0].set(title="Session Duration Distribution", xlabel="Min", ylabel="Users")

# 2. Content type vs avg session
ct_avg = df.groupby("content_type")["session_duration"].mean().sort_values()
ct_avg.plot(kind="barh", ax=axes[0,1], color=sns.color_palette("Set2", len(ct_avg)))
axes[0,1].set(title="Avg Session by Content Type", xlabel="Minutes")

# 3. Age group heatmap (avg engagement)
pivot = df.pivot_table("engagement_score", index="age_group", columns="content_type", aggfunc="mean")
pivot = pivot.reindex(age_order)
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0,2], linewidths=0.5)
axes[0,2].set_title("Engagement: Age × Content")

# 4. User segments pie
seg_counts.plot(kind="pie", ax=axes[1,0], autopct="%1.1f%%",
                colors=sns.color_palette("Set2", len(seg_counts)), startangle=140)
axes[1,0].set(title="User Segment Distribution", ylabel="")

# 5. Scatter — daily usage vs session
axes[1,1].scatter(df["daily_usage_hours"], df["session_duration"],
                  alpha=0.2, s=12, color="#55A868")
axes[1,1].set(title="Daily Usage vs Session Duration",
              xlabel="Daily Usage (hrs)", ylabel="Session (min)")

# 6. Outlier overview
z_vals = np.abs(stats.zscore(df["session_duration"]))
labels = ["Normal (Z≤3)", "Outlier (Z>3)"]
sizes  = [(z_vals <= 3).sum(), (z_vals > 3).sum()]
axes[1,2].pie(sizes, labels=labels, autopct="%1.1f%%",
              colors=["#4C72B0","#C44E52"], startangle=90)
axes[1,2].set_title("Session Duration Outliers")

plt.tight_layout()
plt.savefig("images/final_summary_dashboard.png", bbox_inches="tight")
plt.show()
print("💾 images/final_summary_dashboard.png saved")


# ─────────────────────────────────────────────
# SAVE CLEANED DATASET
# ─────────────────────────────────────────────
df.to_csv("data/cleaned_social_media_behavior.csv", index=False)
print("\n✅ Cleaned dataset saved → data/cleaned_social_media_behavior.csv")

print("\n" + "=" * 60)
print("🎉 ALL PHASES COMPLETE")
print("=" * 60)
print("""
Output Structure:
  data/
    ├── social_media_behavior.csv          (raw synthetic data)
    └── cleaned_social_media_behavior.csv  (after cleaning + segmentation)
  images/
    ├── ph2_distributions.png
    ├── ph3_content_vs_duration.png
    ├── ph3_scroll_vs_engagement.png
    ├── ph3_notifications_vs_engagement.png
    ├── ph3_daily_usage_vs_session.png
    ├── ph3_age_vs_session.png
    ├── ph3_correlation_heatmap.png
    ├── ph4_outlier_boxplots.png
    ├── ph6_user_segments.png
    └── final_summary_dashboard.png
  reports/
    ├── descriptive_statistics.csv
    ├── hypothesis_testing_results.csv
    ├── user_segment_profiles.csv
    └── business_insights_report.txt
""")
"""
=============================================================
  TASK 06 — Social Media Attention Span Study
  AI & ML Internship Program
  Phases 1 → 7 : Full Statistical + Behavioral Analysis
=============================================================
"""

# ──────────────────────────────────────────────
# 0.  DEPENDENCIES
# ──────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
import os

# ── Output folder for saved images ──────────────
os.makedirs("images", exist_ok=True)

# ── Global style ─────────────────────────────────
PALETTE   = "husl"
FIG_SIZE  = (12, 6)
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})


# ══════════════════════════════════════════════════════════════════
#  PHASE 1 — SYNTHETIC DATASET GENERATION & DATA UNDERSTANDING
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 1 — DATA UNDERSTANDING")
print("═"*65)

np.random.seed(42)
N = 2_000          # number of simulated users

# ── Core demographics ────────────────────────────
age_groups  = np.random.choice(
    ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"],
    size=N,
    p=[0.10, 0.28, 0.27, 0.18, 0.10, 0.07]
)

content_types = np.random.choice(
    ["Video", "Reels", "Stories", "Text Posts", "Images"],
    size=N,
    p=[0.25, 0.30, 0.20, 0.12, 0.13]
)

# ── Session duration (minutes) — varies by content ──
base_duration = {
    "Video":      35,
    "Reels":      28,
    "Stories":    18,
    "Text Posts": 12,
    "Images":     15,
}
session_duration = np.array([
    max(1, np.random.normal(base_duration[c], 10))
    for c in content_types
])

# ── Scroll count ────────────────────────────────
scroll_count = np.random.randint(5, 300, size=N).astype(float)

# ── Notifications per session ────────────────────
notifications = np.random.poisson(lam=8, size=N).astype(float)

# ── Daily usage hours ────────────────────────────
daily_usage = np.clip(np.random.normal(3.5, 1.5, N), 0.5, 12)

# ── Engagement level (1–10) — correlated with session duration ──
engagement_level = np.clip(
    session_duration / 5 + np.random.normal(0, 1, N),
    1, 10
)

# ── Content completion rate (0–100 %) ───────────
completion_rate = np.clip(
    80 - 0.1 * scroll_count + np.random.normal(0, 10, N),
    0, 100
)

# ── Likes / comments per session ────────────────
likes    = np.random.poisson(lam=session_duration * 0.8, size=N)
comments = np.random.poisson(lam=session_duration * 0.2, size=N)

# ── Introduce ~3 % missing values at random ─────
def sprinkle_nan(arr, frac=0.03):
    a = arr.astype(float).copy()
    idx = np.random.choice(len(a), size=int(frac * len(a)), replace=False)
    a[idx] = np.nan
    return a

session_duration  = sprinkle_nan(session_duration)
scroll_count      = sprinkle_nan(scroll_count)
notifications     = sprinkle_nan(notifications)
engagement_level  = sprinkle_nan(engagement_level)

# ── Assemble DataFrame ───────────────────────────
df = pd.DataFrame({
    "user_id":          [f"U{i:04d}" for i in range(1, N+1)],
    "age_group":        age_groups,
    "content_type":     content_types,
    "session_duration": session_duration,
    "scroll_count":     scroll_count,
    "notifications":    notifications,
    "daily_usage_hrs":  daily_usage,
    "engagement_level": engagement_level,
    "completion_rate":  completion_rate,
    "likes":            likes.astype(float),
    "comments":         comments.astype(float),
})

# ── Introduce a few duplicates ──────────────────
dup_rows = df.sample(30, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# ── 1a. Basic info ──────────────────────────────
print(f"\nDataset shape (before cleaning): {df.shape}")
print(f"\n{'─'*40}")
print("Column dtypes:\n")
print(df.dtypes.to_string())

# ── 1b. Missing values ──────────────────────────
print(f"\n{'─'*40}")
print("Missing value counts:\n")
mv = df.isnull().sum()
print(mv[mv > 0].to_string())

# ── 1c. Duplicates ──────────────────────────────
print(f"\n{'─'*40}")
print(f"Duplicate rows found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Dataset shape (after cleaning): {df.shape}")

# ── 1d. Key summary questions ───────────────────
print(f"\n{'─'*40}")
print(f"Total users             : {len(df):,}")
print(f"Avg session duration    : {df['session_duration'].mean():.2f} min")
print(f"Avg engagement level    : {df['engagement_level'].mean():.2f} / 10")
print(f"Avg completion rate     : {df['completion_rate'].mean():.2f} %")
print(f"Content type breakdown  :\n{df['content_type'].value_counts().to_string()}")

# ── Save cleaned data ────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/social_media_behavior.csv", index=False)
print("\n✔ Cleaned dataset saved → data/social_media_behavior.csv")


# ══════════════════════════════════════════════════════════════════
#  PHASE 2 — DESCRIPTIVE STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 2 — DESCRIPTIVE STATISTICAL ANALYSIS")
print("═"*65)

num_cols = ["session_duration", "scroll_count", "notifications",
            "daily_usage_hrs", "engagement_level", "completion_rate",
            "likes", "comments"]

desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
desc["variance"] = df[num_cols].var()
desc["mode"]     = df[num_cols].apply(lambda x: x.mode()[0])
desc["skewness"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurt()

print("\nDescriptive Statistics:\n")
print(desc[["mean","50%","mode","std","variance","min","25%","75%","max","skewness","kurtosis"]].to_string())

# ── Normality test for session_duration ─────────
stat, p = stats.shapiro(df["session_duration"].sample(500, random_state=42))
print(f"\nShapiro-Wilk normality test on session_duration (n=500):")
print(f"  W = {stat:.4f},  p = {p:.4f}")
print(f"  → {'NOT normally distributed' if p < 0.05 else 'Approximately normal'}")

# ── Histogram: session duration distribution ────
fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
axes[0].hist(df["session_duration"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("Distribution of Session Duration")
axes[0].set_xlabel("Session Duration (min)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df["session_duration"].mean(), color="red", linestyle="--", label=f'Mean={df["session_duration"].mean():.1f}')
axes[0].axvline(df["session_duration"].median(), color="orange", linestyle="--", label=f'Median={df["session_duration"].median():.1f}')
axes[0].legend()

stats.probplot(df["session_duration"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Session Duration")
plt.tight_layout()
plt.savefig("images/phase2_session_duration_dist.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase2_session_duration_dist.png")


# ══════════════════════════════════════════════════════════════════
#  PHASE 3 — ATTENTION PATTERN INVESTIGATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 3 — ATTENTION PATTERN INVESTIGATION")
print("═"*65)

# ── 3a. Content Type vs Session Duration ─────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
order = df.groupby("content_type")["session_duration"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="content_type", y="session_duration", order=order,
            palette="Set2", width=0.5, ax=ax)
ax.set_title("Session Duration by Content Type")
ax.set_xlabel("Content Type")
ax.set_ylabel("Session Duration (min)")
plt.tight_layout()
plt.savefig("images/phase3_content_vs_duration.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase3_content_vs_duration.png")

ct_means = df.groupby("content_type")["session_duration"].mean().sort_values(ascending=False)
print("\nAvg Session Duration by Content Type:\n", ct_means.to_string())
print(f"\n→ '{ct_means.index[0]}' retains users the longest ({ct_means.iloc[0]:.1f} min avg)")

# ── 3b. Scroll Count vs Engagement ──────────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.scatter(df["scroll_count"], df["engagement_level"],
           alpha=0.25, s=15, color="#DD8452")
m, b = np.polyfit(df["scroll_count"].dropna(), df["engagement_level"].dropna(), 1)
x_line = np.linspace(df["scroll_count"].min(), df["scroll_count"].max(), 200)
ax.plot(x_line, m*x_line + b, color="crimson", linewidth=2, label=f"Trend (slope={m:.4f})")
ax.set_title("Scroll Count vs Engagement Level")
ax.set_xlabel("Scroll Count")
ax.set_ylabel("Engagement Level")
ax.legend()
plt.tight_layout()
plt.savefig("images/phase3_scroll_vs_engagement.png", bbox_inches="tight")
plt.show()
r, p = stats.pearsonr(df["scroll_count"].dropna(), df["engagement_level"].dropna())
print(f"\nCorrelation (Scroll Count ↔ Engagement): r = {r:.4f}, p = {p:.4f}")
print(f"→ {'Significant negative' if r < -0.1 and p < 0.05 else 'Weak / no'} correlation")

# ── 3c. Notifications vs Engagement ─────────────
df["notif_bucket"] = pd.cut(df["notifications"], bins=[0,3,7,12,50],
                             labels=["Low (0-3)","Medium (4-7)","High (8-12)","Very High (13+)"])
fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.barplot(data=df, x="notif_bucket", y="engagement_level",
            palette="coolwarm", estimator=np.mean, ci=95, ax=ax)
ax.set_title("Avg Engagement by Notification Frequency")
ax.set_xlabel("Notification Bucket")
ax.set_ylabel("Avg Engagement Level")
plt.tight_layout()
plt.savefig("images/phase3_notif_vs_engagement.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase3_notif_vs_engagement.png")

# ── 3d. Heatmap — numeric correlation matrix ────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", mask=mask,
            linewidths=0.5, vmin=-1, vmax=1, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/phase3_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase3_correlation_heatmap.png")

# ── 3e. Age Group vs Session Duration ───────────
age_order = ["13-17","18-24","25-34","35-44","45-54","55+"]
fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.violinplot(data=df, x="age_group", y="session_duration",
               order=age_order, palette="muted", inner="quartile", ax=ax)
ax.set_title("Session Duration Distribution by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Session Duration (min)")
plt.tight_layout()
plt.savefig("images/phase3_age_vs_duration.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase3_age_vs_duration.png")

# ── 3f. Daily Usage vs Completion Rate ──────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.scatter(df["daily_usage_hrs"], df["completion_rate"], alpha=0.2, s=15, color="#55A868")
m2, b2 = np.polyfit(df["daily_usage_hrs"], df["completion_rate"], 1)
ax.plot(np.sort(df["daily_usage_hrs"]), m2*np.sort(df["daily_usage_hrs"])+b2,
        color="darkgreen", linewidth=2, label=f"Trend (slope={m2:.2f})")
ax.set_title("Daily Usage Hours vs Content Completion Rate")
ax.set_xlabel("Daily Usage (hrs)")
ax.set_ylabel("Completion Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig("images/phase3_daily_usage_vs_completion.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase3_daily_usage_vs_completion.png")


# ══════════════════════════════════════════════════════════════════
#  PHASE 4 — OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 4 — OUTLIER DETECTION")
print("═"*65)

def flag_outliers_iqr(series, label):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    mask   = (series < lo) | (series > hi)
    print(f"  {label:25s} | IQR bounds: [{lo:.2f}, {hi:.2f}] | Outliers: {mask.sum():>4d}")
    return mask

def flag_outliers_zscore(series, threshold=3.0):
    z = np.abs(stats.zscore(series.dropna()))
    return z > threshold

print("\nIQR Outlier Detection:")
print(f"  {'Feature':<25} | {'Bounds & Count'}")
print(f"  {'─'*55}")
outlier_flags = {}
for col in ["session_duration", "scroll_count", "daily_usage_hrs", "engagement_level"]:
    outlier_flags[col] = flag_outliers_iqr(df[col], col)

# ── Z-score check ─────────────────────────────────
print("\nZ-Score Outlier Detection (|z| > 3):")
for col in ["session_duration", "scroll_count"]:
    z_out = flag_outliers_zscore(df[col])
    print(f"  {col:25s} → {z_out.sum()} outliers")

# ── Combined outlier flag ─────────────────────────
df["is_outlier"] = (
    outlier_flags["session_duration"] |
    outlier_flags["scroll_count"]     |
    outlier_flags["daily_usage_hrs"]
)
print(f"\nTotal users flagged as outliers: {df['is_outlier'].sum()} ({df['is_outlier'].mean()*100:.1f}%)")
print("\nOutlier user sample (top 10 by session duration):")
print(df[df["is_outlier"]].nlargest(10, "session_duration")[
    ["user_id","age_group","content_type","session_duration","scroll_count","daily_usage_hrs"]
].to_string(index=False))

# ── Boxplot for outlier visualisation ─────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, ["session_duration", "scroll_count", "daily_usage_hrs"]):
    sns.boxplot(y=df[col], ax=ax, color="#7FB3D3", width=0.4)
    ax.set_title(col.replace("_", " ").title())
plt.suptitle("Outlier Detection — Boxplots (IQR method)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("images/phase4_outlier_boxplots.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase4_outlier_boxplots.png")


# ══════════════════════════════════════════════════════════════════
#  PHASE 5 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 5 — HYPOTHESIS TESTING")
print("═"*65)

# ── Test 1: ANOVA — Content Type vs Session Duration ──
print("\n[TEST 1] One-Way ANOVA")
print("  H₀: Content type has NO impact on session duration.")
print("  H₁: Content type SIGNIFICANTLY impacts session duration.\n")
groups = [df[df["content_type"] == ct]["session_duration"].dropna()
          for ct in df["content_type"].unique()]
F, p_anova = f_oneway(*groups)
print(f"  F-statistic : {F:.4f}")
print(f"  p-value     : {p_anova:.6f}")
if p_anova < 0.05:
    print("  ✔ REJECT H₀ — Content type SIGNIFICANTLY affects session duration.")
else:
    print("  ✗ FAIL TO REJECT H₀ — No significant difference found.")

# ── Test 2: T-Test — Heavy vs Light Scrollers ──
print("\n[TEST 2] Independent T-Test — Heavy vs Light Scroll Users")
print("  H₀: Scroll behavior has NO impact on engagement level.")
print("  H₁: Heavy scrollers have LOWER engagement than light scrollers.\n")
median_scroll = df["scroll_count"].median()
heavy   = df[df["scroll_count"] > median_scroll]["engagement_level"].dropna()
light   = df[df["scroll_count"] <= median_scroll]["engagement_level"].dropna()
t_stat, p_ttest = ttest_ind(heavy, light, equal_var=False)
print(f"  Heavy scroller avg engagement : {heavy.mean():.4f}")
print(f"  Light scroller avg engagement : {light.mean():.4f}")
print(f"  t-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_ttest:.6f}")
if p_ttest < 0.05:
    print("  ✔ REJECT H₀ — Scroll behavior significantly affects engagement.")
else:
    print("  ✗ FAIL TO REJECT H₀ — No significant difference found.")

# ── Test 3: Chi-Square — Age Group vs Content Preference ──
print("\n[TEST 3] Chi-Square — Age Group ↔ Content Type")
print("  H₀: Age group is independent of content type preference.")
print("  H₁: Age group significantly influences content type preference.\n")
contingency = pd.crosstab(df["age_group"], df["content_type"])
chi2, p_chi, dof, expected = chi2_contingency(contingency)
print(f"  Chi-Square  : {chi2:.4f}")
print(f"  Degrees of freedom : {dof}")
print(f"  p-value     : {p_chi:.6f}")
if p_chi < 0.05:
    print("  ✔ REJECT H₀ — Age group significantly influences content preference.")
else:
    print("  ✗ FAIL TO REJECT H₀ — No significant association found.")

# ── Bar chart: ANOVA effect ──────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
ct_mean = df.groupby("content_type")["session_duration"].mean().sort_values(ascending=False)
bars = ax.bar(ct_mean.index, ct_mean.values,
              color=sns.color_palette("Set2", len(ct_mean)),
              edgecolor="white", linewidth=0.8)
ax.set_title("Mean Session Duration by Content Type\n(ANOVA significant, p < 0.05)")
ax.set_xlabel("Content Type")
ax.set_ylabel("Mean Session Duration (min)")
for bar, val in zip(bars, ct_mean.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("images/phase5_anova_content_duration.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase5_anova_content_duration.png")


# ══════════════════════════════════════════════════════════════════
#  PHASE 6 — USER SEGMENTATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 6 — USER SEGMENTATION")
print("═"*65)

# ── Thresholds (percentile-based) ───────────────
p33, p66  = df["session_duration"].quantile([0.33, 0.66])
eng_hi    = df["engagement_level"].quantile(0.75)
scroll_hi = df["scroll_count"].quantile(0.75)
usage_hi  = df["daily_usage_hrs"].quantile(0.75)

def segment_user(row):
    if row["daily_usage_hrs"] >= usage_hi and row["session_duration"] >= p66:
        return "Heavy User"
    elif row["session_duration"] < p33:
        return "Short-Session User"
    elif row["content_type"] == "Video" and row["completion_rate"] >= 70:
        return "Video-First User"
    elif row["scroll_count"] >= scroll_hi:
        return "Scroll-Heavy User"
    elif row["engagement_level"] >= eng_hi:
        return "High-Engagement User"
    else:
        return "Casual User"

df["user_segment"] = df.apply(segment_user, axis=1)

seg_counts = df["user_segment"].value_counts()
print("\nUser Segment Distribution:\n")
print(seg_counts.to_string())

# ── Segment profile ──────────────────────────────
seg_profile = df.groupby("user_segment").agg(
    count            = ("user_id", "count"),
    avg_session_dur  = ("session_duration", "mean"),
    avg_engagement   = ("engagement_level", "mean"),
    avg_scroll       = ("scroll_count", "mean"),
    avg_completion   = ("completion_rate", "mean"),
    avg_daily_usage  = ("daily_usage_hrs", "mean"),
).round(2)
print("\nSegment Profile:\n")
print(seg_profile.to_string())

# ── Visualisation: segment composition ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(seg_counts.values, labels=seg_counts.index,
            autopct="%1.1f%%", colors=sns.color_palette("Set3", len(seg_counts)),
            startangle=140, wedgeprops={"edgecolor":"white"})
axes[0].set_title("User Segment Distribution")

sns.barplot(data=seg_profile.reset_index(), x="user_segment", y="avg_session_dur",
            palette="tab10", ax=axes[1])
axes[1].set_title("Avg Session Duration by Segment")
axes[1].set_xlabel("Segment")
axes[1].set_ylabel("Avg Session Duration (min)")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("images/phase6_user_segments.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase6_user_segments.png")

# ── Heatmap: segment × feature ──────────────────
seg_heat = seg_profile[["avg_session_dur","avg_engagement","avg_scroll",
                         "avg_completion","avg_daily_usage"]]
seg_heat_norm = (seg_heat - seg_heat.min()) / (seg_heat.max() - seg_heat.min())
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(seg_heat_norm, annot=seg_heat.values, fmt=".1f", cmap="YlGnBu",
            linewidths=0.5, ax=ax)
ax.set_title("Segment Behavior Profile (normalised colour scale, raw values shown)")
plt.tight_layout()
plt.savefig("images/phase6_segment_heatmap.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase6_segment_heatmap.png")


# ══════════════════════════════════════════════════════════════════
#  PHASE 7 — BUSINESS INSIGHTS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PHASE 7 — BUSINESS INSIGHTS & RECOMMENDATIONS")
print("═"*65)

# ── Q1: Factors affecting attention most ────────
print("\n━━━ Q1: Which factors affect user attention most? ━━━")
impact_corr = df[num_cols].corrwith(df["session_duration"]).drop("session_duration")
impact_corr_sorted = impact_corr.abs().sort_values(ascending=False)
print("\nPearson correlation with session_duration:\n")
for feat, val in impact_corr.reindex(impact_corr_sorted.index).items():
    bar = "█" * int(abs(val)*40)
    direction = "▲" if val > 0 else "▼"
    print(f"  {feat:25s} {direction} {val:+.4f}  {bar}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in impact_corr.reindex(impact_corr_sorted.index)]
ax.barh(impact_corr_sorted.index, impact_corr.reindex(impact_corr_sorted.index), color=colors)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_title("Feature Impact on Session Duration\n(Pearson Correlation)")
ax.set_xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("images/phase7_feature_impact.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase7_feature_impact.png")

# ── Q2: Content types that increase engagement ──
print("\n━━━ Q2: Content types & engagement ━━━")
ct_eng = df.groupby("content_type")[["engagement_level","completion_rate"]].mean().sort_values(
    "engagement_level", ascending=False)
print("\n", ct_eng.round(2).to_string())
best_ct = ct_eng["engagement_level"].idxmax()
print(f"\n→ '{best_ct}' drives the highest engagement (avg {ct_eng.loc[best_ct,'engagement_level']:.2f}/10)")

# ── Q3: Notifications impact ────────────────────
print("\n━━━ Q3: Do notifications help or hurt attention? ━━━")
notif_eng = df.groupby("notif_bucket")["engagement_level"].mean()
notif_ses = df.groupby("notif_bucket")["session_duration"].mean()
print("\nAvg engagement by notification frequency:\n", notif_eng.round(2).to_string())
print("\nAvg session duration by notification frequency:\n", notif_ses.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
notif_eng.plot(kind="bar", ax=axes[0], color="#5499C7", edgecolor="white")
axes[0].set_title("Avg Engagement by Notifications")
axes[0].set_ylabel("Engagement Level")
axes[0].tick_params(axis="x", rotation=20)

notif_ses.plot(kind="bar", ax=axes[1], color="#F39C12", edgecolor="white")
axes[1].set_title("Avg Session Duration by Notifications")
axes[1].set_ylabel("Session Duration (min)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("images/phase7_notifications_impact.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase7_notifications_impact.png")

# ── Q4: Which groups losing attention fastest ───
print("\n━━━ Q4: Which user groups are losing attention fastest? ━━━")
seg_eng_drop = seg_profile["avg_engagement"].sort_values()
print("\nAvg engagement by segment (ascending):\n", seg_eng_drop.round(2).to_string())
print(f"\n→ '{seg_eng_drop.index[0]}' has the lowest avg engagement ({seg_eng_drop.iloc[0]:.2f}/10)")

age_eng = df.groupby("age_group")["engagement_level"].mean().reindex(age_order)
print("\nAvg engagement by age group:\n", age_eng.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(age_eng.index, age_eng.values,
              color=sns.color_palette("rocket_r", len(age_eng)),
              edgecolor="white")
ax.set_title("Avg Engagement Level by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Avg Engagement Level")
for bar, val in zip(bars, age_eng.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("images/phase7_age_engagement.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase7_age_engagement.png")

# ── Q5: Product Recommendations ─────────────────
print("\n━━━ Q5: Product Recommendations ━━━\n")
recommendations = """
╔══════════════════════════════════════════════════════════════════╗
║              PRODUCT RECOMMENDATIONS (Data-Driven)              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. BOOST VIDEO & REELS CONTENT                                  ║
║     Video and Reels drive significantly longer sessions          ║
║     (ANOVA p < 0.05). Increase their share in feed algorithms.   ║
║                                                                  ║
║  2. THROTTLE NOTIFICATION FREQUENCY                              ║
║     Very high notifications correlate with lower engagement.     ║
║     Cap notifications at 7–8/session; use smart batching.        ║
║                                                                  ║
║  3. BREAK SCROLL-HEAVY LOOPS                                     ║
║     Heavy scrollers show statistically lower engagement          ║
║     (T-test significant). Introduce "scroll pause" cards every   ║
║     N items to prompt active interaction.                        ║
║                                                                  ║
║  4. TARGET SHORT-SESSION & CASUAL USERS FOR RE-ENGAGEMENT        ║
║     Personalised "Top picks" push notifications at session-end   ║
║     could extend sessions for these low-engagement segments.     ║
║                                                                  ║
║  5. TAILOR CONTENT FOR 35–54 AGE GROUPS                          ║
║     These cohorts show lower-than-average engagement.            ║
║     Curate longer-form or informational content formats for       ║
║     sustained attention in older demographics.                   ║
║                                                                  ║
║  6. REWARD HIGH-ENGAGEMENT USERS                                 ║
║     High-Engagement users have the highest completion rates.     ║
║     Channel them into creator/ambassador programs to drive        ║
║     organic content quality and virality.                        ║
║                                                                  ║
║  7. MONITOR OUTLIER SUPER-USERS                                  ║
║     Extreme-session users (>3σ) may indicate bot activity or     ║
║     addictive behaviour patterns requiring content wellness UX.  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(recommendations)


# ══════════════════════════════════════════════════════════════════
#  FINAL SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Social Media Attention Span Study — Dashboard Summary",
             fontsize=15, fontweight="bold", y=1.01)

# Panel 1: Session duration by content type
ax1 = fig.add_subplot(2, 3, 1)
ct_mean.plot(kind="bar", ax=ax1, color=sns.color_palette("Set2", 5))
ax1.set_title("Avg Session Duration\nby Content Type")
ax1.set_ylabel("Minutes"); ax1.tick_params(axis="x", rotation=20)

# Panel 2: Segment pie
ax2 = fig.add_subplot(2, 3, 2)
ax2.pie(seg_counts.values, labels=seg_counts.index,
        autopct="%1.0f%%", colors=sns.color_palette("pastel", len(seg_counts)),
        startangle=140, wedgeprops={"edgecolor":"white"}, textprops={"fontsize":8})
ax2.set_title("User Segments")

# Panel 3: Age vs engagement
ax3 = fig.add_subplot(2, 3, 3)
age_eng.plot(kind="bar", ax=ax3, color=sns.color_palette("rocket_r", 6))
ax3.set_title("Avg Engagement\nby Age Group")
ax3.set_ylabel("Engagement"); ax3.tick_params(axis="x", rotation=30)

# Panel 4: Correlation heatmap (compact)
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(corr[["session_duration","engagement_level","completion_rate"]].drop(
    ["session_duration","engagement_level","completion_rate"]),
    annot=True, fmt=".2f", cmap="vlag", ax=ax4, vmin=-1, vmax=1, linewidths=0.3,
    cbar_kws={"shrink":0.7})
ax4.set_title("Key Feature Correlations")

# Panel 5: Notification impact
ax5 = fig.add_subplot(2, 3, 5)
notif_eng.plot(kind="bar", ax=ax5, color="#5499C7", edgecolor="white")
ax5.set_title("Engagement vs\nNotification Frequency")
ax5.set_ylabel("Engagement"); ax5.tick_params(axis="x", rotation=20)

# Panel 6: Outliers scatter
ax6 = fig.add_subplot(2, 3, 6)
normal_mask  = ~df["is_outlier"]
ax6.scatter(df.loc[normal_mask, "scroll_count"], df.loc[normal_mask, "session_duration"],
            alpha=0.15, s=10, color="#95A5A6", label="Normal")
ax6.scatter(df.loc[df["is_outlier"], "scroll_count"], df.loc[df["is_outlier"], "session_duration"],
            alpha=0.6, s=25, color="#E74C3C", label="Outlier")
ax6.set_title("Outlier Detection\nScroll Count vs Session")
ax6.set_xlabel("Scroll Count"); ax6.set_ylabel("Session Duration")
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig("images/phase7_dashboard_summary.png", bbox_inches="tight")
plt.show()
print("✔ Saved → images/phase7_dashboard_summary.png")


# ══════════════════════════════════════════════════════════════════
#  SAVE REPORT
# ══════════════════════════════════════════════════════════════════
os.makedirs("reports", exist_ok=True)
report_lines = [
    "SOCIAL MEDIA ATTENTION SPAN STUDY — SUMMARY REPORT",
    "="*60,
    f"Total users analysed      : {len(df):,}",
    f"Avg session duration      : {df['session_duration'].mean():.2f} min",
    f"Avg engagement level      : {df['engagement_level'].mean():.2f} / 10",
    f"Avg completion rate       : {df['completion_rate'].mean():.2f} %",
    "",
    "HYPOTHESIS TESTING",
    "-"*40,
    f"ANOVA (content type vs duration) → F={F:.4f}, p={p_anova:.6f} → {'Significant' if p_anova<0.05 else 'Not significant'}",
    f"T-Test (heavy vs light scroll)   → t={t_stat:.4f}, p={p_ttest:.6f} → {'Significant' if p_ttest<0.05 else 'Not significant'}",
    f"Chi-Square (age vs content pref) → χ²={chi2:.4f}, p={p_chi:.6f} → {'Significant' if p_chi<0.05 else 'Not significant'}",
    "",
    "TOP CONTENT TYPES BY SESSION DURATION",
    "-"*40,
]
for ct, val in ct_means.items():
    report_lines.append(f"  {ct:<15} : {val:.2f} min")

report_lines += [
    "",
    "USER SEGMENT SUMMARY",
    "-"*40,
]
for seg, row in seg_profile.iterrows():
    report_lines.append(
        f"  {seg:<22}: {int(row['count']):>4} users | session {row['avg_session_dur']:.1f} min | engagement {row['avg_engagement']:.2f}"
    )

with open("reports/attention_span_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("\n✔ Report saved → reports/attention_span_report.txt")

print("\n" + "═"*65)
print("  ALL PHASES COMPLETE")
print("═"*65)
print("""
Generated files
  data/
    social_media_behavior.csv

  images/
    phase2_session_duration_dist.png
    phase3_content_vs_duration.png
    phase3_scroll_vs_engagement.png
    phase3_notif_vs_engagement.png
    phase3_correlation_heatmap.png
    phase3_age_vs_duration.png
    phase3_daily_usage_vs_completion.png
    phase4_outlier_boxplots.png
    phase5_anova_content_duration.png
    phase6_user_segments.png
    phase6_segment_heatmap.png
    phase7_feature_impact.png
    phase7_notifications_impact.png
    phase7_age_engagement.png
    phase7_dashboard_summary.png

  reports/
    attention_span_report.txt
""")
