# Data Analysis — Reference Corpus

## Descriptive Statistics
Measures of central tendency: mean (sum/n — sensitive to outliers), median (middle value — robust), mode (most frequent). Measures of spread: range, variance (Σ(xᵢ-x̄)²/n), standard deviation (√variance), IQR (Q3-Q1). Percentiles and quartiles: Q1=25th, Q2=50th (median), Q3=75th. Coefficient of variation: (SD/mean)×100 — useful to compare variability across different scales.

## Data Visualization
Histogram: shows distribution shape of a single numeric variable (bins). Boxplot: shows median, IQR, whiskers (1.5×IQR), and outliers. Scatter plot: relationship between two numeric variables. Line chart: time series or ordered data. Bar chart: comparing categories. Heatmap: correlation matrix. Pair plot: scatter plot matrix for multiple variables. Seaborn and matplotlib are standard Python libraries.

## Probability Basics
Classical probability: P(A) = favorable outcomes / total outcomes. Addition rule: P(A∪B) = P(A) + P(B) - P(A∩B). Multiplication rule: P(A∩B) = P(A)P(B|A). Conditional probability: P(A|B) = P(A∩B)/P(B). Independence: P(A|B) = P(A). Bayes' theorem: P(A|B) = P(B|A)P(A)/P(B) — updates prior P(A) with likelihood P(B|A) to get posterior P(A|B).

## Distributions
Normal (Gaussian): symmetric bell curve, defined by μ and σ. 68-95-99.7 rule. Z-score: (x-μ)/σ. Standard normal: N(0,1). Binomial: counts successes in n independent trials with probability p. Poisson: counts rare events in fixed time/space with rate λ. Central Limit Theorem: sampling distribution of the mean approaches normal as n→∞ regardless of population distribution (key for inference).

## Hypothesis Testing
Null hypothesis H₀ (no effect), alternative H₁. p-value: probability of observing data at least this extreme if H₀ were true — NOT the probability H₀ is true. Significance level α (typically 0.05). Reject H₀ if p < α. Type I error: false positive (reject true H₀), probability = α. Type II error: false negative (fail to reject false H₀), probability = β. Power = 1-β. One-sample t-test, independent two-sample t-test, paired t-test. Chi-squared test for categorical data.

## Correlation and Regression
Pearson correlation r: measures linear association, range [-1,1]. Spearman ρ: rank-based, robust to outliers. Simple linear regression: ŷ = b₀ + b₁x. Slope b₁ = Cov(X,Y)/Var(X). Intercept b₀ = ȳ - b₁x̄. R²: proportion of variance in y explained by x. Residual = y - ŷ. Key principle: correlation ≠ causation. Multiple regression: multiple predictors, controlling for confounders.

## Pandas Basics
`pd.read_csv()` loads data. `df.head()`, `df.info()`, `df.describe()` for quick inspection. `df['col']` or `df.loc[:, 'col']` selects a column. Boolean indexing: `df[df['col'] > 5]`. `df.groupby('col').agg({'val': 'mean'})` — split-apply-combine. `df.merge(other, on='key')` for joins. `df.pivot_table()` for cross-tabulations. `df.sort_values()`, `df.drop_duplicates()`, `df.fillna()`.

## Data Cleaning
Missing values: `df.isnull().sum()` to count. Drop with `df.dropna()`. Impute with mean/median (numeric) or mode (categorical). Outliers: Z-score >3, IQR method (1.5×IQR fence). Duplicates: `df.drop_duplicates()`. Categorical encoding: `pd.get_dummies()` for one-hot, `OrdinalEncoder` for ordered. Data type fixing: `df['col'].astype(int)`. String cleaning: `.str.strip()`, `.str.lower()`.

## Exploratory Data Analysis
Systematic EDA process: (1) understand shape and data types, (2) summarize each variable (distribution, range, missing), (3) examine relationships between variables (correlation matrix, pair plot), (4) identify anomalies (outliers, unexpected distributions), (5) formulate hypotheses. Document findings with annotated plots. Skewness: positive (right tail) → mean > median > mode; negative (left tail) → opposite. Multicollinearity: two predictors highly correlated (r > 0.9) cause instability in regression coefficients.

## Statistical Reporting
Report effect sizes alongside p-values: Cohen's d for mean differences (small=0.2, medium=0.5, large=0.8), r for correlations, η² for ANOVA. Confidence intervals: 95% CI means "if we repeat the study 100 times, 95 of the CIs would contain the true parameter" — NOT "95% probability the true value is in this interval." Report CI width (precision) alongside point estimates. Be transparent about multiple comparisons (Bonferroni correction). Distinguish statistical from practical significance.
