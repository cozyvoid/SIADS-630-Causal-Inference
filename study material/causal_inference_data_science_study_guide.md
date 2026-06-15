---
title: "SIADS 630 Causal Inference for Data Science - Study Guide"
author: "Prepared for Remarkable / eInk study"
date: "Summer 2026"
geometry: margin=0.65in
fontsize: 11pt
colorlinks: true
linkcolor: black
urlcolor: black
---

\newpage

# How to Use This Guide

This guide is built for two uses: quick navigation on an eInk tablet and active recall before quizzes, assignments, and oral explanations. Each method section follows the same pattern:

1. **What problem it solves**
2. **Identification assumptions**
3. **Core formula**
4. **Python implementation pattern**
5. **How to explain results in words**
6. **Common mistakes**
7. **Fill-in practice box**

Use the blank spaces as places to write with the Remarkable pen.

**Core question of the course:** How can we estimate a causal effect when treatment assignment is not fully under our control?

**Five main observational / quasi-experimental strategies:**

| Method | Treatment assignment logic | Best when... | Main threat |
|---|---|---|---|
| Controlled regression | Compare treated/control after adjusting for observed covariates | Key confounders are observed | Omitted variable bias |
| Matching | Compare treated/control units with similar covariates | Treated and control units overlap on observed covariates | Hidden confounding, bad matches |
| Instrumental variables | Use a source of variation that shifts treatment but does not directly shift outcome | Treatment is endogenous, but a valid instrument exists | Invalid exclusion restriction or weak first stage |
| Regression discontinuity | Compare units just above/below a cutoff | Treatment changes sharply or probabilistically at a threshold | Manipulation around cutoff, functional form |
| Differences-in-differences | Compare pre/post changes in treatment vs control groups | Groups would have followed parallel trends without treatment | Non-parallel trends |

\newpage

# Course Map from Lectures, Readings, and Assignments

## Module 1: Causal Thinking, Selection Bias, Statistical Inference, Randomized Experiments

**Lectures/readings shown:** introduction to causal inference, selection bias, statistical inference, randomized experiments.

**Main assignment context:** labor market discrimination using observational survey data and randomized resume experiment data.

**Skills to master:**

- Distinguish correlation from causation.
- Define treatment, outcome, treated group, control group, potential outcomes, and counterfactuals.
- Explain selection bias using covariate imbalance.
- Use t-tests and group means to assess balance and outcome differences.
- Explain why random assignment supports causal identification.

## Module 2: Matching, Regression, Regression and Causality

**Lectures/readings shown:** matching, regression, regression and causality.

**Main assignment context:** Nike Vaporfly shoes and marathon race times.

**Skills to master:**

- Log-transform outcomes for approximate percent interpretations.
- Estimate naive mean differences.
- Use nearest-neighbor matching and propensity score weighting/matching.
- Run robust OLS regressions.
- State and apply the omitted variable bias formula.
- Explain why adding a control variable changes the treatment coefficient.

## Module 3: Instrumental Variables and Two-Stage Least Squares

**Lectures/readings shown:** instrumental variables, IV with constant effects, IV with heterogeneous effects, two-stage least squares.

**Main assignment context:** Coursera encouragement message as an instrument for binge learning behavior.

**Skills to master:**

- Explain why treatment can be endogenous.
- Define instrument, endogenous treatment, first stage, reduced form, ITT, and LATE.
- Check first-stage relevance.
- Explain exclusion restriction and monotonicity.
- Calculate Wald/IV estimate manually.
- Estimate 2SLS with robust standard errors.

## Module 4: Regression Discontinuity and Differences-in-Differences

**Lectures/readings shown:** regression discontinuity designs, sharp RD, fuzzy RD, module readings, differences-in-differences, DD regression.

**Main assignment contexts:** angel investor vote cutoff for startup support; Michigan cigarette tax policy and smoking during pregnancy.

**Skills to master:**

- Define running variable, cutoff, treatment indicator, and bandwidth.
- Re-center running variables at the cutoff.
- Use interactions or polynomial terms to model trends around the cutoff.
- Interpret the discontinuity coefficient as a local causal effect.
- Evaluate pre-treatment parallel trends.
- Calculate DiD manually using a 2-by-2 table.
- Estimate DiD using an interaction regression.

\newpage

# 1. Causal Inference Foundations

## 1.1 Core Vocabulary

| Term | Meaning | Course example |
|---|---|---|
| Unit | Entity receiving or not receiving treatment | Resume, runner, learner, startup, mother |
| Treatment $D_i$ | Exposure/intervention of interest | Black-sounding name, Vaporfly shoes, binge behavior, investor support, tax policy |
| Outcome $Y_i$ | Result we want to explain | Callback, race time, course completion, revenue growth, smoking |
| Potential outcome | Outcome under a possible treatment state | $Y_i(1)$ if treated, $Y_i(0)$ if untreated |
| Counterfactual | Potential outcome we do not observe for the same unit | What would this treated runner's time have been without Vaporfly? |
| Treatment effect | Difference between potential outcomes | $Y_i(1)-Y_i(0)$ |
| ATE | Average treatment effect across all units | Average effect of Vaporfly for all runners |
| ATT | Average treatment effect among treated units | Effect of Vaporfly among runners who wore Vaporfly |

## 1.2 Fundamental Problem of Causal Inference

For each unit, we observe only one potential outcome:

$$Y_i = D_iY_i(1) + (1-D_i)Y_i(0)$$

We never observe both $Y_i(1)$ and $Y_i(0)$ for the same unit at the same time. This missing counterfactual is why causal inference is difficult.

**Plain-English version:** We can see what happened, but we cannot directly see what would have happened to the same exact person/unit under the opposite treatment condition.

## 1.3 Correlation vs Causation

A difference in average outcomes between treated and untreated groups is not automatically causal:

$$E[Y|D=1] - E[Y|D=0]$$

This combines both the true treatment effect and selection bias.

A common decomposition is:

$$E[Y|D=1] - E[Y|D=0] = ATT + \left(E[Y(0)|D=1] - E[Y(0)|D=0]\right)$$

The second term is selection bias: the treated and untreated groups may have differed even without treatment.

## 1.4 Selection Bias

Selection bias occurs when treatment status is related to other characteristics that also affect the outcome.

**Examples:**

- Runners who buy expensive Vaporfly shoes may differ by age, income, experience, or seriousness.
- Students who binge course content may be more motivated than students who do not binge.
- States that adopt a policy may differ from states that do not adopt it.

**Quick diagnostic question:** Would treated and untreated units have had the same average outcome if nobody received treatment?

Write your own example of selection bias:

______________________________________________________________________________

______________________________________________________________________________

\newpage

# 2. Statistical Inference for Causal Work

## 2.1 Group Means

Many assignments begin with comparing group means:

$$\bar{Y}_{D=1} - \bar{Y}_{D=0}$$

In Python:

```python
data.groupby("treatment")["outcome"].mean()
```

Interpretation template:

> The treated group has an average outcome of ___, while the control group has an average outcome of ___. The raw difference is ___. This is descriptive unless treatment was randomly assigned or identification assumptions are credible.

## 2.2 Two-Sided t-Test

A t-test is often used to test whether two group means differ.

**Null hypothesis:**

$$H_0: \mu_1 = \mu_0$$

**Alternative hypothesis:**

$$H_A: \mu_1 \ne \mu_0$$

**Decision rule using p-value:**

- If p-value $< 0.05$, reject $H_0$ at the 5% level.
- If p-value $\ge 0.05$, fail to reject $H_0$ at the 5% level.

**Decision rule using t-statistic for large samples:**

- Reject $H_0$ if $t < -1.96$ or $t > 1.96$.
- Fail to reject $H_0$ if $-1.96 \le t \le 1.96$.

Python pattern:

```python
from scipy import stats

g1 = data.loc[data["group"] == 1, "variable"]
g0 = data.loc[data["group"] == 0, "variable"]

ttest_result = stats.ttest_ind(g1, g0, equal_var=False)
print(ttest_result.statistic, ttest_result.pvalue)
```

## 2.3 Balance Tests

Balance means treated and control groups look similar on pre-treatment covariates.

| If covariates are balanced... | If covariates are imbalanced... |
|---|---|
| Groups are more comparable | Selection bias is more likely |
| Randomization looks credible | Need controls, matching, or another design |
| Raw outcome differences are more interpretable | Raw differences may be confounded |

**Important:** Balance on observed variables does not prove balance on unobserved variables.

## 2.4 Cross-Tabulation for Binary/Categorical Variables

```python
pd.crosstab(data["treatment"], data["covariate"])
```

Use this for variables like gender, race assignment, computer skills, education category, or treatment group.

Fill-in interpretation:

> The cross-tabulation shows that ______ is / is not balanced across treatment groups because ______. This matters because ______.

______________________________________________________________________________

\newpage

# 3. Randomized Experiments

## 3.1 Experimental Ideal

Random assignment makes treatment independent of potential outcomes:

$$D_i \perp (Y_i(1), Y_i(0))$$

This means treated and control groups should be comparable in expectation.

**Key idea:** Randomization breaks the link between treatment status and confounders.

## 3.2 Resume Experiment Example

In the randomized resume experiment, names were randomly assigned to otherwise similar resumes. Treatment was whether the resume had a Black-sounding name. Outcome was whether the resume received a callback.

| Object | Example |
|---|---|
| Unit | Resume |
| Treatment | Black-sounding name indicator |
| Control | White-sounding name indicator |
| Outcome | Callback indicator |
| Causal question | Does perceived race affect callbacks? |

Because race-signaling names were randomly assigned, differences in callback rates are more plausibly causal than in the observational survey data.

## 3.3 Potential Outcomes in Experiments

For resume $i$:

- $Y_i^{black}$ = callback outcome if assigned a Black-sounding name.
- $Y_i^{white}$ = callback outcome if assigned a White-sounding name.

Only one is observed. The other is counterfactual.

## 3.4 Assignment 1 Checklist

- [ ] Identify treatment variable.
- [ ] Identify outcome variable.
- [ ] Use t-tests for covariate balance.
- [ ] Explain p-value decision rule.
- [ ] Distinguish observational data from randomized experiment data.
- [ ] Explain why observational differences are not automatically causal.
- [ ] Explain why randomized differences are more causally interpretable.

**Oral-exam answer template:**

> In the observational data, a racial difference in employment cannot be interpreted as causal by itself because groups may differ on education, experience, or other confounders. In the randomized resume experiment, the race signal is randomly assigned to resumes, so the treatment is not selected by the applicant. That makes the treatment and control groups more comparable and gives the callback difference a causal interpretation under the design assumptions.

\newpage

# 4. Ordinary Least Squares Regression

## 4.1 Basic OLS Model

A simple regression model:

$$Y_i = \alpha + \beta D_i + u_i$$

A controlled regression model:

$$Y_i = \alpha + \beta D_i + \gamma X_i + u_i$$

Where:

- $Y_i$ = outcome.
- $D_i$ = treatment.
- $X_i$ = control variable(s).
- $\beta$ = coefficient of interest.
- $u_i$ = unobserved factors affecting $Y_i$.

## 4.2 What the Coefficient Means

If $D_i$ is binary:

> $\beta$ is the estimated difference in average outcome between treated and untreated units, holding included controls constant.

If the outcome is logged, a coefficient can be interpreted approximately as a percent change:

$$100 \times \beta \approx \text{percent change in }Y$$

More exact conversion:

$$100 \times (e^\beta - 1)$$

## 4.3 Robust Standard Errors

Robust standard errors are used when error variance may not be constant across observations.

Statsmodels pattern:

```python
import statsmodels.formula.api as smf

model = smf.ols(
    "outcome ~ treatment + control1 + control2",
    data=data
).fit()

robust = model.get_robustcov_results(cov_type="HC1")
print(robust.summary())
```

Alternative direct pattern:

```python
robust = smf.ols("Y ~ D + X1 + X2", data=data).fit(cov_type="HC1")
```

## 4.4 How to Read an OLS Table

| Table part | What to focus on | Why it matters |
|---|---|---|
| Dep. Variable | Outcome being predicted | Tells you what the model explains |
| coef column | Estimated effect/direction | Sign and magnitude of each variable |
| treatment row | Main causal coefficient | Usually the answer to the research question |
| std err | Uncertainty in coefficient | Smaller means more precise estimate |
| t or z statistic | Signal relative to noise | Used for significance testing |
| `P>|t|` or `P>|z|` | p-value | Used to judge statistical significance |
| confidence interval | plausible coefficient range | Shows direction and precision |
| R-squared | variance explained | Not the main causal criterion |

**Most important rule:** The treatment coefficient is usually the coefficient of interest, but it is causal only under a credible identification strategy.

## 4.5 Regression Interpretation Template

> The coefficient on ___ is ___. Since the outcome is ___, this means ___. The sign is positive/negative, so the treatment is associated with a higher/lower outcome. The p-value is ___, so at the 5% level we reject/fail to reject the null that the coefficient equals zero. This is/is not causal because ___.

\newpage

# 5. Omitted Variable Bias

## 5.1 What Is OVB?

Omitted variable bias occurs when a variable that affects the outcome is left out of the regression and is correlated with the treatment.

The omitted variable must satisfy two conditions:

1. It affects the outcome $Y$.
2. It is correlated with the treatment $D$.

If either condition fails, omitting it does not bias the treatment coefficient in the same way.

## 5.2 OVB Formula

Suppose the true model is:

$$Y_i = \alpha + \beta D_i + \gamma Z_i + u_i$$

But we estimate:

$$Y_i = \alpha + \tilde{\beta}D_i + e_i$$

The omitted variable bias in the estimated treatment coefficient is:

$$\tilde{\beta} - \beta = \gamma \times \delta$$

Where:

- $\gamma$ = effect of omitted variable $Z$ on outcome $Y$, holding treatment constant.
- $\delta$ = coefficient from regressing omitted variable $Z$ on treatment $D$.

Plain-English version:

> Bias = how much the omitted variable matters for the outcome times how strongly the treatment predicts the omitted variable.

## 5.3 OVB Sign Table

| Effect of omitted variable on outcome $(\gamma)$ | Relationship between treatment and omitted variable $(\delta)$ | Bias sign |
|---|---|---|
| Positive | Positive | Positive |
| Positive | Negative | Negative |
| Negative | Positive | Negative |
| Negative | Negative | Positive |

## 5.4 Vaporfly Example

Outcome: $\ln(\text{race time})$.

Treatment: wearing Vaporfly shoes.

Omitted variable: age.

Reasoning used in the assignment:

- Older runners are expected to be faster in endurance races, so age is expected to have a **negative** effect on log race time: older age -> lower race time.
- Younger runners may be less likely to buy expensive Vaporfly shoes, so Vaporfly runners may be older on average. Regressing age on Vaporfly would likely produce a **positive** coefficient.
- Bias = negative $\times$ positive = **negative bias**.

That means the model omitting age may make Vaporfly look more beneficial than it really is, because part of the lower race time is due to older/faster runners rather than the shoe itself.

## 5.5 Assignment 2 Regression Comparison

If the Vaporfly coefficient becomes less negative after adding age, that is consistent with negative omitted variable bias in the model that omitted age.

Interpretation template:

> In the model without age, the Vaporfly coefficient was more negative. After controlling for age, the coefficient became less negative. This suggests the omitted age variable biased the original coefficient downward, making Vaporfly appear to reduce race time more than it actually does after adjusting for age.

Fill in your own OVB example:

- Treatment: _________________________________
- Outcome: ___________________________________
- Omitted variable: ___________________________
- Effect on outcome: positive / negative
- Relationship with treatment: positive / negative
- Expected bias: positive / negative

\newpage

# 6. Matching

## 6.1 What Problem Matching Solves

Matching tries to construct a comparison group that looks similar to the treated group on observed covariates.

Instead of comparing all treated units to all control units, we compare treated units to control units with similar $X$ values.

## 6.2 Key Assumption: Conditional Independence

Matching relies on selection on observables:

$$Y(1), Y(0) \perp D \mid X$$

Plain-English version:

> After controlling for observed covariates $X$, treatment assignment is as good as random.

This is a strong assumption. Matching does not solve bias from unobserved confounders.

## 6.3 Overlap / Common Support

For matching to work, treated and control units must have comparable covariate values.

$$0 < P(D=1|X) < 1$$

Plain-English version:

> For each type of treated unit, there should be similar untreated units available for comparison.

## 6.4 Nearest-Neighbor Matching

Nearest-neighbor matching pairs each treated unit with the closest control unit according to a distance metric.

For one covariate, such as age, closeness is simple. For multiple covariates, distances can become harder to interpret because of scale differences.

## 6.5 Propensity Score

The propensity score is the probability of receiving treatment given observed covariates:

$$p(X) = P(D=1|X)$$

Instead of matching on many covariates directly, we can match or weight using the propensity score.

## 6.6 Matching Workflow

1. Choose pre-treatment covariates.
2. Estimate propensity scores or distances.
3. Match or weight treated/control observations.
4. Check balance after matching/weighting.
5. Estimate ATE or ATT.
6. Explain that causality still depends on no unobserved confounding.

## 6.7 Python Pattern: Creating Dummies

```python
import numpy as np

data["seasoned"] = np.where(data["marathoner_type"] == "seasoned", 1, 0)
data["enthusiastic"] = np.where(data["marathoner_type"] == "enthusiastic", 1, 0)
```

If `first_timer` is left out, it becomes the baseline category.

## 6.8 Python Pattern: Log Outcome

```python
import numpy as np

data["ln_race_time"] = np.log(data["race_time"])
```

Interpretation:

- Negative coefficient on Vaporfly means lower race time.
- Lower race time means faster marathon completion.
- In a log outcome model, $-0.04$ is approximately a 4% decrease.

## 6.9 Matching Interpretation Template

> Matching estimates the treatment effect by comparing treated and control observations that are similar on observed variables such as ___. This reduces bias from observed confounders, but it does not remove bias from unobserved variables. The estimate is credible only if treatment assignment is independent of potential outcomes after conditioning on the matched covariates.

\newpage

# 7. Controlled Regression as Causal Strategy

## 7.1 Regression as Automated Matching

Controlled regression compares treated and untreated observations while holding observed covariates constant. This can be thought of as an automated version of matching across many covariate patterns.

## 7.2 Controlled Regression Identification Assumption

The main assumption is conditional independence:

$$Y(1), Y(0) \perp D \mid X$$

Same core idea as matching: after controlling for $X$, treatment assignment is as good as random.

## 7.3 What Controls Should Be Included?

Good controls are usually:

- Pre-treatment variables.
- Variables related to both treatment and outcome.
- Confounders that are not caused by treatment.

Avoid controlling for:

- Post-treatment variables.
- Mediators, unless you are estimating a direct effect.
- Colliders.

## 7.4 Regression with Categorical Variables

Use dummy variables. Leave one category out as the reference group.

Example:

```python
model = smf.ols(
    "ln_race_time ~ vaporfly + male + seasoned + enthusiastic",
    data=data
).fit(cov_type="HC1")
```

If `first_timer` is omitted, then coefficients for `seasoned` and `enthusiastic` are interpreted relative to first-time marathoners.

## 7.5 Regression Causal Interpretation Checklist

- [ ] Is treatment clearly defined?
- [ ] Is outcome clearly defined?
- [ ] Are key confounders observed?
- [ ] Are controls pre-treatment?
- [ ] Is there overlap between treated and control units?
- [ ] Are robust standard errors used if appropriate?
- [ ] Is the treatment coefficient interpreted, not just R-squared?

\newpage

# 8. Instrumental Variables

## 8.1 Why IV Is Needed

Instrumental variables are used when treatment is endogenous. Endogeneity means treatment is related to unobserved factors in the error term.

Common sources of endogeneity:

- Omitted variables.
- Reverse causality.
- Measurement error.
- Self-selection into treatment.

## 8.2 IV Setup

| Symbol | Meaning | Coursera example |
|---|---|---|
| $Y$ | Outcome | Completing next course week |
| $D$ | Endogenous treatment | Bingeing course content |
| $Z$ | Instrument | Randomly assigned message |
| $X$ | Controls | Prior week number, minutes, paid enrollment |

## 8.3 IV Assumptions

### 1. Relevance

The instrument must affect the treatment:

$$Cov(Z,D) \ne 0$$

In practice: first-stage coefficient should be nonzero and strong.

### 2. Independence

The instrument should be as good as randomly assigned:

$$Z \perp (Y(1), Y(0))$$

In the Coursera example, random assignment of message helps support this.

### 3. Exclusion Restriction

The instrument affects the outcome only through the treatment.

Coursera version:

> The message can affect completion only by changing binge behavior, not through another channel such as motivation, reminders, emotional encouragement, or direct information about course completion.

### 4. Monotonicity

No defiers.

Coursera version:

> There are no learners who would binge only if they did not receive the message but would not binge if they did receive the message.

## 8.4 Compliance Types

| Type | Behavior |
|---|---|
| Always-takers | Take treatment whether $Z=1$ or $Z=0$ |
| Never-takers | Do not take treatment whether $Z=1$ or $Z=0$ |
| Compliers | Take treatment if $Z=1$, do not if $Z=0$ |
| Defiers | Do opposite of assignment |

With monotonicity, defiers are assumed not to exist.

## 8.5 First Stage

Regress treatment on instrument:

$$D_i = \pi_0 + \pi_1 Z_i + v_i$$

$\pi_1$ tells how much the instrument changes treatment take-up.

Python:

```python
first_stage = smf.ols("binge ~ message", data=data).fit(cov_type="HC1")
print(first_stage.summary())
```

## 8.6 Reduced Form / ITT

Regress outcome on instrument:

$$Y_i = \rho_0 + \rho_1 Z_i + e_i$$

$\rho_1$ is the intention-to-treat effect: effect of being assigned the instrument, regardless of compliance.

```python
reduced_form = smf.ols("complete ~ message", data=data).fit(cov_type="HC1")
```

## 8.7 Wald / IV Estimate

For a single binary instrument and binary treatment:

$$\text{IV Estimate} = \frac{\text{Reduced Form}}{\text{First Stage}} = \frac{\rho_1}{\pi_1}$$

Plain-English version:

> Divide the effect of the instrument on the outcome by the effect of the instrument on the treatment.

## 8.8 LATE

With heterogeneous treatment effects, IV estimates the Local Average Treatment Effect:

$$LATE = E[Y(1)-Y(0)|\text{compliers}]$$

This is the treatment effect for compliers: units whose treatment status is changed by the instrument.

## 8.9 Two-Stage Least Squares

Stage 1:

$$D_i = \pi_0 + \pi_1 Z_i + v_i$$

Stage 2:

$$Y_i = \alpha + \beta \hat{D}_i + u_i$$

2SLS uses predicted treatment variation from the instrument.

Python pattern with `linearmodels`:

```python
from linearmodels.iv import IV2SLS

iv_data = data[["complete", "binge", "message"]].dropna()

iv_model = IV2SLS.from_formula(
    "complete ~ 1 + [binge ~ message]",
    data=iv_data
).fit(cov_type="robust")

print(iv_model.summary)
```

With controls:

```python
iv_model = IV2SLS.from_formula(
    "complete ~ 1 + paid_enroll + prv_wk_nbr + prv_wk_min + [binge ~ message]",
    data=iv_data
).fit(cov_type="robust")
```

## 8.10 IV Oral Explanation Template

> I use the message as an instrument for bingeing because bingeing itself may be selected: more motivated learners may be more likely to binge and complete the course. The instrument must shift bingeing, be randomly assigned or independent of potential outcomes, and affect completion only through bingeing. The first stage checks whether the message changes bingeing. The reduced form checks whether the message changes completion. The IV estimate divides reduced form by first stage and identifies the effect for compliers under monotonicity and exclusion restriction.

\newpage

# 9. Regression Discontinuity Design

## 9.1 What Problem RD Solves

Regression discontinuity uses a cutoff rule that assigns treatment based on a running variable.

Units just below and just above the cutoff are assumed to be comparable, except for treatment status.

## 9.2 RD Vocabulary

| Term | Meaning | Angel investor example |
|---|---|---|
| Running variable | Variable that determines treatment | Number of votes |
| Cutoff | Threshold for treatment | 20 votes |
| Treatment indicator | Whether unit crosses cutoff | $D=1$ if votes >= 20 |
| Bandwidth | Window around cutoff used for estimation | 17 to 23 votes if bandwidth = 3 |
| Discontinuity | Jump in outcome at cutoff | Effect of investor support |

## 9.3 Sharp RD

In sharp RD, treatment switches deterministically at the cutoff:

$$D_i = 1(X_i \ge c)$$

Example: if ventures with at least 20 votes always receive support and those below 20 do not.

## 9.4 Fuzzy RD

In fuzzy RD, crossing the cutoff changes the probability of treatment but does not perfectly determine treatment.

Fuzzy RD uses the cutoff as an instrument for treatment.

## 9.5 Re-Centering the Running Variable

If the cutoff is 20:

```python
data["D"] = np.where(data["votes"] >= 20, 1, 0)
data["vote_c"] = data["votes"] - 20
```

Re-centering makes `vote_c = 0` at the cutoff. Then the coefficient on `D` estimates the jump at the cutoff.

## 9.6 Linear RD with Different Slopes

Model:

$$Y_i = \alpha + \tau D_i + \beta_1 vote\_c_i + \beta_2(D_i \times vote\_c_i) + u_i$$

Python:

```python
rd_model = smf.ols(
    "dau_mau ~ vote_c + D + vote_c:D",
    data=data
).fit(cov_type="HC1")

rd_effect = rd_model.params["D"]
```

Interpretation:

> The coefficient on `D` estimates the jump in the outcome at the cutoff, allowing the slope of the running variable to differ on each side.

## 9.7 Nonlinear RD

If the relationship between running variable and outcome is nonlinear, add polynomial terms:

```python
data["vote_c_sq"] = data["vote_c"] ** 2

rd_poly = smf.ols(
    "revenue_g ~ vote_c + vote_c_sq + D",
    data=data
).fit(cov_type="HC1")
```

## 9.8 Bandwidth Restriction

Instead of modeling the full nonlinear relationship, restrict to observations close to the cutoff:

```python
bw = data.loc[(data["votes"] >= 17) & (data["votes"] <= 23)].copy()

rd_bw = smf.ols(
    "revenue_g ~ vote_c + D",
    data=bw
).fit(cov_type="HC1")
```

Tradeoff:

| Wider bandwidth | Narrower bandwidth |
|---|---|
| More observations | Fewer observations |
| More precision | Less precision |
| More risk of functional-form bias | Units more comparable near cutoff |

## 9.9 RD Identification Assumptions

- Units cannot precisely manipulate the running variable around the cutoff.
- Potential outcomes are continuous at the cutoff without treatment.
- Units just below and above cutoff are comparable.
- The model uses an appropriate bandwidth / functional form.

## 9.10 RD Interpretation Template

> The RD estimate compares units just above and just below the cutoff. Because treatment changes at the cutoff, a discontinuous jump in the outcome at that point is interpreted as the local causal effect of treatment for units near the threshold. This interpretation depends on no precise manipulation of the running variable and smooth potential outcomes around the cutoff.

\newpage

# 10. Differences-in-Differences

## 10.1 What Problem DiD Solves

Differences-in-differences compares changes over time between a treated group and a control group.

It removes:

- Time-invariant differences between groups.
- Common time shocks affecting both groups.

## 10.2 DiD Setup

| Group | Pre-treatment | Post-treatment | Change |
|---|---:|---:|---:|
| Treatment group | $Y_{T,pre}$ | $Y_{T,post}$ | $Y_{T,post}-Y_{T,pre}$ |
| Control group | $Y_{C,pre}$ | $Y_{C,post}$ | $Y_{C,post}-Y_{C,pre}$ |

DiD estimate:

$$\left(Y_{T,post}-Y_{T,pre}\right) - \left(Y_{C,post}-Y_{C,pre}\right)$$

Plain-English version:

> Compare how much the treated group changed to how much the control group changed.

## 10.3 Michigan Tax Example

Treatment group: Michigan.

Control group: Iowa.

Treatment: cigarette tax increase in Michigan.

Outcome: smoking during pregnancy.

Pre-treatment years: years before tax hike.

Post-treatment: year after tax hike.

## 10.4 Parallel Trends Assumption

The key assumption is that, without treatment, the treatment and control groups would have followed parallel trends.

This does **not** require equal levels. It requires similar trends.

| Good for DiD | Bad for DiD |
|---|---|
| Groups have similar pre-treatment slopes | Groups are already trending differently |
| No other simultaneous policy shock affects only one group | Another event happens at the same time as treatment |
| Control group is a plausible counterfactual | Control group is structurally very different |

## 10.5 DiD Regression

Model:

$$Y_{it} = \alpha + \beta_1 Treat_i + \beta_2 Post_t + \beta_3(Treat_i \times Post_t) + u_{it}$$

The DiD estimate is $\beta_3$.

Python:

```python
data["treat"] = np.where(data["state"] == 26, 1, 0)  # Michigan
data["post"] = np.where(data["year"] == 3, 1, 0)

model = smf.ols(
    "smoked ~ treat + post + treat:post",
    data=data
).fit(cov_type="HC1")

did_effect = model.params["treat:post"]
```

## 10.6 Manual DiD Calculation in Python

```python
cell_means = data.groupby(["treat", "post"])["smoked"].mean()

control_pre = cell_means.loc[(0, 0)]
control_post = cell_means.loc[(0, 1)]
treat_pre = cell_means.loc[(1, 0)]
treat_post = cell_means.loc[(1, 1)]

did = (treat_post - treat_pre) - (control_post - control_pre)
```

## 10.7 DiD Interpretation Template

> The treatment group changed by ___ from pre to post. The control group changed by ___. The difference between these changes is ___, which is the DiD estimate. Under the parallel trends assumption, this estimates the causal effect of the policy on the outcome.

\newpage

# 11. Method Selection Decision Guide

Use this page when deciding which causal method fits a prompt.

## 11.1 Quick Decision Tree

**Was treatment randomized?**

- Yes -> randomized experiment; compare means/regression can be causal.
- No -> continue.

**Are all key confounders observed?**

- Yes -> controlled regression or matching may be appropriate.
- No -> continue.

**Is there a valid instrument that shifts treatment?**

- Yes -> instrumental variables / 2SLS.
- No -> continue.

**Is treatment assigned by a cutoff?**

- Yes -> regression discontinuity.
- No -> continue.

**Is there a treated group, control group, pre period, and post period?**

- Yes -> differences-in-differences.
- No -> identification may be weak; need stronger design or more assumptions.

## 11.2 Method Comparison Table

| Method | Coefficient of interest | Core assumption | Main diagnostic |
|---|---|---|---|
| RCT | Treatment coefficient or mean difference | Random assignment | Balance checks |
| Regression | Treatment coefficient | No omitted confounders after controls | Sensitivity to controls |
| Matching | ATE/ATT after matching/weighting | Selection on observables | Covariate balance and overlap |
| IV | Endogenous treatment coefficient | Relevance, independence, exclusion, monotonicity | First stage strength |
| RD | Jump at cutoff | No manipulation, continuity | Plot around cutoff, bandwidth checks |
| DiD | Treatment x post interaction | Parallel trends | Pre-treatment trend graph |

## 11.3 Common Prompt Clues

| Prompt clue | Likely method |
|---|---|
| "randomly assigned" | Randomized experiment |
| "groups differ on covariates" | Selection bias, balance, controls, matching |
| "omitted variable" | OVB / controlled regression |
| "encouragement" or "instrument" | IV |
| "cutoff", "threshold", "score" | RD |
| "before and after", "treated and control group" | DiD |

\newpage

# 12. Python Code Cookbook

## 12.1 Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
```

Optional:

```python
from linearmodels.iv import IV2SLS
```

## 12.2 Group Means and Differences

```python
means = data.groupby("D")["Y"].mean()
mean_diff = float(round(means.loc[1] - means.loc[0], 4))
```

## 12.3 t-Test

```python
g1 = data.loc[data["D"] == 1, "X"]
g0 = data.loc[data["D"] == 0, "X"]
ttest = stats.ttest_ind(g1, g0, equal_var=False)
```

## 12.4 Crosstab

```python
pd.crosstab(data["D"], data["category"])
```

## 12.5 Create Binary Treatment

```python
data["D"] = np.where(data["running_var"] >= cutoff, 1, 0)
```

## 12.6 OLS with Robust Standard Errors

```python
model = smf.ols("Y ~ D + X1 + X2", data=data).fit(cov_type="HC1")
coef = float(round(model.params["D"], 4))
```

If your assignment specifically asks for `.get_robustcov_results()`:

```python
model = smf.ols("Y ~ D + X1 + X2", data=data).fit()
robust = model.get_robustcov_results(cov_type="HC1")
```

## 12.7 Extract Coefficients from Formula Model

```python
coef = model.params["D"]
pval = model.pvalues["D"]
se = model.bse["D"]
```

For robust results objects from `.get_robustcov_results()`, names may not always index cleanly. Use:

```python
names = model.model.exog_names
coef_dict = dict(zip(names, robust.params))
d_coef = coef_dict["D"]
```

## 12.8 Log Outcome

```python
data["ln_Y"] = np.log(data["Y"])
```

Approximate percent interpretation:

```python
100 * coef
```

Exact percent interpretation:

```python
100 * (np.exp(coef) - 1)
```

## 12.9 IV / 2SLS

```python
iv_data = data[["Y", "D", "Z", "X1", "X2"]].dropna()

iv = IV2SLS.from_formula(
    "Y ~ 1 + X1 + X2 + [D ~ Z]",
    data=iv_data
).fit(cov_type="robust")
```

## 12.10 RD Plot

```python
plt.scatter(data["vote_c"], data["Y"], alpha=0.4)
plt.axvline(0, linestyle="--")
plt.xlabel("running variable centered at cutoff")
plt.ylabel("outcome")
plt.show()
```

## 12.11 DiD Plot of Pre-Trends

```python
pre = data.loc[data["post"] == 0]
rates = pre.groupby(["group", "year"])["Y"].mean().reset_index()

for group, temp in rates.groupby("group"):
    plt.plot(temp["year"], temp["Y"], marker="o", label=group)

plt.legend()
plt.xlabel("year")
plt.ylabel("outcome rate")
plt.show()
```

\newpage

# 13. Formula Sheet

| Concept | Formula | What it means |
|---|---|---|
| Individual treatment effect | $Y_i(1)-Y_i(0)$ | Difference between two potential outcomes for one unit |
| ATE | $E[Y(1)-Y(0)]$ | Average treatment effect for all units |
| ATT | $E[Y(1)-Y(0)|D=1]$ | Average treatment effect among treated units |
| Observed outcome | $Y_i = D_iY_i(1)+(1-D_i)Y_i(0)$ | We observe only one potential outcome |
| Raw difference | $E[Y|D=1]-E[Y|D=0]$ | Descriptive difference; causal only under assumptions |
| OLS | $Y_i=\alpha+\beta D_i+\gamma X_i+u_i$ | Regression with treatment and controls |
| OVB | $Bias = \gamma\delta$ | Bias from omitted variable equals two relationships multiplied |
| Propensity score | $p(X)=P(D=1|X)$ | Probability of treatment given covariates |
| First stage | $D_i=\pi_0+\pi_1Z_i+v_i$ | Effect of instrument on treatment |
| Reduced form | $Y_i=\rho_0+\rho_1Z_i+e_i$ | Effect of instrument on outcome |
| IV/Wald | $\rho_1/\pi_1$ | Reduced form divided by first stage |
| 2SLS stage 2 | $Y_i=\alpha+\beta\hat{D}_i+u_i$ | Use predicted treatment from instrument |
| Sharp RD | $D_i=1(X_i\ge c)$ | Cutoff deterministically assigns treatment |
| RD model | $Y_i=\alpha+\tau D_i+f(X_i-c)+u_i$ | $\tau$ is jump at cutoff |
| DiD | $(Y_{T,post}-Y_{T,pre})-(Y_{C,post}-Y_{C,pre})$ | Difference in changes |
| DiD regression | $Y=\alpha+\beta_1Treat+\beta_2Post+\beta_3Treat\times Post+u$ | $\beta_3$ is DiD estimate |

\newpage

# 14. Common Mistakes and Fixes

## 14.1 Conceptual Mistakes

| Mistake | Fix |
|---|---|
| Calling any regression coefficient causal | Explain identification assumptions first |
| Focusing on R-squared instead of treatment coefficient | Treatment coefficient answers causal question |
| Treating balance tests as proof of randomization | Balance supports but does not prove design validity |
| Controlling for post-treatment variables | Use pre-treatment confounders unless estimating direct effects |
| Forgetting the counterfactual | Always ask: compared to what would have happened otherwise? |
| Saying DiD requires equal pre levels | DiD requires parallel trends, not equal levels |
| Saying IV estimates ATE for everyone | With heterogeneous effects, IV estimates LATE for compliers |
| Ignoring first stage | IV requires instrument relevance |
| Using all RD data without checking functional form | Consider bandwidth and plots around cutoff |

## 14.2 Python Mistakes

| Mistake | Fix |
|---|---|
| Comparing strings/numbers incorrectly | Check `data.dtypes` |
| Forgetting `.copy()` after subsetting | Use `subset = data.loc[condition].copy()` |
| Incorrect interaction syntax | Use `treat:post` or `treat * post` in formulas |
| Extracting wrong coefficient | Check `model.params.index` or `model.model.exog_names` |
| Not dropping NAs before IV | Use `.dropna()` on needed variables |
| Forgetting robust SE request | Use `cov_type="HC1"` or `.get_robustcov_results()` |
| Logging zero or negative outcome | Check min value before `np.log()` |
| Treating p-value as probability null is true | P-value is probability of data as extreme under null |

\newpage

# 15. Assignment-Specific Review Sheets

## 15.1 Assignment 1: Observational Survey vs Randomized Resume Experiment

**Main question:** Can we conclude racial discrimination from observational or experimental data?

### Observational survey data

- Treatment/group variable: ___________________________
- Outcome variable: _________________________________
- Covariates checked: ________________________________
- Were covariates balanced? __________________________
- Why this matters: _________________________________

Key answer idea:

> Observational differences may reflect discrimination, but they may also reflect confounding variables. Without random assignment or a credible causal design, the raw difference is not automatically causal.

### Randomized resume data

- Treatment variable: ________________________________
- Outcome variable: _________________________________
- Why randomization helps: ___________________________
- Counterfactual for one resume: ______________________

Key answer idea:

> Since names were randomly assigned to resumes, the treatment should be independent of resume characteristics. Therefore, callback differences by assigned name are more causally interpretable.

## 15.2 Assignment 2: Vaporfly, Matching, Regression, OVB

**Main question:** Do Vaporfly shoes make runners faster?

- Outcome: log race time.
- Treatment: Vaporfly indicator.
- Important controls: age, gender, marathoner type.
- Matching method: nearest neighbor / propensity score.
- Regression issue: omitted age variable.

OVB fill-in:

- Omitted variable: age.
- Effect of age on log race time: ____________________
- Relationship between Vaporfly and age: _____________
- Expected bias sign: _______________________________
- What happened after adding age? ____________________

## 15.3 Assignment 3: Coursera IV

**Main question:** Does bingeing cause learners to complete the next course week?

- Endogenous treatment: bingeing.
- Outcome: course completion.
- Instrument: randomized message.

Assumptions fill-in:

- Relevance means: _________________________________
- Exclusion restriction means: _______________________
- Independence means: ______________________________
- Monotonicity means: ______________________________

Manual IV:

- First stage coefficient: ___________________________
- Reduced form coefficient: __________________________
- IV estimate = reduced form / first stage: ___________

## 15.4 Assignment 4 Part 1: Angel Investor RD

**Main question:** Does angel investor support improve startup outcomes?

- Running variable: votes.
- Cutoff: 20 votes.
- Treatment: support from angel group.
- Outcomes: DAU/MAU and revenue growth.

RD fill-in:

- Why re-center running variable? ____________________
- Coefficient of interest: ___________________________
- Bandwidth used: __________________________________
- Tradeoff of smaller bandwidth: _____________________

## 15.5 Assignment 4 Part 2: Michigan Cigarette Tax DiD

**Main question:** Did Michigan's cigarette tax reduce smoking during pregnancy?

- Treatment group: Michigan.
- Control group: Iowa.
- Pre-period: years before tax hike.
- Post-period: after tax hike.
- Outcome: smoked during pregnancy.

DiD fill-in:

| Group | Pre | Post | Change |
|---|---:|---:|---:|
| Michigan | _____ | _____ | _____ |
| Iowa | _____ | _____ | _____ |
| Difference in changes |  |  | _____ |

Parallel trends explanation:

______________________________________________________________________________

______________________________________________________________________________

\newpage

# 16. Oral Exam Response Bank

## 16.1 What is causal inference?

> Causal inference is about estimating what would happen to the same unit under different treatment conditions. Since we cannot observe both potential outcomes for the same unit, we need research designs or assumptions that help construct a credible counterfactual.

## 16.2 What is selection bias?

> Selection bias occurs when treated and untreated groups differ for reasons other than treatment, and those differences also affect the outcome. In that case, the raw treated-control difference mixes the treatment effect with pre-existing differences.

## 16.3 Why does random assignment matter?

> Random assignment makes treatment independent of potential outcomes in expectation. This means treated and control groups should be similar except for treatment, so outcome differences can be interpreted causally under the experiment's assumptions.

## 16.4 What does a regression coefficient mean?

> In a controlled regression, the treatment coefficient estimates the difference in the outcome associated with treatment while holding included controls constant. It is causal only if the included controls are sufficient to remove confounding and the model is appropriate.

## 16.5 What is omitted variable bias?

> Omitted variable bias occurs when a left-out variable affects the outcome and is correlated with treatment. The bias equals the effect of the omitted variable on the outcome multiplied by the relationship between treatment and the omitted variable.

## 16.6 What is the difference between matching and regression?

> Both adjust for observed covariates. Matching tries to create comparable treated and control groups by pairing or weighting similar observations, while regression adjusts parametrically by holding covariates constant. Both rely on selection on observables.

## 16.7 What makes an instrument valid?

> A valid instrument must be relevant, independent of potential outcomes, and satisfy exclusion restriction. With heterogeneous effects, monotonicity is also needed to interpret the estimate as LATE for compliers.

## 16.8 What is 2SLS doing intuitively?

> 2SLS first isolates the part of treatment variation explained by the instrument. Then it uses that predicted treatment variation to estimate the effect on the outcome. This avoids using endogenous variation in treatment.

## 16.9 What is RD doing intuitively?

> RD compares units just below and just above a treatment cutoff. Near the cutoff, units are assumed similar except for treatment assignment, so a jump in the outcome at the cutoff can be interpreted as a local treatment effect.

## 16.10 What is DiD doing intuitively?

> DiD compares how the treated group changes after treatment to how a control group changes over the same period. The control group's change estimates what would have happened to the treated group without treatment, assuming parallel trends.

\newpage

# 17. Active Recall Pages

## 17.1 One-Sentence Definitions

Causal inference: _____________________________________________________________

Counterfactual: ______________________________________________________________

Selection bias: ______________________________________________________________

Random assignment: __________________________________________________________

Omitted variable bias: _______________________________________________________

Propensity score: ____________________________________________________________

Instrument: _________________________________________________________________

Exclusion restriction: _______________________________________________________

First stage: _________________________________________________________________

Reduced form: _______________________________________________________________

LATE: ______________________________________________________________________

Running variable: ___________________________________________________________

Bandwidth: __________________________________________________________________

Parallel trends: _____________________________________________________________

## 17.2 Formula Practice

Write the OVB formula:

______________________________________________________________________________

Write the IV/Wald formula:

______________________________________________________________________________

Write the DiD formula:

______________________________________________________________________________

Write the DiD regression:

______________________________________________________________________________

Write the sharp RD treatment rule:

______________________________________________________________________________

## 17.3 Method Identification Practice

For each scenario, name the likely method and the key assumption.

1. A scholarship is awarded to students with test scores above 85.

Method: _______________________ Assumption: _________________________________

2. A state passes a law in 2024 and a neighboring state does not.

Method: _______________________ Assumption: _________________________________

3. A randomized email increases use of a tutoring service.

Method: _______________________ Assumption: _________________________________

4. Treated and untreated patients differ in age, income, and baseline health, all measured.

Method: _______________________ Assumption: _________________________________

5. A randomized resume experiment changes only the name shown on resumes.

Method: _______________________ Assumption: _________________________________

\newpage

# 18. Final Exam / Quiz Checklist

Before answering a causal inference question, write down:

- [ ] What is the unit?
- [ ] What is the treatment?
- [ ] What is the outcome?
- [ ] What is the comparison group?
- [ ] What is the counterfactual?
- [ ] What identification strategy is being used?
- [ ] What coefficient or statistic answers the causal question?
- [ ] What assumptions are required?
- [ ] What would violate those assumptions?
- [ ] Is the answer descriptive, associational, or causal?

## 18.1 Fast Interpretation Templates

**p-value:**

> The p-value is ___. Since it is less/greater than 0.05, I reject/fail to reject the null at the 5% level.

**OLS treatment coefficient:**

> The coefficient on ___ is ___. Holding included controls constant, treated units have an outcome that is ___ higher/lower than control units. This is causal only if ___.

**Log outcome coefficient:**

> Since the outcome is logged, the coefficient of ___ means the treatment is associated with approximately ___ percent higher/lower outcome.

**OVB sign:**

> The omitted variable affects the outcome positively/negatively, and treatment is positively/negatively related to the omitted variable. Multiplying those signs gives ___ bias.

**IV:**

> The IV estimate is the reduced form divided by the first stage. It estimates the effect of treatment for compliers if relevance, independence, exclusion restriction, and monotonicity hold.

**RD:**

> The RD coefficient estimates the jump in the outcome at the cutoff. It is local to units near the threshold.

**DiD:**

> The interaction coefficient is the DiD estimate. It is causal if the treatment and control groups would have followed parallel trends without treatment.

\newpage

# 19. Blank Notes

Topic: ______________________________________________________________________

Key formula:

______________________________________________________________________________

Key assumption:

______________________________________________________________________________

Python pattern:

______________________________________________________________________________

______________________________________________________________________________

Example from assignment:

______________________________________________________________________________

______________________________________________________________________________

What I still need to review:

______________________________________________________________________________

______________________________________________________________________________

\newpage

# 20. One-Page Ultra Condensed Review

1. **Causal inference asks counterfactual questions.** What would have happened to the same unit without treatment?
2. **Selection bias is the enemy.** Treated and control groups may differ before treatment.
3. **Randomization solves selection bias in expectation.** Check balance but remember balance tests are not the design itself.
4. **Regression and matching adjust for observed confounders.** They rely on no unobserved confounding.
5. **OVB formula:** bias = effect of omitted variable on outcome $\times$ relationship between treatment and omitted variable.
6. **IV uses external variation in treatment.** Valid IV requires relevance, independence, exclusion, and monotonicity.
7. **2SLS:** first predict treatment using instrument, then regress outcome on predicted treatment.
8. **RD compares units near a cutoff.** The coefficient on treatment estimates the jump at the threshold.
9. **DiD compares changes.** DiD = treated change - control change.
10. **Parallel trends is the key DiD assumption.** Equal levels are not required; similar pre-treatment trends are.

Final self-test:

- The treatment is: ___________________________________________________________
- The outcome is: _____________________________________________________________
- The method is: ______________________________________________________________
- The coefficient of interest is: _____________________________________________
- The identifying assumption is: ______________________________________________
- The biggest threat is: ______________________________________________________
