from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ai_data_analyst_agents.statistics.assumptions import AssumptionCheck, regression_design_checks
from ai_data_analyst_agents.statistics.confidence_intervals import regression_coefficient_cis
from ai_data_analyst_agents.statistics.effect_sizes import regression_fit_effects
from ai_data_analyst_agents.statistics.limitations import build_statistical_limitations
from ai_data_analyst_agents.statistics.models import RegressionRequest, StatisticalResult


def _standardized_coefficients(design: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    x = design.drop(columns=["const"], errors="ignore")
    if x.empty:
        return {}
    x_std = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    y_std = (target - target.mean()) / target.std(ddof=0) if target.std(ddof=0) else target * 0
    model = sm.OLS(y_std, sm.add_constant(x_std, has_constant="add")).fit()
    return {str(k): float(v) for k, v in model.params.items() if str(k) != "const"}


def run_ols_regression(df: pd.DataFrame, request: RegressionRequest, *, analysis_id: str) -> tuple[StatisticalResult, pd.DataFrame, dict[str, object]]:
    required = [request.target, *request.predictors]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Regression columns not found: {missing}")

    subset = df[required].copy().dropna()
    target = pd.to_numeric(subset[request.target], errors="coerce")
    predictors = pd.get_dummies(subset[request.predictors], drop_first=True)
    joined = pd.concat([target.rename(request.target), predictors], axis=1).dropna()
    if joined.empty:
        raise ValueError("No complete rows available for OLS regression after dropping missing values.")

    y = joined[request.target].astype(float)
    x = joined.drop(columns=[request.target]).astype(float)
    if x.empty:
        raise ValueError("OLS regression requires at least one usable predictor.")

    design = sm.add_constant(x, has_constant="add")
    checks = regression_design_checks(design, y)
    if any(check.name == "design_matrix_rank" and check.passed is False for check in checks):
        result = StatisticalResult(
            analysis_id=analysis_id,
            analysis_type="regression",
            method="ols",
            method_reason="Ordinary least squares requested, but the encoded design matrix is singular.",
            null_hypothesis="All slope coefficients are zero.",
            alternative_hypothesis="At least one slope coefficient is non-zero.",
            sample_sizes={"n_rows": int(joined.shape[0]), "n_predictors": int(x.shape[1])},
            alpha=request.alpha,
            test_statistic=None,
            p_value=None,
            decision="not_reliable",
            interpretation="OLS could not be fit reliably because the design matrix is rank-deficient.",
            plain_language="Regression is not reliable enough because the predictors are collinear or duplicated.",
            assumptions=checks,
            warnings=[c.detail for c in checks if c.passed is False],
            limitations=build_statistical_limitations(
                StatisticalResult(
                    analysis_id=analysis_id,
                    analysis_type="regression",
                    method="ols",
                    method_reason="Rank-deficient design.",
                    null_hypothesis=None,
                    alternative_hypothesis=None,
                    sample_sizes={},
                    alpha=request.alpha,
                    test_statistic=None,
                    p_value=None,
                    decision="not_reliable",
                    interpretation="",
                    plain_language="",
                    assumptions=checks,
                )
            ),
            status="not_reliable",
        )
        return result, pd.DataFrame(), {"encoded_columns": list(x.columns)}

    model = sm.OLS(y, design).fit()
    coeff_table = pd.DataFrame(
        {
            "term": [str(idx) for idx in model.params.index],
            "coefficient": [float(v) for v in model.params.values],
            "std_error": [float(v) for v in model.bse.values],
            "t_stat": [float(v) for v in model.tvalues.values],
            "p_value": [float(v) for v in model.pvalues.values],
        }
    )
    conf_df = model.conf_int(alpha=request.alpha)
    conf_df.columns = ["lower_bound", "upper_bound"]
    coeff_table = coeff_table.merge(conf_df.reset_index().rename(columns={"index": "term"}), on="term", how="left")

    residuals = pd.Series(model.resid, name="residual")
    fitted = pd.Series(model.fittedvalues, name="fitted")
    jb_res = stats.jarque_bera(residuals)
    jb_stat = float(getattr(jb_res, "statistic", jb_res[0]))
    jb_p = float(getattr(jb_res, "pvalue", jb_res[1]))
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, design)
    outlier_count = int((residuals.abs() > 3 * residuals.std(ddof=0)).sum()) if residuals.std(ddof=0) else 0

    checks.extend(
        [
            AssumptionCheck(
                name="residual_normality",
                passed=bool(jb_p >= request.alpha),
                detail=f"Jarque-Bera p-value for residuals={jb_p:.4f} (statistic={jb_stat:.4f}).",
                severity="warn" if jb_p < request.alpha else "info",
            ),
            AssumptionCheck(
                name="heteroskedasticity",
                passed=bool(bp_p >= request.alpha),
                detail=f"Breusch-Pagan p-value={bp_p:.4f} (statistic={bp_stat:.4f}).",
                severity="warn" if bp_p < request.alpha else "info",
            ),
            AssumptionCheck(
                name="residual_outliers",
                passed=outlier_count == 0,
                detail=f"Residuals beyond 3 standard deviations: {outlier_count}.",
                severity="warn" if outlier_count > 0 else "info",
            ),
        ]
    )

    vif_map: dict[str, float] = {}
    if x.shape[1] > 1:
        vif_design = design.drop(columns=["const"], errors="ignore")
        for idx, col in enumerate(vif_design.columns):
            vif_map[str(col)] = float(variance_inflation_factor(vif_design.to_numpy(), idx))
        high_vif = {k: v for k, v in vif_map.items() if v >= 10.0}
        checks.append(
            AssumptionCheck(
                name="multicollinearity",
                passed=not high_vif,
                detail=f"VIF values: {vif_map}.",
                severity="warn" if high_vif else "info",
            )
        )

    standardized = _standardized_coefficients(design, y) if request.include_standardized else {}
    coeff_table["standardized_beta"] = coeff_table["term"].map(lambda term: standardized.get(str(term)))

    significant_predictors = coeff_table[
        (coeff_table["term"] != "const") & (coeff_table["p_value"] < request.alpha)
    ]
    decision = "reject_null" if not significant_predictors.empty else "fail_to_reject_null"
    interpretation = (
        f"OLS fitted {request.target} on {len(request.predictors)} predictor(s). "
        f"R-squared={float(model.rsquared):.4f}, adjusted R-squared={float(model.rsquared_adj):.4f}, "
        f"F-statistic p-value={float(model.f_pvalue):.4f}."
    )
    plain_language = (
        "At least one predictor shows a statistically detectable association with the target."
        if decision == "reject_null"
        else "The model does not show statistically detectable slope coefficients at the chosen alpha."
    )
    result = StatisticalResult(
        analysis_id=analysis_id,
        analysis_type="regression",
        method="ols",
        method_reason="OLS provides an interpretable linear association model for a numeric target.",
        null_hypothesis="All slope coefficients are zero.",
        alternative_hypothesis="At least one slope coefficient is non-zero.",
        sample_sizes={"n_rows": int(joined.shape[0]), "n_predictors": int(x.shape[1])},
        alpha=request.alpha,
        test_statistic=float(model.fvalue) if model.fvalue is not None else None,
        p_value=float(model.f_pvalue) if model.f_pvalue is not None else None,
        decision=decision,
        interpretation=interpretation,
        plain_language=plain_language + " Regression coefficients reflect association, not necessarily causation.",
        assumptions=checks,
        warnings=[check.detail for check in checks if check.passed is False],
        confidence_intervals=regression_coefficient_cis(model.conf_int(alpha=request.alpha), confidence_level=1.0 - request.alpha),
        effect_sizes=regression_fit_effects(float(model.rsquared), float(model.rsquared_adj)),
        limitations=[],
        metrics={
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
            "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else None,
        },
        extra_outputs={
            "coefficients": coeff_table.to_dict(orient="records"),
            "diagnostics": {
                "jarque_bera_pvalue": float(jb_p),
                "breusch_pagan_pvalue": float(bp_p),
                "outlier_count": outlier_count,
                "vif": vif_map,
                "encoded_columns": list(x.columns),
            },
        },
    )
    result.limitations = build_statistical_limitations(result)
    diagnostics = {
        "jarque_bera_pvalue": float(jb_p),
        "breusch_pagan_pvalue": float(bp_p),
        "outlier_count": outlier_count,
        "vif": vif_map,
        "encoded_columns": list(x.columns),
        "fitted_preview": [float(v) for v in fitted.head(10)],
        "residual_preview": [float(v) for v in residuals.head(10)],
    }
    return result, coeff_table, diagnostics
