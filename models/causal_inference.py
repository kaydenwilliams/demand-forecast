import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_did_analysis():
    df = pd.read_csv("data/processed/model_output.csv", parse_dates=["date"])

    # post period — second half of dataset
    midpoint = df["date"].median()
    df["post"] = (df["date"] > midpoint).astype(int)

    # treated — months with promo flag
    df["treated"] = df["promo_flag"]

    # interaction term — the DiD estimator
    # this isolates the causal effect of promotion in the post period
    df["did"] = df["treated"] * df["post"]

    model = smf.ols("total_revenue ~ treated + post + did", data=df).fit()
    print(model.summary())

    # extract the key number — causal lift from promotions
    did_coef = model.params["did"]
    did_pval = model.pvalues["did"]

    print(f"\n--- Causal Inference Result ---")
    print(f"Promotional lift (DiD estimate): ${did_coef:,.2f}")
    print(f"P-value: {did_pval:.4f}")

    if did_pval < 0.05:
        print("Result: Statistically significant — promotions caused real demand lift")
    else:
        print("Result: Not statistically significant at 0.05 — cannot confirm causal lift")
        print("Note: Low sample size (22 rows) limits statistical power — expected in portfolio setting")

    # save results
    results = pd.DataFrame({
        "coefficient": model.params,
        "p_value": model.pvalues,
        "conf_int_low": model.conf_int()[0],
        "conf_int_high": model.conf_int()[1]
    })
    results.to_csv("data/processed/did_results.csv")
    print("\nDiD results saved.")
    return model

if __name__ == "__main__":
    run_did_analysis()