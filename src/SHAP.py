def run_shap_explainability(model, obs_test, shap_background_size=20, shap_sample_size=20):
    """
    Explains the final PPO model by predicting the cash weight.
    The explanation answers:
      Which features push the policy towards holding more cash?
    """
    if not SHAP_AVAILABLE:
        print("SHAP is not installed. Skipping SHAP explainability.")
        return None, None

    if len(obs_test) < 3:
        print("Not enough test rows for SHAP. Skipping.")
        return None, None

    X = obs_test.copy().reset_index(drop=True).astype(float)

    def predict_cash(X_input):
        X_input = np.asarray(X_input, dtype=np.float32)
        preds = []
        for row in X_input:
            action, _ = model.predict(row, deterministic=True)
            weights = safe_softmax(action)
            preds.append(weights[-1])  # last weight is cash
        return np.array(preds)

    bg_n = min(shap_background_size, len(X))
    sample_n = min(shap_sample_size, len(X))

    try:
      background = shap.utils.sample(X, bg_n, random_state=SEED)
    except Exception:
      background = shap.sample(X, bg_n, random_state=SEED)
    sample_rows = X.iloc[:sample_n]

    explainer = shap.Explainer(predict_cash, background, algorithm="permutation")
    shap_values = explainer(sample_rows)

    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values.values, sample_rows, feature_names=X.columns, show=False)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path1}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, sample_rows, feature_names=X.columns, plot_type="bar", show=False)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "shap_bar.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path2}")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("MeanAbsSHAP", ascending=False)

    imp_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    importance.to_csv(imp_path, index=False)
    print(f"Saved: {imp_path}")
    print("\nTop SHAP features:")
    print(importance.head(10))

    return shap_values, importance
