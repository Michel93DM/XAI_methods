import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np

def plot_ale_interactive(model, X_test, features=None, grid_size=50, include_CI=True, C=0.95, figsize=(8, 5)):
    """
    Displays an interactive widget to select a feature
    and visualize its ALE (Accumulated Local Effects) plot.

    Parameters
    ----------
    model : scikit-learn estimator or pipeline
        The trained model or pipeline to evaluate.

    X_test : pandas.DataFrame
        The dataset on which the ALE is computed.

    features : list of str, optional
        List of features to include in the dropdown. If None, all columns in X_test are used.

    grid_size : int, default=50
        Number of bins to divide the feature range into for computing ALE.

    include_CI : bool, default=True
        Whether to include confidence intervals in the ALE plot.

    C : float, default=0.95
        Confidence level for the interval if `include_CI=True`.

    figsize : tuple, default=(8, 5)
        Size of the matplotlib figure.
    """
    from PyALE import ale

    if features is None:
        features = list(X_test.columns)
    
    dropdown = widgets.Dropdown(
        options=features,
        description='Feature:',
        layout=widgets.Layout(width='300px')
    )

    output = widgets.Output()

    def update(feature_name):
        with output:
            output.clear_output(wait=True)
            print(f"ALE for: {feature_name}")
            ale_result = ale(
                X=X_test,
                model=model,
                feature=[feature_name],
                grid_size=grid_size,
                plot=True,
                include_CI=include_CI,
                C=C
            )
            plt.title(f"ALE Plot for {feature_name}", fontsize=16)
            plt.tight_layout()
            plt.show()

    widgets.interact(update, feature_name=dropdown)
    display(output)


def plot_alibi_ale_interactive(model, X_test, features=None, figsize=(8,5)):
    """
    Displays an interactive widget to select a feature
    and visualize its ALE (Accumulated Local Effects) plot using Alibi.

    Parameters
    ----------
    model : scikit-learn estimator
        The trained model to evaluate.
    X_test : pandas.DataFrame or array-like
        Test dataset (will be transformed by pipeline before explanation).
    features : list of str, optional
        List of feature names to include in the dropdown. If None, all columns of X_test are used.
    bins : int, default=50
        Number of bins/grid points for computing ALE.
    figsize : tuple, default=(8, 5)
        Size of the matplotlib figure.
    """
    from alibi.explainers import ALE as AlibiALE

    if features is None and type(X_test) == np.ndarray:
        raise Exception("If X_test is not a dataframe, features cannot be None")
    elif features is None:
        features = list(X_test.columns)

    # Initialize Alibi ALE explainer
    ale_explainer = AlibiALE(
        predictor=model.predict,
        feature_names=features
    )
    ale_exp = ale_explainer.explain(X_test)

    dropdown = widgets.Dropdown(
        options=features,
        description='Feature:',
        layout=widgets.Layout(width='300px')
    )
    output = widgets.Output()

    def update(feature_name):
        with output:
            output.clear_output(wait=True)
            f_idx = list(ale_explainer.feature_names).index(feature_name)
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(ale_exp.ale_values[f_idx], label=feature_name)
            ax.set_xlabel(f"{feature_name} bins")
            ax.set_ylabel("ALE value")
            ax.set_title(f"ALE Plot - {feature_name}")
            ax.grid(True)
            plt.tight_layout()
            plt.show()

    widgets.interact(update, feature_name=dropdown)
    display(output)

