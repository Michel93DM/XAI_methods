import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import ipywidgets as widgets
from IPython.display import display

def plot_partial_dependence_interactive(pipeline, X_test, features=None, kind="average", grid_resolution=100, figsize=(8, 5)):
    """
    Displays an interactive widget to select a feature
    and visualize its Partial Dependence Plot (PDP).

    Parameters
    ----------
    pipeline : scikit-learn estimator or pipeline
        The trained model or pipeline.

    X_test : pandas.DataFrame or array-like
        The test data used to compute the PDP.

    features : list of str or list of int, optional
        List of features to include in the selection. If None, all columns from X_test will be used.

    kind : str, default="average"
        Type of PDP to display: "average" for the mean PDP, or "individual" for instance-level curves.

    grid_resolution : int, default=100
        Number of points to use for the grid along the feature axis.

    figsize : tuple, default=(8, 5)
        Size of the output plot.
    """

    if features is None:
        features = list(X_test.columns)
    else:
        # s'assurer que les features sont nommées
        # si int, on convertit en nom de colonne
        feats = []
        for f in features:
            if isinstance(f, int):
                feats.append(X_test.columns[f])
            else:
                feats.append(f)
        features = feats

    dropdown = widgets.Dropdown(
        options=features,
        description='Feature:',
        layout=widgets.Layout(width='300px')
    )

    output = widgets.Output()

    def update(feature_name):
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            PartialDependenceDisplay.from_estimator(
                pipeline,
                X_test,
                [feature_name],  # une seule feature
                kind=kind,
                grid_resolution=grid_resolution,
                ax=ax
            )
            ax.set_title(f"PDP for feature: {feature_name}", fontsize=16)
            plt.tight_layout()
            plt.show()

    widgets.interact(update, feature_name=dropdown)

    display(output)