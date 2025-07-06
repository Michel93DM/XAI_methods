import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import ipywidgets as widgets
from IPython.display import display, HTML

def interactive_lime_explanation(model, X: pd.DataFrame, y: pd.Series = None, mode: str = 'classification'):
    """
    Displays an interactive LIME explanation for individual predictions using a slider.

    Parameters
    ----------
    model : object
        A trained model (e.g., scikit-learn estimator).
        For classification, must implement `predict_proba`.
        For regression, must implement `predict`.

    X : pandas.DataFrame
        The input data used to compute LIME explanations. Each row corresponds to one instance.

    y : pandas.Series or None, optional
        The target variable. Required for computing class names in classification mode.

    mode : str, optional
        Either 'classification' or 'regression'. Default is 'classification'.

    Returns
    -------
    None
        Displays an interactive slider and LIME explanations in a Jupyter Notebook.
    """
    # Set up LIME explainer
    class_names = list(map(str, np.unique(y))) if mode == 'classification' and y is not None else None
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=class_names,
        mode=mode,
        verbose=False,
        discretize_continuous=True
    )

    def explain_instance(index):
        instance = X.iloc[index].values

        if mode == 'classification':
            explanation = explainer.explain_instance(instance, model.predict_proba, num_features=len(X.columns))
        else:
            explanation = explainer.explain_instance(instance, model.predict, num_features=len(X.columns))

        html = explanation.as_html()
        display(HTML(html))

    index_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(X) - 1,
        step=1,
        description='Index:',
        continuous_update=False
    )

    widgets.interact(explain_instance, index=index_slider)

    return