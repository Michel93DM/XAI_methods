import shap
import pandas as pd

import ipywidgets as widgets
from IPython.display import display, HTML


def interactive_force_plot(model, X: pd.DataFrame):
    """
    Displays an interactive SHAP force plot for individual predictions using a slider.

    This function creates an interactive widget that allows the user to select an observation
    (by index) from the input dataset `X` and view its corresponding SHAP force plot.
    The plot illustrates how each feature contributes to the prediction for that specific instance.

    Parameters
    ----------
    model : object
        A trained model (e.g., a scikit-learn estimator, XGBoost, LightGBM, etc.)
        compatible with SHAP's `Explainer`.

    X : pandas.DataFrame
        The input data used to compute SHAP values. Each row corresponds to one instance
        for which a local explanation can be generated.

    Returns
    -------
    None
        Displays an interactive slider and SHAP force plots in a Jupyter Notebook.
        The function does not return any value.

    Notes
    -----
    - The function uses `shap.Explainer` to compute SHAP values and `shap.force_plot` 
      to visualize the local explanations.
    - The force plot is rendered in JavaScript (`matplotlib=False`) for interactivity.
    - The `IPython.display` and `ipywidgets` libraries are used to create the slider UI.
    - Works only in Jupyter environments that support HTML/JS output (e.g., Jupyter Notebook or JupyterLab).

    Example
    -------
    >>> interactive_force_plot(trained_model, X_test)
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.initjs()

    def show_force_plot(index):
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[index].values,
            features=X.iloc[index],
            feature_names=X.columns,
            matplotlib=False  # Use JavaScript-based rendering
        )
        
        html = f"""
        <div style='background-color:white;padding:10px'>
            {force_plot.html()}
        </div>
        """
        display(HTML(html))

    index_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(X) - 1,
        step=1,
        description='Index:',
        continuous_update=False
    )

    widgets.interact(show_force_plot, index=index_slider)

    return
