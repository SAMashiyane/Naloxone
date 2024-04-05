

import os
from typing import Optional
from naloxlib.inward.logging import get_logger


def show_yellowbrick_plot(
    visualizer,
    X_train,
    y_train,
    X_test,
    y_test,
    name: str,
    handle_train: str = "fit",
    handle_test: str = "score",
    scale: float = 1,
    save: bool = False,
    fit_kwargs: Optional[dict] = None,
    display_format: Optional[str] = None,
    **kwargs,
):

    logger = get_logger()
    visualizer.fig.set_dpi(visualizer.fig.dpi * scale)

    if not fit_kwargs:
        fit_kwargs = {}

    fit_kwargs_and_kwargs = {**fit_kwargs, **kwargs}

    if handle_train == "draw":
        logger.info("Drawing Model")
        visualizer.draw(X_train, y_train, **kwargs)
    elif handle_train == "fit":
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train, **fit_kwargs_and_kwargs)
    elif handle_train == "fit_transform":
        logger.info("Fitting & Transforming Model")
        visualizer.fit_transform(X_train, y_train, **fit_kwargs_and_kwargs)
    elif handle_train == "score":
        logger.info("Scoring train set")
        visualizer.score(X_train, y_train, **kwargs)

    if handle_test == "draw":
        visualizer.draw(X_test, y_test)
    elif handle_test == "fit":
        visualizer.fit(X_test, y_test, **fit_kwargs)
    elif handle_test == "fit_transform":
        visualizer.fit_transform(X_test, y_test, **fit_kwargs)
    elif handle_test == "score":
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)

    plot_filename = f"{name}.png"
    if save:
        if not isinstance(save, bool):
            plot_filename = os.path.join(save, plot_filename)
        logger.info(f"Saving '{plot_filename}'")
        visualizer.show(outpath=plot_filename, clear_figure=True, bbox_inches="tight")
    else:

        visualizer.show(clear_figure=True)

    logger.info("Visual Rendered Successfully")
    return plot_filename
