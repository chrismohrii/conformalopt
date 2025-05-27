if __name__ == "__main__":

    import conformalopt as oc

    # Get an example dataset.
    scores = oc.data.get_scores("GOOGL_theta_absolute-residual_scores")
    split = int(0.33 * len(scores))
    val_scores, test_scores = scores[:split], scores[split:]

    # Instantiate a ConformalPredictor object for linear quantile tracking.
    cp = oc.ConformalPredictor(quantile_tracker="scalar", alpha=0.1)

    # Fit the ConformalPredictor's hyperparameters (such as learning rate).
    cp.fit(val_scores=val_scores)

    # Prediction loop.
    for t in range(len(test_scores)):
        prediction = cp.predict()  # Make a prediction for the score.
        cp.step(prediction, test_scores[t])  # Update the quantile tracker parameter.

    # Provides an evaluation in out/expt_name folder.
    cp.eval()
