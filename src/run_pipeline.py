"""Run the entire training and prediction pipeline.

This script executes preprocessing, model training, threshold optimization,
 and final prediction to generate the submission file `data/output/submit.csv`.
"""

from preprocess import main as preprocess_main
from train import main as train_main
from threshold_opt import main as threshold_main
from ensemble import main as ensemble_main
from predict import main as predict_main


def main() -> None:
    """Execute all pipeline steps sequentially."""
    preprocess_main()
    train_main()
    threshold_main()
    # optional: ensemble weight/threshold search (safe to run)
    try:
        ensemble_main()
    except Exception:
        pass
    predict_main()


if __name__ == "__main__":
    main()
