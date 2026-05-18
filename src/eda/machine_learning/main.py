import yaml
import logging
from data_loader import load_data, split_features_target
from preprocessing import prepare_train_test_split, apply_smote
from train import train_best_rf
from evaluate import evaluate_performance, plot_visuals

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Data Loading
    df = load_data(config['data_path'])
    X, y = split_features_target(df, 'ChurnStatus')

    # 2. Preprocessing
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, config['test_size'], config['random_state']
    )
    X_train_res, y_train_res = apply_smote(X_train, y_train, config['random_state'])

    # 3. Training
    model = train_best_rf(X_train_res, y_train_res, config['rf_params'], config['random_state'])

    # 4. Evaluation
    y_prob = evaluate_performance(model, X_test, y_test)
    plot_visuals(y_test, y_prob, model, X_test)

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()