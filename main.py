from data_preprocessing import load_and_clean_data
from feature_extraction import vectorize_text
from model import train_model, evaluate_model

def main():
    filepath = 'spam.csv'  # Update this path to your dataset's location
    data = load_and_clean_data(filepath)
    X, y = vectorize_text(data)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
