# Life Expectancy Predictor

A Streamlit web application that predicts life expectancy based on various health, economic, and demographic factors.

## Features

- Interactive web interface for inputting country data
- Machine learning model for life expectancy prediction
- Supports data for multiple countries and years

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv env
   ```

2. Activate the virtual environment:

   - On macOS/Linux: `source env/bin/activate`
   - On Windows: `env\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

To run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser.

## Files

- `app.py`: Main Streamlit application
- `life_expectancy_full_pipeline.pkl`: Trained machine learning model
- `requirements.txt`: Python dependencies
- `env/`: Virtual environment (created during setup)
