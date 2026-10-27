# Smart Expense Leak Detector

A privacy-focused, local expense analysis application built with Python, Streamlit, pandas, and scikit-learn.

Smart Expense Leak Detector helps users understand their spending habits, identify potentially unusual transactions, discover savings opportunities, and estimate future expenses. It also includes a locally running AI assistant powered by Ollama.

The application is designed to run on your own computer, with expense analysis and AI assistance available without relying on a cloud AI service.

## Features

- **CSV Upload and Validation** — Import expense records and validate required columns.
- **Expense Analysis** — Calculate total expenses and analyze transactions.
- **Category Analysis** — Understand spending distribution across categories.
- **Monthly Summaries** — Review changes in spending over time.
- **Rule-Based Anomaly Detection** — Flag potentially unusual transactions using statistical rules.
- **Machine Learning Anomaly Detection** — Use scikit-learn's Isolation Forest alongside a robust statistical check to identify potential outliers.
- **Savings Suggestions** — Explore illustrative suggestions based on spending patterns.
- **Expense Forecasting** — Estimate next month's expenses using historical averages.
- **Local AI Assistant** — Generate explanations of expense patterns using a local Ollama model.
- **Automated Tests** — Run the project's test suite with pytest.

## Technology Stack

- Python
- Streamlit
- pandas
- scikit-learn
- requests
- pytest
- Ollama
- `qwen2.5:0.5b-instruct`

## Requirements

- A compatible Python installation
- The Python dependencies listed in `requirements.txt`
- Ollama for local AI functionality
- The `qwen2.5:0.5b-instruct` model
- Sufficient disk space and memory for the application and model

The application is designed for modest hardware, but actual performance depends on available RAM, CPU, and other running applications.

## Installation and Setup

### 1. Clone the Repository

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd smart_expense_leak_detector
```

Alternatively, download or copy the project files into a local folder.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use this command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Install and Configure Ollama

Install Ollama from its official website:

https://ollama.com/

Download the required model:

```bash
ollama pull qwen2.5:0.5b-instruct
```

Check that the model is available:

```bash
ollama list
```

Make sure the Ollama service is running before using the local AI assistant.

**Note:** The main expense analysis features may work without Ollama, but AI explanations require the local service and model.

## Run the Application

From the project directory, run:

```bash
python -m streamlit run app.py
```

Open the local address displayed in your terminal, usually:

http://localhost:8501

Upload a CSV file containing these required columns:

| Column | Description |
|---|---|
| `Date` | Date of the transaction |
| `Description` | Transaction description |
| `Category` | Expense category |
| `Amount` | Transaction amount |

Use fictional or non-sensitive sample records while testing.

## Run the Tests

Run the complete test suite:

```bash
python -m pytest -v
```

The test suite has previously passed **26 tests** in the development environment.

Passing automated tests helps verify application behavior but does not guarantee that every anomaly detection result, AI explanation, or forecast will be accurate.

## Privacy and Offline Use

Smart Expense Leak Detector is designed for local use.

- Expense analysis runs on your computer.
- The AI assistant uses a locally running Ollama model.
- Once the application dependencies and model are installed, the application can operate without an internet connection, provided the local services are available.
- Initial installation and model downloads may require internet access.

Do not commit real financial records, credentials, API keys, or other sensitive information to a public repository.

## Limitations

- An anomaly is a potential outlier, not proof of fraud or unnecessary spending.
- Machine learning results depend on the available data and detection thresholds.
- Savings suggestions are illustrative and may not suit every user's circumstances.
- Forecasts based on historical averages cannot guarantee future spending.
- Local AI responses may be incomplete or incorrect.
- Performance depends on available memory, CPU, and model size.

## License

No license has been selected yet. Add an appropriate open-source license before publishing if you want to specify how others may use, modify, or redistribute this project.
