# 🛡️ NetSentinel
### Network Traffic Intelligence & Intrusion Detection

NetSentinel is a network traffic analysis platform that combines supervised machine learning, unsupervised anomaly detection, and live traffic monitoring in a Streamlit dashboard.

## Features

- **Traffic Overview:** Explore historical network traffic and label distributions.
- **Supervised Detection:** Classify network flows using a trained Random Forest model.
- **Anomaly Detection:** Use Isolation Forest to identify unusual traffic patterns.
- **Live Monitoring:** Capture local IP traffic and aggregate packet statistics into flow windows.
- **Heuristic Alerts:** Flag possible traffic bursts, high byte rates, and potential TCP SYN scans.
- **Exportable Reports:** Download prediction results, live flow records, and alert history.

## Architecture

```text
CIC-IDS2017 Dataset
        |
        +--> Random Forest Classifier
        |
        +--> Isolation Forest
                    |
                    v
            Streamlit Dashboard
                    ^
                    |
Live Network --> Scapy --> Flow Statistics
                             |
                             v
                       Heuristic Alerts
```

The historical ML models and live heuristic detector are separate pipelines. Live packet features are not passed directly into models trained on CIC-IDS2017 features.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Scapy
- Streamlit
- Plotly
- Joblib

## Dataset and Evaluation

The project uses the CIC-IDS2017 network intrusion detection dataset.

The tuned Isolation Forest model was evaluated using a day-based split:
- Training: Monday benign traffic
- Threshold calibration: Tuesday benign traffic
- Held-out testing: Wednesday traffic

### Held-out results

| Metric | Result |
|---|---:|
| Accuracy | 82% |
| Attack precision | 84% |
| Attack recall | 58% |
| Attack F1-score | 0.69 |
| Benign false-positive rate | 5.68% |

The test used sampled flows, including 30,000 benign and 15,000 attack flows. These results demonstrate performance on this experimental split, not guaranteed performance on real-world traffic.

## Installation

Clone the repository:

```bash
git clone https://github.com/AddisionS/NetSentinel.git
cd NetSentinel
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas numpy scikit-learn streamlit plotly joblib scapy streamlit-autorefresh
```

Download and extract CIC-IDS2017 into the `data/` directory. Train the models using the project's training scripts before launching the dashboard.

## Running NetSentinel

Start the dashboard:

```powershell
streamlit run app.py
```

To enable live traffic monitoring, open a second terminal and run:

```powershell
python live_engine.py
```

On Windows, live packet capture may require Npcap and appropriate capture permissions.

## Project Structure

```text
NetSentinel/
├── app.py
├── live_engine.py
├── live_dashboard.py
├── train_model.py
├── train_anomaly.py
├── tune_anomaly.py
├── evaluate_by_day.py
├── experiment_anomaly.py
├── inspect_data.py
├── models/       # Generated model files
├── data/         # Local dataset files
├── live_flows.jsonl
└── alerts.jsonl
```

Generated models, datasets, and local logs are kept out of version control where configured by `.gitignore`.

## Limitations

- Live detection currently uses heuristics rather than the trained CIC-IDS2017 models.
- Heuristic alerts may include false positives and are not proof of malicious activity.
- Live capture sees only traffic visible to the selected capture interface.
- Experimental results may differ on other datasets, networks, and attack types.

## Future Improvements

- Improve bidirectional flow tracking and flow expiration.
- Evaluate detection across additional unseen days and datasets.
- Tune alert thresholds using representative benign traffic.
- Add alert severity calibration and persistent monitoring.
- Deploy the dashboard for remote demonstrations.

## Disclaimer

Use live monitoring only on networks and devices you own or are authorized to monitor.
