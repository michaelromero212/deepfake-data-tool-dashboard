# 🔬 Deepfake Data Forge

**Deepfake Data Forge** is an MLOps-style dataset preparation pipeline and visualization dashboard designed specifically for deepfake detection models. It automates the discovery, preprocessing, metadata extraction, validation, and scoring of multimedia (images, audio, and video) to build high-quality, reliable datasets.

---

## 🎯 Project Purpose

Training robust deepfake detection models requires massive amounts of well-curated data consisting of both *real* and *synthetic* media. Creating these datasets manually is tedious, prone to errors, and difficult to scale. 

**Deepfake Data Forge** solves this by providing:
1. **Automated Pipeline Orchestration:** A standardized, modular pipeline that digests raw media, normalizes it, and extracts ground-truth metadata.
2. **Inference & Scoring:** Integration with HuggingFace models to run baseline detection inference (scoring how "fake" a sample looks/sounds) during dataset creation.
3. **Data Quality Validation:** Automated checks to ensure all media meets strict quality standards and formatting before it ever reaches a training run.
4. **Interactive Explorer:** A rich Streamlit dashboard to visually explore dataset distributions, validation reports, and baseline model scores.

---

## 📸 Dashboard Overview & Screenshots

The included Streamlit dashboard provides deep insights into the dataset generation and validation process.

### 1. High-Level Metrics
Get an instant overview of the total samples, media types (audio, video, images), and label distribution (real vs. synthetic) in the current dataset iteration.
![Dashboard Overview - High Level Metrics](screenshots/Screenshot\ 2026-03-08\ at\ 9.58.26\ AM.jpg)

### 2. Detection Score Distribution
Visualizes how well the baseline models distinguish between real and synthetic media across the dataset. The histogram shows the distribution of scores, with a decision boundary indicating the split.
![Detection Score Distribution](screenshots/Screenshot\ 2026-03-08\ at\ 9.58.39\ AM.jpg)

### 3. Validation Report & Media Types
A breakdown of the data validation status. The donut chart instantly shows the pass/warn/fail rate for pipeline processing, while the bar chart displays the distribution of data across different media formats.
![Validation & Media Types](screenshots/Screenshot\ 2026-03-08\ at\ 9.58.48\ AM.jpg)

### 4. Interactive Sample Browser
A detailed, filterable data table containing every sample's metadata, detection scores, file sizes, validation status, and inference times. Filter and sort directly in the browser to isolate specific edge cases.
![Interactive Sample Data Table](screenshots/Screenshot\ 2026-03-08\ at\ 9.58.58\ AM.jpg)

---

## 🛠 Tech Stack

- **Core Application:** Python 3.10+
- **Data Engineering / Processing:** Polars, NumPy, OpenCV, Pillow, Librosa, Soundfile
- **ML / Inference:** PyTorch, Transformers (HuggingFace)
- **Visualization:** Streamlit, Matplotlib
- **Orchestration & Tooling:** Click (CLI), Rich (Terminal UI), Pydantic (Schema Validation)
- **Infrastructure:** Moto/Boto3 (AWS S3 Integration)
- **Dependency Management:** Pip / Hatchling

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone & Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-data-tool-dashboard.git
cd deepfake-data-tool-dashboard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Configure Secrets
Create a `.env` file in the root directory and add your HuggingFace API token:
```bash
HUGGINGFACE_TOKEN=your_token_here
```
*(Note: `.env` is ignored by Git for security).*

### 3. Install Dependencies
```bash
# Install the package and its requirements in editable mode
pip install -e "."
```

### 4. Generate Sample Data
To test the pipeline without an external dataset, generate synthetic test files:
```bash
python scripts/generate_sample_data.py
```

### 5. Run the Pipeline
Execute the full MLOps pipeline to process `data/raw/` and output manifest/validation reports into `outputs/`:
```bash
python -m src.pipeline run
```

### 6. Launch the Dashboard
Once the manifest is generated, explore the data in the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

---

## 🗂 Project Structure
```text
deepfake-data-tool-dashboard/
├── data/
│   ├── raw/             # Raw input media (images, audio, video)
│   └── processed/       # Pipeline output (normalized, frames extracted)
├── outputs/             # Generated dataset_manifest.json and validation_report.json
├── screenshots/         # Dashboard screenshots for documentation
├── scripts/
│   └── generate_sample_data.py
├── src/
│   ├── ingestion.py     # Discovers media and derives labels
│   ├── metadata.py      # Computes hashes and structural metadata
│   ├── preprocessing.py # Normalizes data (e.g., video frame extraction)
│   ├── detection.py     # Runs deepfake detection inference (HuggingFace)
│   ├── validation.py    # Ensures schemas and quality standards match
│   ├── schemas.py       # Pydantic models for the pipeline
│   ├── storage.py       # S3 upload functionality
│   └── pipeline.py      # Main CLI orchestrator
├── tests/               # Unit tests (pytest)
├── dashboard.py         # Streamlit web app
└── pyproject.toml       # Python dependencies and build config
```
