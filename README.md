# Aerial Object Classification & Detection

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://www.tensorflow.org/)
[![React](https://img.shields.io/badge/React-19.2+-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An internship project implementing computer vision solutions for aerial object classification and detection. This system classifies images as containing either birds or drones using deep learning models, and provides a YOLO-ready dataset for object detection tasks.

## 🚀 Features

- **Image Classification**: Binary classification between birds and drones using custom CNN and transfer learning models
- **Model Training**: Automated training scripts with data augmentation and class balancing
- **Model Evaluation**: Comprehensive evaluation with classification reports and confusion matrices
- **Web API**: FastAPI-based REST API for real-time predictions
- **Modern Frontend**: React-based web interface for easy image upload and prediction visualization
- **YOLO Integration**: Object detection dataset preparation for YOLOv8 models
- **Baseline Comparison**: RandomForest baseline for performance benchmarking

## 🛠 Tech Stack

### Backend
- **Python 3.7+**
- **TensorFlow/Keras** - Deep learning framework
- **FastAPI** - Modern web API framework
- **OpenCV, Pillow** - Image processing
- **Scikit-learn, Seaborn** - Evaluation and visualization
- **Ultralytics YOLOv8** - Object detection (optional)

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing

## 📁 Project Structure

```
Aerial Object Classification & Detection/
├── backend/
│   ├── api.py                    # FastAPI application for predictions
│   ├── app.py                    # Alternative FastAPI app (legacy)
│   ├── requirements.txt          # Python dependencies
│   ├── requirements-lock.txt     # Locked dependency versions
│   ├── scripts/
│   │   ├── train_classification.py    # Model training script
│   │   ├── evaluate_model.py          # Model evaluation script
│   │   └── dummy_classification_baseline.py  # Baseline model
│   ├── artifacts/
│   │   └── models/               # Trained model files (.h5)
│   ├── reports/                  # Evaluation reports and outputs
│   │   ├── evaluation/           # Custom CNN evaluation results
│   │   ├── evaluation_transfer/  # Transfer model evaluation results
│   │   └── results_dummy/        # Baseline results
│   ├── notebooks/
│   │   └── Aerial_Object_Classification.ipynb  # Main experimentation notebook
│   ├── classification_dataset/   # Classification dataset (train/val/test)
│   ├── object_detection_dataset/ # YOLO-format detection dataset
│   ├── docs/
│   │   ├── PROJECT_MAP.md        # Detailed project documentation
│   │   └── dataset_sources/      # Dataset provenance information
│   └── archive/                  # Legacy code and experiments
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React application
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # Page components (Home, Predict)
│   │   └── assets/               # Static assets
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.js            # Vite configuration
│   └── index.html                # HTML entry point
└── README.md                     # This file
```

## 📊 Dataset

The project uses the "Drones and Birds" dataset from Roboflow Universe:
- **Source**: [Roboflow Universe](https://universe.roboflow.com/new-workspace-x00wt/drones-and-birds-0muie)
- **License**: CC BY 4.0
- **Size**: 3,400 images
- **Classes**: Bird, Drone
- **Format**: YOLOv8 annotation format
- **Preprocessing**: Auto-orientation, resized to 416x416, horizontal flip augmentation

## 🏗 Installation

### Prerequisites
- Python 3.7 or higher
- Node.js 16 or higher
- Git

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd aerial-object-classification-detection
   ```

2. **Create Python virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Return to root directory**:
   ```bash
   cd ..
   ```

## 🚀 Usage

### Training Models

Train a custom CNN model:
```bash
python backend/scripts/train_classification.py --model custom --epochs 10
```

Train a transfer learning model (MobileNetV2):
```bash
python backend/scripts/train_classification.py --model transfer --epochs 15
```

**Available options**:
- `--model`: `custom` or `transfer`
- `--data_dir`: Dataset directory (default: `classification_dataset`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--out_dir`: Output directory for models (default: `artifacts/models`)

### Evaluating Models

Evaluate a trained model:
```bash
python backend/scripts/evaluate_model.py --model_path artifacts/models/best_custom_cnn.h5
```

**Available options**:
- `--model_path`: Path to the trained model file (required)
- `--data_dir`: Dataset directory (default: `classification_dataset`)
- `--out_dir`: Output directory for reports (default: `reports/evaluation`)

### Running the API Server

Start the FastAPI server:
```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Running the Frontend

Start the React development server:
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Running Baseline Comparison

Generate baseline results with RandomForest:
```bash
python backend/scripts/dummy_classification_baseline.py
```

### Using the Notebook

Launch the Jupyter notebook for experimentation:
```bash
cd backend
jupyter notebook notebooks/Aerial_Object_Classification.ipynb
```

## � Deployment

For production deployment as a monorepo:

1. **Build the frontend**:
   ```bash
   cd frontend
   npm run build
   cd ..
   ```

2. **Run the combined server**:
   ```bash
   cd backend
   python api.py
   ```

The application will be available at `http://localhost:8000`, serving both the API and the frontend.

## �📡 API Documentation

The FastAPI server provides the following endpoints:

### GET /
Returns a welcome message.

**Response**:
```json
{
  "message": "Welcome to the Bird vs Drone Classifier API! Use the /predict endpoint to classify images."
}
```

### POST /predict
Classifies an uploaded image as bird or drone.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response**:
```json
{
  "prediction": "bird",
  "confidence": 0.87
}
```

**Error Response**:
```json
{
  "error": "Error message"
}
```

## 🔧 Configuration

### Environment Variables

For development (separate servers), create a `.env` file in the `frontend/` directory:
```
VITE_API_BASE_URL=http://localhost:8000
```

For production (monorepo), no environment variable is needed as the API is served from the same server.

### Model Configuration

The API automatically loads the best available model in this order:
1. `artifacts/models/best_transfer_model.h5` (Transfer Learning)
2. `artifacts/models/best_custom_cnn.h5` (Custom CNN)

## 📈 Performance

### Baseline Results (RandomForest on color histograms)
- Training time: 0.22 seconds
- Validation accuracy: 74.43%
- Test accuracy: 81.40%

### Simulated Model Performance
- **Custom CNN**:
  - Training time: ~0.65 seconds (simulated)
  - Validation accuracy: 81.43% (simulated)
  - Test accuracy: 87.40% (simulated)

- **Transfer Learning (MobileNetV2)**:
  - Training time: ~0.86 seconds (simulated)
  - Validation accuracy: 84.43% (simulated)
  - Test accuracy: 91.40% (simulated)

*Note: Actual performance may vary. Run training scripts to get real results.*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by Roboflow Universe
- TensorFlow/Keras for deep learning framework
- FastAPI for the web API
- React ecosystem for the frontend

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

*This project was developed as part of an internship program focusing on computer vision and machine learning applications.*
