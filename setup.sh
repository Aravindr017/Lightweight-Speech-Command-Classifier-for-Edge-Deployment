# only for setting up the environment and running the full pipeline end-to-end (training, exporting, evaluating)
set -e      # for exiting on any error

echo "Creating virtual environment"
python3.11 -m venv venv
source venv/bin/activate

echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "Generating dataset"
python data/generate_dataset.py

echo "Training prototypes and threshold"
python src/train.py

echo "Exporting model to ONNX and quantizing to INT8"
python src/export_onnx.py

echo "Running evaluation with quantized model (int8)"
python src/evaluate.py

echo "Setup complete. Run 'python run_demo.py' to test the classifier with custom commands."