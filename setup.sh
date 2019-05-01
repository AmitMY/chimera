# Install dependencies
pip install -r requirements.txt

# Download NLTK punkt
python -c "import nltk; nltk.download('punkt')"


# Install OpenNMT
mkdir libs
cd libs
git clone --branch 0.8.2 https://github.com/OpenNMT/OpenNMT-py.git OpenNMT
cd OpenNMT
#python setup.py install
pip install -r requirements.txt
cd ..

## Install E2E-Metrics
#git clone https://github.com/tuetschek/e2e-metrics.git
#cd e2e-metrics
#pip install -r requirements.txt
#
#curl -L https://cpanmin.us | perl - App::cpanminus  # install cpanm
#cpanm XML::Twig

# Install NLG-Eval
git clone https://github.com/Maluuba/nlg-eval.git
cd nlg-eval
pip install -e .
nlg-eval --setup
# python ./measure_scores.py ../../cache/E2E/translate/references.txt  ../../cache/E2E/translate/hypothesis.txt
cd..

# Clone NeuralREG
git clone https://github.com/ThiagoCF05/NeuralREG.git
cd NeuralREG

cd..
