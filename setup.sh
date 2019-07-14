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