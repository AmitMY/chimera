# Install dependencies
pip install -r requirements.txt

# Install OpenNMT
mkdir libs
cd libs
git clone --branch 0.7.0 https://github.com/OpenNMT/OpenNMT-py.git OpenNMT
cd OpenNMT
#python setup.py install
pip install -r requirements.txt
cd ../..