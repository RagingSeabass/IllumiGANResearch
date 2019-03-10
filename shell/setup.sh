module load python3/3.6.2

# Setup virtual env
export PYTHONPATH=
python3 -m venv mlpy3env
source mlpy3env/bin/activate

# Upgrade pip
pip install -U pip

# install 
pip install -r requirements.txt

# stop virtual env
deactivate
chmod -R 777 mlpy3env


