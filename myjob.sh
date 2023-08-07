source ./ultra/bin/activate;
echo "activate python virtual env.";
python --version;
cd inference-api;
uvicorn main:app --host 0.0.0.0 --port 8000;
echo "run uvicorn on remote host.";
