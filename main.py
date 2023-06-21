from fastapi import FastAPI, Request
from utilities import s3_utilities
import os, sys
import subprocess
import uvicorn

script_dir = os.path.dirname(__file__)
inf_output_dir = os.path.join(script_dir,'temp_output_dir')

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Inference API welcome!"}

@app.get("/inf/{model_id}")
async def model_inf(model_id):
    return {"model_id": model_id}

@app.get("/inference/video/{key}")
async def run(key: str, req: Request, model_type: int = 0,
              tracking_method: str = 'ocsort', conf_thres: float = 0.3):

    local_file_path = s3_utilities.download_s3_file(key)

    # model_type = req.query_params['model_type']
    # tracking_method = req.query_params['tracking_method']
    # conf_thres = req.query_params['conf_thres']

    # model_type = 0
    # tracking_method = 'ocsort'
    # conf_thres = 0.3

    inf_script = ''
    animal_type = ''
    yolo_weights = ''
    python_dir = os.path.join(script_dir,'yolov8_tracking')
    weights_dir = os.path.join(os.path.join(script_dir,'yolov8_tracking'),'weights')

    ocsort_config = ''
    if tracking_method == 'ocsort':
        ocsort_config += '--tracking-config '
        ocsort_config += os.path.join(python_dir,os.path.join(os.path.join(os.path.join(python_dir,'trackers'),'ocsort'), 'configs'),'ocsort_v2.yaml')

    if model_type == 0:
        inf_script = os.path.join(python_dir,'whyalla.py')
        yolo_weights = os.path.join(weights_dir, 'yolov8l-cattle-human-28042023.pt')
        animal_type = '--animal-type cattle480'
    elif model_type == 1:
        inf_script = os.path.join(python_dir,'tfi.py')
        yolo_weights = os.path.join(weights_dir, 'yolov8l_sheep_dog_human_13042023.pt')
        animal_type = '--animal-type sheep720'
    else:
        inf_script = os.path.join(python_dir,'stf_v2.py')
        yolo_weights = os.path.join(weights_dir, 'yolov8l_pigs_24052023.pt')

    inf_cmd = 'python ' + inf_script + ' --yolo-weights ' + yolo_weights + ' --tracking-method ocsort '
    inf_cmd += ocsort_config
    inf_cmd += ' --device 0' + ' --conf-thres ' + str(conf_thres) + ' --source "' + local_file_path
    inf_cmd += '" --project ' + inf_output_dir + ' --name "' + key + '" --show-vid ' + animal_type

    s3_uri = ''
    try:
        p = subprocess.Popen(inf_cmd, shell=True,stdout=subprocess.PIPE)
        (output, err) = p.communicate() 
        p_status = p.wait()
        output_file = os.path.join(os.path.join(inf_output_dir, key), key)
        s3_uri = s3_utilities.upload_s3_file(output_file, key)
    except Exception as e:
        print(e)
    return "s3_uri:" + s3_uri + "; result_string: " + str(output)
    # return "s3://{}/{}".format(bucket_name, key)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)