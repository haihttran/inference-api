import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device, strip_optimizer
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker
from collections import Counter
# roi lyc
def draw_ROI(img, rect):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))
    return img

def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def xywh2xyxy(x):
    y = [0, 0, 0, 0]
    y[0] = x[0]  # top left x
    y[1] = x[1]  # top left y
    y[2] = x[0] + x[2] # bottom right x
    y[3] = x[1] + x[3] # bottom right y
    return y

def add_animal_to_indi_count(id , animal_indi_count, animal_counter):
    if id in animal_indi_count:
        pass
    else:
        animal_indi_count[id] = len(animal_counter)

def add_animal_to_indi_count_minuser(id, dict_animal_indi_count, animal_counter_minuser):
    if id in dict_animal_indi_count:
        pass
    else:
        dict_animal_indi_count[id] = len(animal_counter_minuser) * -1

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        animal_type='Sheep',
):
    animal_counter = Counter()
    animal_counter_minuser = Counter()
    dict_animal_xdist_initial = dict()
    dict_animal_xdist_current = dict()
    dict_animal_xdist_max = dict()
    dict_animal_xdist_min = dict()
    dict_animal_indi_count = dict()
    dict_initial_animal_xyxy = dict()
    dict_initial_persons_xyxy = dict()
    dict_initial_animal_twenty_in_xyxy = dict()
    id_counter = Counter()
    run_dir = 'U'  # 'U' = cattle running from Down to Up, 'D' = cattle running Up to Down
    # animal_counter = defaultdict(set)
    global_label_right = ""
    global_label_left = ""
    count_trigger_dist = 200
    
    if animal_type == "cattle480":
        rect = (250, 0, 450, 800)  # horizontal black bars(146,60,171,359)
    counting_rect = xywh2xyxy(rect)
    counting_rect_width_quarter = (counting_rect[2] - counting_rect[0])/4

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            # if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            #     if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
            #         tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            max_length = 200 / im0.shape[1] * im.shape[3]
            det = det[(det[:,2]- det[:,0]< max_length)]

            if det is not None and len(det):
                # if is_seg:
                #     shape = im0.shape
                #     # scale bbox first the crop masks
                #     if retina_masks:
                #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                #         masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                #     else:
                #         masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                # else:
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:


                    for j, (output) in enumerate(outputs[i]):
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]


                        temp = bbox_ioa(counting_rect, bboxes)
                        try:
                            #bottom horizontal of animal - bottom horizonal of counting rect
                            dict_animal_xdist_current[id] = bboxes[0] - counting_rect[0]

                        except KeyError:
                            dict_animal_xdist_initial[id] = bboxes[0] - counting_rect[0]
                            dict_animal_xdist_current[id] = dict_animal_xdist_initial[id]

                        try:
                            if dict_animal_xdist_current[id] > dict_animal_xdist_max[id]:
                                dict_animal_xdist_max[id] = dict_animal_xdist_current[id]
                        except KeyError:
                            dict_animal_xdist_max[id] = dict_animal_xdist_current[id]

                        try:
                            if dict_animal_xdist_current[id]  < dict_animal_xdist_min[id]:
                                dict_animal_xdist_min[id] = dict_animal_xdist_current[id]
                        except KeyError:
                            dict_animal_xdist_min[id] = dict_animal_xdist_current[id]

                        if cls == 0 and id not in dict_initial_animal_xyxy :
                            dict_initial_animal_xyxy[id] = bboxes

                        if cls == 2 and id not in dict_initial_persons_xyxy :
                            dict_initial_persons_xyxy[id] = bboxes

                        det_width = bboxes[2] - bboxes[0]
                        det_height = bboxes[3] - bboxes[1]
                        # if cls == 0 and det_width > 30 * 2 and det_width < 130 * 2 and det_height < 100 * 2: #for 720
                        if  cls == 0 : #  det_width > 40 and det_width < 100 and det_height < 100 and (cls == 0 or cls == 2):  # and det_width < 105 and det_height < 90  : # for 480
                            if temp > 0.2:
                                if id not in dict_initial_animal_twenty_in_xyxy:
                                    dict_initial_animal_twenty_in_xyxy[id] = bboxes

                            # animal walking to right
                            try:
                                if temp > 0.5 and dict_initial_animal_xyxy[id][0] - counting_rect[0] < 50 \
                                        and dict_initial_animal_twenty_in_xyxy[id][0] - counting_rect[0] < 50:
                                    animal_counter[id] += 1
                                    add_animal_to_indi_count(id, dict_animal_indi_count, animal_counter)


                                # animal going left
                                elif temp > 0.9 and dict_initial_animal_xyxy[id][2] - counting_rect[2] > 25 and \
                                        dict_initial_animal_twenty_in_xyxy[id][2] - counting_rect[2] > 0:
                                    animal_counter_minuser[id] += 1
                                    add_animal_to_indi_count_minuser(id, dict_animal_indi_count,
                                                                        animal_counter_minuser)

                                # animal that went left and is coming back right
                                elif temp > 0.3 and id in animal_counter_minuser and bboxes[2] - counting_rect[2] > 0:
                                    animal_counter_minuser.pop(id)
                                    dict_animal_indi_count.pop(id)

                                # animal going right that's coming back left
                                elif temp > 0.1 and id in animal_counter and bboxes[0] - counting_rect[0] < 0:
                                    animal_counter.pop(id)
                                    dict_animal_indi_count.pop(id)
                            except Exception as e:
                                print(e)

                            print_animal_count = len(animal_counter)
                            print_animal_count_minuser = len(animal_counter_minuser)

                            c = int(cls)  # integer class
                            if id in animal_counter:
                                    global_label_right = f'Going right {print_animal_count}' # {names[c]} {conf:.2f}'
                            else:
                                    global_label_left = f'Going left: {print_animal_count_minuser}' # {names[c]} {conf:.2f}'

                            annotator.box_label(counting_rect, None, color=colors(10, True))
                            if id in dict_animal_indi_count :
                                if dict_animal_indi_count[id] >= 0 :
                                    annotator.box_label(bboxes, str(dict_animal_indi_count[id]), color=(0,0,255))
                                elif dict_animal_indi_count[id] < 0:
                                    annotator.box_label(bboxes, str(dict_animal_indi_count[id]), color=(0,255,0))
                            else:
                                annotator.box_label(bboxes,'', color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))


                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            #if c == 0:
                            #    annotator.box_label(bboxes, label, color=color)
                            
                            # if save_trajectories and tracking_method == 'strongsort':
                            #     q = output[7]
                            #     tracker_list[i].trajectory(im0, q, color=color)
                            # if save_crop:
                            #     txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            #     save_one_box(np.array(bboxes, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()


            annotator.box_label((4, 0, 2, 2), global_label_right, color=colors(0, True))
            annotator.box_label((4, 60, 2, 2), global_label_left, color=colors(10, True))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                   # cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                   # cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                im0 = draw_ROI(im0, rect)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    print('animal_head:{}'.format(len(animal_counter)))
    print('animal_head_minuser:{}'.format(len(animal_counter_minuser)))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default= '/home/rama/Documents/code-repo/github/yolov8_tracking/weights/yolo/yolov8l_cattle_25042023.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'reid/fast-reid-cattle-14sept2022_osnet_x0_25.pt')
    #parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'reid/osnet_x1_0_imagenet.engine')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_false', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_false', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--animal-type', default='sheep', help='cattle, sheep, swine')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
