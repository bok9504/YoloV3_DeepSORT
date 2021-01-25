import time, random
import numpy as np
import pandas as pd
import math
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def calc_len(point1, point2):
    if len(point1) != len(point2):
        print("point matching error")
        return 0
    else:
        point1 = list(point1)        
        point2 = list(point2)
        if len(point1) == 2:
            point_len = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        elif len(point1) == 3:
            point_len = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return point_len

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    percep_frame = 5
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, FPS, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 

    fps = 0.0
    count = 0
    speed = 0
    frame_num = 0

    pointNum1 = (3555, 1101)
    pointNum4 = (1050, 564)
    realPoint1 = (392703.310, 294899.253, 48.316)
    realPoint4 = (392718.770, 294695.129, 38.871)

    frame_len = calc_len(pointNum1, pointNum4)
    real_len = calc_len(realPoint1, realPoint4)

    from _collections import deque
    pts = [deque(maxlen=percep_frame+1) for _ in range(10000)]

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        
        img = cv2.circle(img, pointNum1, 10, (0,0,0),-1)
        img = cv2.circle(img, pointNum4, 10, (0,0,0),-1)


        # 모든 프레임 저장
        #cv2.imwrite('../yolo_matching/frames/' + str(frame_num) + '.jpg' ,img)

        # 이전프레임 현 프레임 매칭
        # FLANN 알고리즘
        # if frame_num == 0:
        #     pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img = cv2.circle(img, pointNum1, 10, (0,0,0),-1)
        #     img = cv2.circle(img, pointNum4, 10, (0,0,0),-1)
        # else:
        #     aft_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #     orb = cv2.ORB_create()
        #     kp1, des1 = orb.detectAndCompute(pre_img, None)
        #     kp2, des2 = orb.detectAndCompute(aft_img, None)

        #     FLANN_INDEX_LSH = 6
        #     index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        #     search_params = dict(ckecks=32)

        #     flann = cv2.FlannBasedMatcher(index_params, search_params)
        #     matches = flann.match(des1, des2)

        #     matches = sorted(matches, key=lambda x:x.distance)
        #     matches = matches[:int(len(matches)*0.1)]
        #     #matches = [x for x in matches if x.distance<3]
        #     res = cv2.drawMatches(pre_img, kp1, aft_img, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        #     tmp_x = []
        #     tmp_y = []
        #     for mtch in matches:
        #         tmp_x.append(kp1[mtch.queryIdx].pt[0] - kp2[mtch.trainIdx].pt[0])
        #         tmp_y.append(kp1[mtch.queryIdx].pt[1] - kp2[mtch.trainIdx].pt[1])
            
        #     mod_x = int(np.mean(tmp_x))
        #     mod_y = int(np.mean(tmp_y))

        #     if mod_x == 0 and mod_y == 0:
        #         img = cv2.circle(img, pointNum1, 10, (0,0,0),-1)
        #         img = cv2.circle(img, pointNum4, 10, (0,0,0),-1)
        #     else:
        #         pointNum1 = list(pointNum1)                
        #         pointNum1[0] = pointNum1[0] + mod_x
        #         pointNum1[1] = pointNum1[1] + mod_y
        #         pointNum1 = tuple(pointNum1)

        #         pointNum4 = list(pointNum4)
        #         pointNum4[0] = pointNum4[0] + mod_x
        #         pointNum4[1] = pointNum4[1] + mod_y
        #         pointNum4 = tuple(pointNum4)

        #         img = cv2.circle(img, pointNum1, 10, (0,0,0),-1)
        #         img = cv2.circle(img, pointNum4, 10, (0,0,0),-1)

        #     pre_img = aft_img

        # frame_num+=1

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        # tlwh : (top left x, top left y, width, height)
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # tracker.tracks : tracker.tracks의 List
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # 그림 그리기
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            # 속도 추출
            if len(pts[track.track_id]) == 0:
                pts[track.track_id].append(frame_num)
            center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1]) + (bbox[3]))/2))
            pts[track.track_id].append(center)
            if len(pts[track.track_id]) == percep_frame + 1:
                move_len = np.sqrt(pow(pts[track.track_id][-1][0] - pts[track.track_id][-percep_frame][0], 2) + pow(pts[track.track_id][-1][1] - pts[track.track_id][-percep_frame][1], 2))
                realMove_Len = real_len * move_len / frame_len
                speed = realMove_Len * FPS * 3.6 / (pts[track.track_id][0]-frame_num)
                pts[track.track_id].clear()
            cv2.putText(img, " spd:" + str(int(speed)),(int(bbox[0]+(len(str(track.track_id)))*17+25), int(bbox[1]-10)),0, 0.5, (255,255,255),2)
        ## UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        # for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(80,107,227), 2)
        #    cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(classes[0])+len(str(int(scores[0]*10000)/100)))*17, int(bbox[1])), (80,107,227), -1)
        #    cv2.putText(img, str(det.class_name) + " " + str(int(det.confidence*10000)/100) + "%",(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')
        frame_num += 1
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
