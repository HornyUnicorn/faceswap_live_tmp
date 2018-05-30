import os, sys # crucial imports
import cv2

# ===================================================
# DEEPFAKES PART:
# ===================================================

from dlib_face_recognition.recog import _raw_face_landmarks

lib_dir = 'lib_deepfakes'
sys.path.append(lib_dir)
from faceswap import FaceIt, SwapGetter

swap_name = 'emma_to_jade'
faceit = FaceIt(swap_name, 'emma', 'jade') 

# ---------------------------------------
# LOADING DEEPFAKES MODEL AND CONVERTER:
# ---------------------------------------

DEEPFAKES_MODEL_LOAD_OPT = 'queue' # there were options: 1,2,3,4 'self', 'inside', ... see in archive
SHOW_SWAP_INSIDE_TRACK = 0

# ---------------------------------------
if DEEPFAKES_MODEL_LOAD_OPT == 'preload':
    path_back = os.getcwd()
    os.chdir(lib_dir)
    model = faceit.get_model()
    os.chdir(path/_back)

    converter = faceit.get_converter(model)

# ---------------------------------------
if DEEPFAKES_MODEL_LOAD_OPT == 'queue':
    from multiprocessing import Queue
    deepf_queues = [Queue(), Queue()]

    model_path = './lib_deepfakes/models/emma_to_jade'    
    deepf_instance = SwapGetter(deepf_queues, model_path=model_path)
    
    deepf_instance.start()

# ---------------------------------------

# testing live-conversion in place:
from lib.faces_detect import detect_faces_LIVE

def convert_frame_deepf(frame, converter, change_order=False):

    DEBUG_MODE = 0
    
    for face in detect_faces_LIVE(frame, "cnn", change_order=change_order):
        
        if DEBUG_MODE:
            # print(dir(face)) # image, x, y, w, h, landmarks

            print('Got face!')
            print('Face geometry: ({},{},{},{})'.format(face.x,face.y,face.w,face.h))
            print('Face landmarks: {}'.format(face.landmarks))

            cv2.imshow('Face', face.image)
            continue

        frame = converter.patch_image(frame, face)

    return frame

live = 0
if DEEPFAKES_MODEL_LOAD_OPT != 'preload': 
    live = 0

if live:
    
    PATH_TO_VIDEO = './lib_deepfakes/data/videos/emma_360_cut.mp4'
    
    print('Staring live mode ...')
    
    video_capture = cv2.VideoCapture(PATH_TO_VIDEO)
    
    frame_counter = 0    
    while True:
        frame_counter += 1
        if frame_counter == 10: # litle magic here
            break

        ret, frame = video_capture.read()
        # print(frame.shape, frame.dtype) # ((360, 640, 3), dtype('uint8'))

        if not ret:
            print("RET IS NONE ... I'M QUIT")
            video_capture.release()
            break

        print('HANDLING NEW FRAME ...')

        try: # with flip: 
            
            image = convert_frame_deepf(cv2.flip(frame, 1), converter=converter)
            image = cv2.flip(image, 1)
            print('GOT AN IMAGE (with flip)!')
          
        except:

            try: # without flip:
    
                image = _convert_frame(frame, converter=converter)
                print('GOT AN IMAGE!')

            except:

                print("HMM ... CONVERTATION FAILED ... I'M QUIT")
                video_capture.release()
                break
                # continue

        # cv2.imshow('Video', image)
        # cv2.imshow('Original', frame)

        # Hit 'q' on the keyboard to quit!
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print("KEYBOARD INTERRUPT ... I'M QUIT")
    #         video_capture.release()
    #         break

    # cv2.destroyAllWindows()
    # exit()

    sys.exit(0)
# ===================================================


# ------------------------------------------
# Supressing/enabling console output
# ------------------------------------------

# Disable printout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore ...
def enablePrint():
    sys.stdout = sys.__stdout__


# printout constants:
PRINT_SEP_UNO = '-------------------------'
PRINT_SEP_DUO = '========================='


# ==========================================
# TEST FLAGS PLACEHOLDER
# current: db_testing 17_05_18
# ==========================================

# ==============
# blockPrint() # frequently used option
# ==============

DB_TESTING_17_05_15 = 0
DB_WRITE_EMO_18_05_18 = 0


if DB_TESTING_17_05_15:
    print('{0}{0}\nEMO_DICT CREATION IS: {1}\nWRITE EMO_DICT TO DB IS: {2}\n{0}{0}\n'.format(PRINT_SEP_DUO, DB_TESTING_17_05_15, DB_WRITE_EMO_18_05_18))
    blockPrint()

    from collections import OrderedDict
else:
    DB_WRITE_EMO_18_05_18 = 0

# ------------------------------------------
# CUDA FLAG DETERMINATION:
# ------------------------------------------

try:
    import cv2.cuda
    cuda_flag = True
except:
    cuda_flag = False

print('{0}\nCUDA FLAG: {1}\n{0}\n'.format(PRINT_SEP_DUO, cuda_flag))

# ------------------------------------------
# SOME FLAGS FOR TESTS PUPROSES:
# IN GENERAL USED TO SPEED UP THE PROJECT LOADING
# ------------------------------------------


USE_YOLO = 0
IMPORT_ALPR_LOCALLY = 0
USE_TFLOW_HEAVY_STUFF = 0


print('{0}\nYOLO IS ON: {1}\nALPR LOCAL IMPORT: {2}\nEAST/CTPN SESS IMPORT: {3} \
        \n{0}\n'.format(PRINT_SEP_DUO, USE_YOLO, IMPORT_ALPR_LOCALLY, USE_TFLOW_HEAVY_STUFF))

# ------------------------------------------
# RELOAD ALPR CLASS (DIRTY): 
# (project crashes otherwise)
# ------------------------------------------

if IMPORT_ALPR_LOCALLY:

    import importlib
    mod = importlib.import_module('.openalpr', '_lib_openalpr')
    alpr = mod.Alpr("us", "", "")

    print('{0}\nALPR CLASS IS LOADED!\n{0}\n'.format(PRINT_SEP_UNO))
    
# ------------------------------------------
# PRELOAD EAST/CTPN MODEL (DIRTY)
# (project crashes otherwise)
# ------------------------------------------

if USE_TFLOW_HEAVY_STUFF:

    PATH_TO_TFLOW_PLATE_DETECT = '_text_detect_tmp'
    sys.path.insert(0, PATH_TO_TFLOW_PLATE_DETECT)
    from tflow_text_detect import GetPlate

    # Supressing FutureWarning message:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)

        path_back = os.getcwd() # DIRTY
        os.chdir(PATH_TO_TFLOW_PLATE_DETECT)

        key = 'EAST'
        key = 'CTPN'
        CAR_PLATES_SESS = {key: GetPlate(key)}

        os.chdir(path_back) # DIRTY

    print('{0}{0}\nTHE {1} SESSION WAS SUCCESFULLY LOADED!\n{0}{0}\n'.format(PRINT_SEP_UNO, key))


    # sys.exit(0) # for tests

# ------------------------------------------
# TESSERACT MODULE: 08_05_18
# ------------------------------------------

from tesseract_tmp import get_tesseract_results

# ------------------------------------------
# EMOTION BLOCK: 14_05_18
# ------------------------------------------

PRINT_EMOTIONS_ONLY = 0
if PRINT_EMOTIONS_ONLY:
    enablePrint() 

APPLY_EMO = 0 # DETECT EMOTIONS WITH EMOTION-MODULE

APPLY_GENDER = 0 # DETECT GENDER WITH EMOTION-MODULE
DRAW_EMOTIONS = 0 # FLAG FOR DRAWING 'EMOTION'-TEXT ON FRAME

if APPLY_EMO:
    from _lib_emotions.src.emotion_from_frame import get_emotion, get_gender
    print('{0}\nEMOTIONS DETECTION IS ON\nGENDER DETECTION (FROM EMO-BLOCK) IS: {1}\nEMOTION DRAWING IS: {2}\n{0}\n'.format(PRINT_SEP_DUO, APPLY_GENDER, DRAW_EMOTIONS))
    TEST_IN_PLACE_EMO = 1
else:
    APPLY_GENDER, DRAW_EMOTIONS = 0, 0
    TEST_IN_PLACE_EMO = 0

if TEST_IN_PLACE_EMO: # NEED TO RUN A SIMPLE TEST CASE (project crashes otherwise)
        
    print('{0}{0}\nRUNNING THE EMOTIONS TEST CASE ...\n'.format(PRINT_SEP_UNO))

    PATH_TO_EMOTION_TEST_FILE = 'data/emotio/000.jpg'
    face_frame = cv2.imread(PATH_TO_EMOTION_TEST_FILE)

    # WITHOUT GENDER:
    emo, time_str = get_emotion(face_frame, get_time=True)
    print('THE EMOTION: {0}\n{1}'.format(emo, time_str))

    if APPLY_GENDER: # WITH GENDER
        gen, time_str = get_gender(face_frame, get_time=True)
        print('THE GENDER: {0}\n{1}'.format(gen, time_str))

    print('{0}{0}'.format(PRINT_SEP_UNO))


    # sys.exit(0) # for tests

if PRINT_EMOTIONS_ONLY:
    blockPrint()

# ------------------------------------------
# THREAD WITH RETVAL: 11_05_18
# (maybe redundant)
# ------------------------------------------

from threading import Thread
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return

# ---------------------------------
# JAILBASE INDEX: 20_04_18
# ---------------------------------

from tmp_dlib_test import FaissIndex

METHODS_ALLOWED = FaissIndex.PATHS_TO_INDEX_DEFAULT.keys() # ('resnet', 'openface')
INDEX_JAIL = {method: FaissIndex(method=method) for method in METHODS_ALLOWED}

# ---------------------------------

from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QString, QRunnable, QThreadPool, QMutex
import numpy.core._methods
import numpy.lib.format
from imutils.face_utils import FaceAligner
from face_search import *
from multiprocessing import Process, Pipe, Queue
import cProfile, pstats, StringIO

from dlib_face_recognition.recog import compare_faces, pose_predictor_68_point
from FaceSwap import *
from yolo_detector import YoloDetector
from __number_plates_custom import find_plate
from __number_plates import get_plate_boxes

# ---------------------------------------------

from pydispatch import dispatcher
from caffe_processing import Caffe, face_parts, col
# s = Sender()
# net = Caffe()
# dispatcher.connect(net.get_segm, signal='segmentation')
# dispatcher.connect(net.get_age_gender, signal='age_gender')


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


def recogFace(orgImg, faceImg):
    try:
        orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2GRAY)
        faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
        w, h = orgImg.shape[::-1]
        res = cv2.matchTemplate(faceImg, orgImg, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            return True
        return False
    except:
        print("Error Template Matching")
        return False

def match_template(temp_color, faceIdx, frame, y, x, h, w, this_face):
    bExistFace = False
    existFaceIdx = -1
    for beforeIdx in range(0, faceIdx):
        if beforeIdx == this_face:
            continue
        full_path = os.getcwd()
        full_path = full_path.replace("\\", "/")
        filePath = full_path + "/faceTemplates/" + str(beforeIdx) + '.png'
        orgImg = cv2.imread(filePath)
        h1, w1 = orgImg.shape[0:2]
        yt, xt = y, x
        wt, ht = 0, 0
        newh, neww = h, w
        if h1 > h:
            ht = int(numpy.ceil((h1 - h + 1) / 2))
            yt = max(0, y - ht)
            newh = h1
        if w1 > w:
            wt = int(numpy.ceil((w1 - w + 1) / 2))
            xt = max(0, x - wt)
            neww = w1

        if [x, y, 0, 0] != [xt, yt, wt, ht]:
            temp_color2 = frame[yt:yt + newh, xt:xt + neww]
            bExistFace = recogFace(orgImg, temp_color2)
        else:
            bExistFace = recogFace(orgImg, temp_color)

        if bExistFace == True:
            existFaceIdx = beforeIdx
            break

    return bExistFace, existFaceIdx


# 8765
# ====================================================================
# FAISS BLOCK:
    # - index creation
    # - searching within it

# also added improvements for better performance on large datasets:
    # - dataset segmentation (via Voronoi's cells)
    # https://github.com/facebookresearch/faiss/wiki/Faster-search
    # - shrinkage the dataset (SUSPENDED - IT NEEDS BIGGER DATASET FOR CORRECT TESTING)
    # https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint
# ====================================================================

# importing the faiss module:
import faiss

from copy import deepcopy

from dlib_face_recognition.recog import face_encodings
from openface_matching import get_represent_features, set_openface_cpu_mode, set_openface_gpu_mode

templates_dir = 'faceTemplates' # database

project_dir = os.getcwd()
path_to_templates = os.path.join(project_dir, templates_dir)
path_to_vector = os.path.join(path_to_templates, '1.png')  # query vector

# names of encode methods and index files:
enc_resn_name = 'resnet'
enc_open_name = 'openface'
path_to_index_resn = "__index_resn.index"
path_to_index_open = "__index_open.index"
PATH_TO_INDEX_SWITCHER = {enc_resn_name:path_to_index_resn,
                          enc_open_name:path_to_index_open}

# BUG REPORT (*3452): see__for_tests.py
# ----------------------
# global WAS_LOADED
# WAS_LOADED = False
# ----------------------

# ---------------------------------------
# DATA PREPARATION TO FIT FAISS MODEL:
# ---------------------------------------

def get_encoded_frame(frame, method='resnet'):
    # auxiliary function - encodes frame(array) in a way depending on the value of 'method' arg
    if method == 'resnet':
        return face_encodings(frame, [[1, frame.shape[1], frame.shape[0], 1]])[0]
    if method == 'openface':
        return get_represent_features(frame)
    else:
        raise ValueError("METHOD NAME IS INCORRECT! (should be 'resnet' or 'openface')")

def get_faiss_vector(path_to_vector, method='resnet', is_frame=False):
    # receiving 'faiss-ready' vector from path to img,
    # for example, for using it in searching via index

    if not is_frame:
        # print("FVECTOR. HANDLING:", path_to_vector)
        frame = cv2.imread(path_to_vector)
    else:
        frame = path_to_vector
    frame_encoded = get_encoded_frame(frame, method)
    faiss_vector = frame_encoded[np.newaxis].astype('float32')

    return faiss_vector

def get_faiss_matrix(path_to_templates, img_ext='.png', method='resnet'):
    # receiving 'faiss-ready' matrix from path to database of imgs
    # for example, for using it for training index

    temp_matrix = []
    for file in sorted(os.listdir(path_to_templates)): # ?sorted
        if file.endswith(img_ext):
            print(file)
            path_to_vector = os.path.join(path_to_templates, file)
            faiss_vector = get_faiss_vector(path_to_vector, method)
            temp_matrix.append(faiss_vector)

    if temp_matrix == []:
        return None
    else:
        faiss_matrix = np.concatenate(temp_matrix).astype('float32')
        return faiss_matrix

# ---------------------------------------
# USING FAISS FUNCTIONALITY:
# ---------------------------------------

def index_init(encode_method='resnet', search_type='BruteForce'):
    # getting an index from database of imgs
    # it's used at start of the module in the case an appropriate index file doesn't exist
    # (see get_index_start)

    # search_type == 'BruteForce': standart index => bruteforce searching
    # search_type == 'Voronoi': trained index (Voronoi's cells) => improved searching
    # search_type ~ 'Compressed' IS SUSPENDED - IT NEEDS BIGGER DATASET FOR CORRECT TESTING

    INDEX_D = 128  # dimension, see: recog.py --> face_encodings documentation

    project_dir = os.getcwd()
    path_to_templates = os.path.join(project_dir, templates_dir)

    index = None

    train_matrix = get_faiss_matrix(path_to_templates, method=encode_method)
    if train_matrix is None:
        # print("THE TRAIN MATRIX IS EMPTY, THE INDEX WASN'T CREATED!")
        return index

    nlist = train_matrix.shape[0]
    print(nlist)

    if search_type == 'BruteForce':
        # -----------------------------------------------
        # standart index (Brute-Force)
        # -----------------------------------------------

        # print("BRUTEFORCE RUNNING ...")

        index = faiss.IndexFlatL2(INDEX_D)  # build the index

    elif search_type == 'Voronoi':
        # -----------------------------------------------
        # ~kinda make it faster (Voronoi)
        # -----------------------------------------------

        # print("VORONOI's IMPROVEMENT RUNNING ...")

        # nlist = num_of_clusters => SHOULD BE less or equal of num_of_vectors in the database

        quantizer = faiss.IndexFlat(INDEX_D)  # the other index
        index = faiss.IndexIVFFlat(quantizer, INDEX_D, nlist, faiss.METRIC_L2)
        index.train(train_matrix)

    else:
        raise ValueError("SEARCH_TYPE NAME IS INCORRECT! (should be 'BruteForce' or 'Voronoi')")

    index_update(train_matrix, encode_method=encode_method, search_type=search_type)

    return index

def index_update(faiss_obj, encode_method='resnet', search_type='BruteForce'):
    # updating the existing index and saving it to an appropriate file
    # print("UPDATING INDEX")
    path_to_index = os.path.join("__FAISS_INDEX", os.path.join(search_type, PATH_TO_INDEX_SWITCHER[encode_method]))
    index = None
    try:
        index = faiss.read_index(path_to_index)
    except:
        pass
    if index is None:
        # index == None ==> creation index instead of updating;
        # NOTE: faiss_obj should have the same encode_method as the input argument

        INDEX_D = 128

        if search_type == 'BruteForce':
            index = faiss.IndexFlatL2(INDEX_D)
        elif search_type == 'Voronoi':
            nlist = faiss_obj.shape[0]
            quantizer = faiss.IndexFlat(INDEX_D)
            index = faiss.IndexIVFFlat(quantizer, INDEX_D, nlist, faiss.METRIC_L2)
            index.train(faiss_obj)

    if not os.path.exists("__FAISS_INDEX"):
        os.makedirs("__FAISS_INDEX")
    if not os.path.exists(os.path.join("__FAISS_INDEX", search_type)):
        os.makedirs(os.path.join("__FAISS_INDEX", search_type))
    path_to_index = os.path.join(os.path.join("__FAISS_INDEX", search_type), PATH_TO_INDEX_SWITCHER[encode_method])

    # print("ADDING FAISS OBJECT")
    index.add(faiss_obj)  # adding vector/matrix to the index
    # print("SAVING INDEX")
    faiss.write_index(index, path_to_index) # saving index
    return index

def get_index_start(encode_method='resnet', search_type='BruteForce'):
    # getting the index at start of the module:
        # - or from appropriate file (if exists)
        # - either from database of imgs

    path_to_index = os.path.join("__FAISS_INDEX", os.path.join(search_type, PATH_TO_INDEX_SWITCHER[encode_method]))

    try:
        index = faiss.read_index(path_to_index)

    except:
        index = index_init(encode_method=encode_method, search_type=search_type)

    return index

def index_search(path_to_img, is_frame=False, encode_method='resnet', search_type='BruteForce', num_neighb=1, nprobe=1, update=False):
    path_to_index = os.path.join("__FAISS_INDEX", os.path.join(search_type, PATH_TO_INDEX_SWITCHER[encode_method]))

    try:
        index = faiss.read_index(path_to_index)
    except:
        return False, -1
    # searching for nearest neighbors of some input image using the index
    query_vector = get_faiss_vector(path_to_img, method=encode_method, is_frame=is_frame)
    if index is None:
        # index_update(index, query_vector, encode_method=encode_method, search_type=search_type)
        return False, -1
    if search_type == 'Voronoi':
        index.nprobe = nprobe

    D, I = index.search(query_vector, num_neighb)
    # if update:
    #     index_update(index, query_vector, encode_method=encode_method, search_type=search_type)
    return D, I

# ====================================================================

def resnet(temp_color, faceIdx, frame, y, x, h, w, this_face):

    bExistFace = False
    existFaceIdx = -1

    encodings = face_encodings(temp_color, [[0, temp_color.shape[1], temp_color.shape[0], 0]])

    for beforeIdx in range(0, faceIdx):
        if beforeIdx == this_face:
            continue
        full_path = os.getcwd()
        full_path = full_path.replace("\\", "/")
        filePath = full_path + "/faceTemplates/" + str(beforeIdx) + '.png'

        orgImg = cv2.imread(filePath)
        try:
            known_encodings = face_encodings(orgImg, [[1, orgImg.shape[1], orgImg.shape[0], 1]])
        except AttributeError:
            continue

        results = compare_faces(known_encodings, encodings[0])

        if results:
            bExistFace = True

        if bExistFace == True:
            existFaceIdx = beforeIdx
            break
    return bExistFace, existFaceIdx


# This code is the example of FAISS in work.

if __name__ == "__main__":
    frame1 = cv2.imread("Screenshots/man1.jpg")
    frame2 = cv2.imread("Screenshots/man2.jpg")
    index = get_index_start('resnet', 'BruteForce')
    vector1 = get_faiss_vector(frame1, method='resnet', is_frame=True)
    vector2 = get_faiss_vector(frame2, method='resnet', is_frame=True)
    # index_update(vector1, encode_method='openface')
    D, I = index_search(frame1, encode_method='resnet', is_frame=True)
    print(D, I)
    D, I = index_search(frame2, encode_method='resnet', is_frame=True)
    print(D, I)

# ====================================================================

def get_face(faces, width=0, flip=False):
    result = []
    for (x, y, w, h) in faces:
        if not flip:
            result.append((x, y, x + w, y + h))
        else:
            result.append((width - x - w, y, width - x, y + h))

    return result


queues = [Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), Queue()]
yolo_queues = [Queue(), Queue()]


from face_search import update_event, insert_event, insert_group, insert_photo, insert_photo_group, insert_search
def save_photo(temp_color, rgb_color):

    if DB_WRITE_EMO_18_05_18:
        enablePrint()

    full_path = os.getcwd()
    path = os.path.join(full_path, 'faceTemplates')
    # print("inserting photo")

    # -------------------------------------------
    # EDITED: len(os.listdir(path)) is OS SPECIFIC
    # -------------------------------------------
    
    files_in_path = os.listdir(path)
    id = len(files_in_path)
    if '.directory' in files_in_path:
        id -= 1
    # -------------------------------------------

    insert_photo(path, id)
    # print("inserting event")
    event_id = insert_event(id)

    # --------------------------------------
    if DB_WRITE_EMO_18_05_18:
        enablePrint()
        print('\n\nINSIDE SAVE PHOTO ...')
        print('EVENT_ID: {}'.format(event_id))
        insert_emo(event_id)
    # --------------------------------------

    cv2.imwrite(os.path.join(path, '{0}.png'.format(id)), temp_color)
    # print("saving image {0}".format(os.path.join(path, '{0}.png'.format(id))))
    cv2.imwrite(os.path.join(path.replace('faceTemplates', 'faces'), '{0}.png'.format(id)), rgb_color)

    # TODO: fix search type and encode method
    try:
        path_to_index = os.path.join("__FAISS_INDEX", os.path.join("BruteForce", PATH_TO_INDEX_SWITCHER["openface"]))
        faiss.read_index(path_to_index)

        vector = get_faiss_vector(temp_color, method='openface', is_frame=True)
        index_update(vector, encode_method='openface', search_type='BruteForce')
        vector = get_faiss_vector(temp_color, method='resnet', is_frame=True)
        index_update(vector, encode_method='resnet', search_type='BruteForce')
    except:
        pass

    queues[0].put((rgb_color, os.path.join(path.replace('faceTemplates', 'faces'), '{0}.png'.format(id))))
    tup = queues[1].get()
    save_face(tup)

    if DB_WRITE_EMO_18_05_18:
        blockPrint()

    return id, event_id


# thr = threading.Thread(target=caffe_processing, args=(net,))
# thr.start()
def get_parsing(im, rect):
    queues[4].put((im, rect))
    # ll = len(os.listdir('masks_'))
    # if '.directory' in os.listdir('masks_'):
    #     ll -= 1
    # cv2.imwrite("masks_/{}.png".format(ll), im)
    masks = queues[5].get()
    mask = np.zeros(im.shape[:2], dtype=np.uint8)
    im_ = im.astype(np.float32)
    for i, part in enumerate(face_parts):
        im_[masks[part].astype(bool)] = 0.6 * im_[masks[part].astype(bool)] + 0.4 * np.array(col[i])
    return im_.astype(np.uint8)
    # ll = len(os.listdir('masks'))
    # if '.directory' in os.listdir('masks'):
    #     ll -= 1
    # cv2.imwrite('masks/{}.png'.format(ll), im_.astype(np.uint8))


def save_face(tup):
    out, mask, path = tup
    # dispatcher.send(im=im, signal='segmentation')
    # out, _ = out_q.get()
    # in_q.put((im, 0))
    # out, _ = out_q.get()
    print("\tFORWARDED")
    cv2.imwrite(path.replace('.png', '_segm.png'), out)


class TrackerTypeChanged(Exception):
    def __init__(self):
        Exception.__init__(self)


dlib_cuda_lock = Lock()


class getPostThread(QThread):

    bThreading = True
    swapPath = ""

    def __init__(self, path, stream=0):
        QThread.__init__(self)
        self.path = path
        self.gpu = cuda_flag
        self.faceIdx = None
        self.cap = None
        self.stream = stream
        self.paused = False
        self.type = None
        self.method = None
        self.new_type = None
        self.new_method = None
        self.find_matches = None
        self.match_pool = None
        self.recog_pool = None
        self.swap_pool = None
        self.save_pool = None
        self.display_pool = None
        self.find_match_pool = None
        self.group_pool = None
        self.yolo_pool = None
        self.data = None
        self.min_area = 80
        self.trackers = []
        self.track_op = "dlib"
        self.false_rects = []
        self.disable_tracking = False
        self.num_tracked = 0
        self.new_faces = []
        self.old_faces = []
        self.initial_faces = 0
        self.aligner = FaceAligner(pose_predictor_68_point, desiredFaceWidth=256)
        self.align = True
        self.swapPath = None
        self.do_swap = False
        self.search = "PicTriev"
        self.swaps = []
        self.new_swaps = []
        self.swap_data = ['']
        self.swap_groups_active = []
        self.swap_groups = []
        self.swappers = []
        self.groups = []
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=20, detectShadows=False)
        self.display_data = None
        self.faces = None
        self.faces_lock = QMutex()
        self.started = None
        self.events = None
        self.ids = None
        self.estimator = "KERAS"
        self.search_type = 'BruteForce'
        self.encode_method = 'resnet'
        self.new_encode_method = self.encode_method
        self.new_search_type = self.search_type
        self.mode = "faces" # or 'yolo'
        self.new_mode = self.mode
        # t0 = time.time()

        self.parse_face = False

        # ----------------------------
        # 14_05_18
        if USE_YOLO:
            self.yolo = YoloDetector(net_path='models/yolo/yolov3.weights',
                                     config_path='models/yolo/cfg/yolov3.cfg',
                                     gpu=True,
                                     queues=yolo_queues)
            self.yolo.start()
        # ----------------------------

        self.yolo_objects = []
        self.object_trackers = []
        self.need_yolo_update = True
        self.skip = 0
        self.plate_algo = "anpr"

        self.draw_rects = True
        self.classifiersList = None
        self.old_track = None
        self.unite_rects = False

        self.last_bboxes = []

        # (8765 - code to find Maxim's edits)
        # ========================================================
        self.cap_frame_w = None
        self.cap_frame_h = None
        # ========================================================

        self.detector = dlib.get_frontal_face_detector()
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
        self.side_cascade = cv2.CascadeClassifier('models/side.xml')
        self.front_cascade = cv2.CascadeClassifier('models/front.xml')


        self.save_pool = QThreadPool()
        self.save_pool.setMaxThreadCount(1)
        self.find_match_pool = QThreadPool()
        self.find_match_pool.setMaxThreadCount(1)
        self.group_pool = QThreadPool()
        self.group_pool.setMaxThreadCount(2)
        self.yolo_pool = QThreadPool()
        self.yolo_pool.setMaxThreadCount(6)
        self.match_pool = QThreadPool()
        self.match_pool.setMaxThreadCount(1)
        self.display_pool = QThreadPool()
        self.display_pool.setMaxThreadCount(2)
        self.recog_pool = QThreadPool()
        self.recog_pool.setMaxThreadCount(2)
        self.caffe = Caffe(queues=queues)
        self.caffe.start()

        # -------------------------------
        # 20_04_18:
        # -------------------------------

        self.jailbase_flag = True
        self.jailbase_threshold = 0.3

        # -------------------------------

        get_index_start('openface', 'BruteForce')
        get_index_start('resnet', 'BruteForce')

        # -------------------------------
        # 16_05_18:
        # -------------------------------
        self.emo_state = 0 # 30_05_18
        self.emo_gen_state = 0
        self.emo_draw_state = 0

        self.emo_count_dict = None
        if DB_TESTING_17_05_15:
            self.emo_count_dict = OrderedDict()

        self.frame_count = 0

        # -------------------------------
        # 29_05_18:
        # -------------------------------

        if DEEPFAKES_MODEL_LOAD_OPT == 'self':
    
            swap_name = 'emma_to_jade'
            faceit = FaceIt(swap_name, 'emma', 'jade')
            self.deepf_model = run_func_inside_dir(lib_dir, faceit.get_model)
            self.deepf_converter = faceit.get_converter(self.deepf_model)

        # -------------------------------

    def __del__(self):
        self.wait()

    def setType(self, type):
        if not type in ["dlib", "cascade", "cnn"]:
            self.emit(SIGNAL("QtSigNoType"))
            return
        self.type = type

    def set_method(self, method_name):
        if not method_name in ['template', 'faiss']:
            self.emit(SIGNAL("QtSigNoMethod"))
            return
        self.method = method_name

    def setPath(self, path):
        self.path = path

    def get_detector(self):
        if self.type == "dlib":
            return dlib.get_frontal_face_detector()
        elif self.type == "cascade":
            return [cv2.CascadeClassifier('side.xml'), cv2.CascadeClassifier('front.xml')]
        else:
            return dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

    # @profile
    def run(self):
        if not os.path.exists('faces'):
            os.makedirs(str('faces'))

        if not os.path.exists('faceTemplates'):
            os.makedirs(str('faceTemplates'))

        self.faceIdx = 0
        fileList = os.listdir(str('faces'))
        for fileName in fileList:
            self.faceIdx = self.faceIdx + 1

        self.initial_faces = self.faceIdx

        self.need_yolo_update = True

        cap = None
        image = None
        if type(self.stream) == int or type(self.stream) == str and not self.stream.split('.')[-1] in ['png', 'jpg', 'bmp']:
            cap = cv2.VideoCapture(self.stream)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

            # (8765)
            # ========================================================
            assert self.cap_frame_w and self.cap_frame_h, "CAP FRAME W,H ARE NOT SET!"
            print(self.cap_frame_w, self.cap_frame_h)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_frame_h)
            # ========================================================

            codec = 1196444237.0  # MJPG
            cap.set(cv2.CAP_PROP_FOURCC, codec)
            if (cap.isOpened() == False):
                print("Error Connecting Camera")
                self.emit(SIGNAL("QString"), QString("Cap is not opened."))
        else:
            image = cv2.imread(self.stream)

        self.cap = cap
        self.started = False

        if cap is None:
            self.run2(image, self.faceIdx)
        else:
            self.run2(cap, self.faceIdx)

        self.paused = False

        self.emit(SIGNAL("PyQtFinished"))
        self.quit()

    #New process with tracking and background substraction
    # @profile
    def run2(self, cap, face_idx):

        self.faceIdx = face_idx

        self.groups = []

        #For images
        r = getattr(cap, "read", None)
        if r is None:
            self.type = self.new_type
            self.encode_method = self.new_encode_method
            self.search_type = self.new_search_type
            self.mode = self.new_mode
            update_faiss = False
            if self.method != self.new_method and self.new_method == "faiss":
                update_faiss = True
            self.method = self.new_method
            self.swaps = self.new_swaps
            if self.type == "dlib":
                self.classifiersList = [self.detector]
            elif self.type == "cnn":
                self.classifiersList = [self.cnn_face_detector]
            else:
                self.classifiersList = [self.side_cascade, self.front_cascade]
            self.do_swap = False
            if self.mode == "yolo":
                gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
                tmp_frame = cv2.GaussianBlur(gray, (31, 31), 0)
                mask = self.bg_sub.apply(tmp_frame)
                ker = np.ones((5, 5))
                mask = cv2.dilate(mask, ker)
                a = np.linspace(0.5, 1, num=2000)
                size = (int(a[mask.shape[0]] * 50 if mask.shape[0] < 2000 else 50),
                        int(a[mask.shape[1]] * 50 if mask.shape[1] < 2000 else 50))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(size))
                res = self.process_frame_yolo(cap, [(0, 0, cap.shape[1], cap.shape[0])])
                self.process_results_yolo(cap, res, gray, mask, True)
                return
            else:
                self.get_faces(cap, classifiers=self.classifiersList, track=False, start=True)
                for i in range(10):
                    time.sleep(1)
                    self.emit(SIGNAL("PyQtUpdateImages"))
                return

        start = True

        update_count = 0

        self.ids = []
        self.old_track = self.track_op
        # self.new_mode = "yolo"

        prev_cnt = 0
        prev_mask = None
        ker = np.ones((5, 5))
        a = np.linspace(0.5, 1, num=2000)

        # prof = Profiler()
        # prof.start()

        detect_count = 0
        detect_inner_count = 0
        self.swap_count = 0
        self.swap_init_count = 0
        self.tracker_init_count = 0
        frame_count = 0
        self.update_count = 0

        pr = cProfile.Profile()
        pr.enable()

        self.new_method = self.method
        self.new_encode_method = self.encode_method
        self.new_search_type = self.search_type
        self.new_type = self.type

        if self.gpu:
            set_openface_gpu_mode()
        else:
            set_openface_cpu_mode()

        while (cap.isOpened()):
            if self.bThreading == False:
                break
            if self.paused:
                continue
            self.type = self.new_type
            self.encode_method = self.new_encode_method
            self.search_type = self.new_search_type
            self.mode = self.new_mode
            self.method = self.new_method
            self.swaps = self.new_swaps
            if self.type == "dlib":
                self.classifiersList = [self.detector]
            elif self.type == "cnn":
                self.classifiersList = [self.cnn_face_detector]
            else:
                self.classifiersList = [self.side_cascade, self.front_cascade]
            if self.method == "faiss":
                self.recog = self.faiss_recog
            else:
                self.recog = match_template
            ret, frame = cap.read()
            if type(self.stream) == int:
                frame = cv2.flip(frame, 1)

            frame_count += 1

            try:
                tmp_frame = cv2.GaussianBlur(frame, (31, 31), 0)
            except:
                self.bThreading = False
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = self.clahe.apply(gray_frame)
            gray3d = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            mask = self.bg_sub.apply(tmp_frame)

            if self.bThreading == False:
                break

            count, semi_final, rects = self.count_faces2(mask)

            if self.unite_rects:
                final = self.unite_close_rects(semi_final, 10)
            else:
                final = semi_final

            if len(final) > 0:
                detect_count += 1
                detect_inner_count += len(final)

            # -------------------------------------
            # if DB_TESTING_17_05_15:
            if 0:
                enablePrint()
                print('\n============\n{}'.format(frame_count))
                self.frame_count = frame_count
                blockPrint()

            self.frame_count = frame_count # 30_05_18
            # -------------------------------------

            if self.mode == "faces":

                faces = self.detect(gray3d, final)
                new_frame = len(final) == 1 and final[0][2] * final[0][3] > 0.7 * frame.shape[0] * frame.shape[1]

                # start = True # 30_05_18
                if start and ret:
                # if 1: # 30_05_18
                    # if DB_TESTING_17_05_15:
                    #     enablePrint() 
                    #     print('CONDITION_1')
                    #     print('============\n')
                    #     blockPrint()
                    faces, _, update, draw_fcs, draw_objs = self.get_faces(frame, classifiers=self.classifiersList,
                                                                           track=True, start=True, old_track=self.old_track, faces=faces, gray=gray_frame)
                    start = False

                elif ret and new_frame or self.old_track != self.track_op:
                # elif 0: # 30_05_18
                    # if DB_TESTING_17_05_15: 
                    #     enablePrint() 
                    #     print('CONDITION_2 ... NEWFRAME: {} ... TRACKS_NOT_EQUAL: {}'.format(new_frame, self.old_track != self.track_op))
                    #     print('============\n')
                    #     blockPrint()

                    self.get_faces(frame, classifiers=self.classifiersList, track=True, old_track=self.old_track, faces=faces, gray=gray_frame, new_frame=True)
                elif ret:
                # elif 0: # 30_05_18
                    # if DB_TESTING_17_05_15:
                    #     enablePrint() 
                    #     print('CONDITION_3 (MAIN) ... update_events: {}'.format(update_count == 0))
                    #     print('============\n')
                    #     blockPrint()

                    #main procedure
                    try:
                        faces, _ = self.track(frame, gray3d, mask, update_count == 0)

                        if DB_WRITE_EMO_18_05_18 and update_count == 0:
                            enablePrint() 
                            self.emo_count_dict = OrderedDict((key, dict()) for key in self.emo_count_dict) 
                            print('*********** EMO_COUNT_DICT WAS REFRESHED! ***********')
                            blockPrint()

                    except TrackerTypeChanged:
                        faces, _, update, draw_fcs, draw_objs = self.get_faces(frame, classifiers=self.classifiersList,
                                                                               track=True, old_track=self.old_track, faces=faces, gray=gray_frame, new_frame=True)
                else:
                    # when ret is False assuming stream ended/no longer accessible
                    self.bThreading = False
                    break

                if self.bThreading == False:
                    break

            elif self.mode == "yolo":

                res = self.process_frame_yolo(frame, final)
                self.process_results_yolo(frame, res, gray_frame, mask, update_count == 0)

            if update_count == 0:
                self.emit(SIGNAL("PyQtUpdateImages"))
            update_count += 1
            if update_count == 20:
                update_count = 0
            self.old_track = self.track_op

        self.trackers = []
        self.object_trackers = []

        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

        print("Det: {}\nDet inner: {}\nFrames: {}\nSwaps: {}\nSwap inits: {}\nTrack inits: {}\nTrack upds:{}"
              .format(detect_count, detect_inner_count, frame_count, self.swap_count, self.swap_init_count,
                      self.tracker_init_count, self.update_count))

        # prof.stop()
        # print(prof.output_text(color=True))
        self.yolo_pool.setExpiryTimeout(0.1)
        cap.release()

    # For processing in separate thread, must have given better processing speed
    def yolo_process(self, frame):
        print("\tYoloProcessing")
        YOLO = YoloProcessing(self.yolo, frame, self.yolo_objects)
        # YOLO.signals.frame_displayed.connect(self.display_frame)
        YOLO.signals.results.connect(self.process_yolo_results)
        self.yolo_pool.start(YOLO)

    # For processing in-place
    def process_frame_yolo(self, frame, rects):
        results = dict()
        print(rects)
        if len(self.yolo_objects) > 0:
            for obj in self.yolo_objects:
                results[str(obj)] = []
            for x, y, w, h in rects:
                analyzed_frame = frame[y:y + h, x:x + w, :]
                print(analyzed_frame.shape)
                yolo_queues[0].put((analyzed_frame, self.yolo_objects))
                # tmp_results, _ = self.yolo.process_frame(analyzed_frame, self.yolo_objects)
                tmp_results = yolo_queues[1].get()
                for res in tmp_results.items():
                    for r in res[1]:
                        results[res[0]].append((r[0] + x, r[1] + y, r[2] + r[0] + x, r[1] + r[3] + y))

        return results

    def process_results_yolo(self, frame, res, gray, mask, update_events):
        self.min_area = min(self.min_area, frame.shape[1] * frame.shape[0] / 200)

        people = []
        if "person" in res:
            people = res['person']

        print("YOLO: {}".format(res))
        faces = self.detect(frame, people)
        items = res.items()
        length = np.sum(np.array([len(r[1]) for r in items]))
        displayFrame = frame.copy()
        if self.gpu:
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if length > 0:
            _, _, _, draw_fcs, draw_objs = self.get_faces(frame, classifiers=self.classifiersList, track=True,
                                                          start=False, old_track=self.old_track, display=False,
                                                          people_bboxes=people, objects_bboxes=res, faces=faces)
            draw_objs = [(i + self.num_tracked) for i in draw_objs]
            _, displayFrame = self.track(displayFrame, gray, mask, update_events, display=False, trackers_to_update=draw_fcs+draw_objs, change_last=False)
            for key, r in items:
                self.yolo.draw(displayFrame, r, key)
        else:
            self.track(frame, gray, mask, update_events, display=False)

            col = (0, 255, 0)
            for i in range(len(self.trackers) + len(self.object_trackers)):
                bbox = self.last_bboxes[i]
                if i == len(self.trackers):
                    col = (0, 0, 255)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1]+bbox[3]))
                if self.draw_rects:
                    cv2.rectangle(displayFrame, p1, p2, col, displayFrame.shape[1] / 100)

        self.emit(SIGNAL("signal(PyQt_PyObject)"), displayFrame)
        return faces

    def display_frame(self, frame):
        if self.bThreading:
            self.emit(SIGNAL("signal(PyQt_PyObject)"), frame)

    def process_yolo_results(self, frame, res):
        if self.bThreading:
            # analyze for subject of people
            tmp_frame = cv2.GaussianBlur(frame, (31, 31), 0)
            mask = self.bg_sub.apply(tmp_frame)
            self.min_area = min(self.min_area, mask.shape[1] * mask.shape[0] / 200)

            people = None
            print("Results: ", res)
            items = res.items()
            length = np.sum(np.array([len(r[1]) for r in items]))
            if length > 0:
                if "person" in res:
                    people = res['person']
                faces, displayFrame, _ = self.get_faces(frame, classifiers=self.classifiersList, track=True,
                                                        start=False, old_track=self.old_track, display=False,
                                                        people_bboxes=people, objects_bboxes=res)
            else:
                faces = []
                displayFrame = frame.copy()
                self.skip = 5

            if length > 0:
                for key, r in items:
                    self.yolo.draw(displayFrame, r, key)
            else:
                col = (0, 255, 0)
                for i in range(len(self.trackers) + len(self.object_trackers)):
                    ok, bbox = self.update_tracker(i, frame, mask)
                    if i == len(self.trackers):
                        col = (0, 0, 255)
                    if ok:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        if self.draw_rects:
                            cv2.rectangle(displayFrame, p1, p2, col, displayFrame.shape[1] / 100)

            self.emit(SIGNAL("signal(PyQt_PyObject)"), displayFrame)
            return faces
        else:
            return []

    def check_person(self, person, frame, mask, track=True):
        x, y, w, h = person
        left = x
        right = x+w
        low = y
        high = y+h
        dh = h / 10
        dw = w / 10
        if track:
            for i in range(len(self.trackers)):
                try:
                    ok, bbox = self.update_tracker(i, frame, mask)
                except:
                    ok = False
                if ok:
                    points = [(bbox[0], bbox[1]), (bbox[0], bbox[1]+bbox[3]), (bbox[0]+bbox[2], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])]
                    for p in points:
                        if p[0] < left - dw or p[0] > right + dw or p[1] < low - dh or p[1] > high + dh:
                            print("Unchecked")
                            return False
            print("TRACKERS:", self.trackers)
            if len(self.trackers) > 0:
                print("Checked")
            else:
                print("Unchecked")
            return len(self.trackers) > 0
        else:
            for bbox in [frame]:
                points = [(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3])]
                for p in points:
                    if p[0] < left - dw or p[0] > right + dw or p[1] < low - dh or p[1] > high + dh:
                        return False
            return True

    def update_swaps(self, tracks):
        # self.emit(SIGNAL("PyQtUpdateSwaps"), tracks)
        pass

    # @profile
    def track(self, frame, analyzed_frame, mask, update_events, display=True, trackers_to_update=None, change_last=True):
        print("\ttracking: {0} people, {1} objects".format(len(self.trackers), len(self.object_trackers)))
        displayFrame = frame.copy()

        # =====================================================
        if SHOW_SWAP_INSIDE_TRACK and DEEPFAKES_MODEL_LOAD_OPT == 'queue':
            
            print('\nFRAME (track): {}'.format(frame.shape))
            
            deepf_queues[0].put(cv2.flip(frame, 1))
            print('PUT A FRAME (with flip)!')
            
            while 1:
                try:
                    image = deepf_queues[1].get()
                    print('GOT AN IMAGE ... get_faces!')
                    # print(type(image))
                    # print(image.shape)
                    
                    image = cv2.flip(image, 1)

                    # path = './data/deepf_{}.png'.format(self.frame_count)
                    # cv2.imwrite(path, image)

                    displayFrame = image
                    break
                except:
                    pass
        # =====================================================

        self.false_rects = []
        self.num_tracked = len(self.trackers)
        faces = []
        if len(self.swaps) < self.num_tracked:
            self.swaps = self.swaps + [False for _ in range(len(self.swaps), self.num_tracked)]
            self.swappers = self.swappers + [None for _ in range(len(self.swappers), self.num_tracked)]
            self.groups = self.groups + [None for _ in range(len(self.groups), len(self.swaps))]

        for i, id in enumerate(self.groups):
            if i < len(self.swaps):
                self.swaps[i] = id in self.swap_groups_active
                if self.swaps[i]:
                    num_swap = self.swap_groups.index(id)
                    if self.swappers[i] is None and self.swaps[i]:
                        if self.swap_data[num_swap] == '' or not self.swap_data[num_swap].endswith('.jpg') and not self.swap_data[num_swap].endswith('.png'):
                            swap_file = self.swapPath
                        else:
                            swap_file = self.swap_data[num_swap]
                        self.swappers[i] = FaceSwapper(self.detector, swap_file)
                    elif self.swaps[i]:
                        if self.swappers[i].swap_path != self.swap_data[num_swap]:
                            self.swappers[i].update_swap_img(self.swap_data[num_swap])
                            self.swap_init_count += 1

        if change_last:
            self.last_bboxes = []
        if trackers_to_update is None:
            trackers_to_update = range(0, self.num_tracked + len(self.object_trackers))
        for i in trackers_to_update:
            try:
                ok, bbox = self.update_tracker(i, analyzed_frame, mask)
                self.update_count += 1
            except cv2.error:
                print("In opencv error handling")
                ok = False
                bbox = None
            except Exception as e:
                print(e.message)
                raise TrackerTypeChanged()
            if change_last:
                self.last_bboxes.append(bbox)

            # -------------------------------------------
            # 15_05_18
            # GET EMOTIONS BLOCK
            # -------------------------------------------
            # if ok and APPLY_EMO:
            if ok and self.num_tracked > i and self.emo_state: 
                
                if PRINT_EMOTIONS_ONLY:
                    enablePrint()

                print('\n{0}{0}\n<<< GOT FACE! ... INSIDE TRACKER >>>\nTHE TRACKER (~FACE) NUM: {1}\n'.format(PRINT_SEP_UNO, i))
                
                x1 = int(bbox[0])
                x2 = x1 + int(bbox[2])
                y1 = int(bbox[1])
                y2 = y1 + int(bbox[3])

                print('THE BOX: ({}, {}, {}, {})\n'.format(x1, x2, y1, y2))

                face = frame[y1:y2, x1:x2]

                # ---------------------------------
                # DEPRECATED:
                # ---------------------------------
                    # path_to_tmp_out = os.path.join('data', 'emotio', '000_{}.jpg'.format(self.update_count))
                    # print(path_to_tmp_out)
                    # cv2.imwrite(path_to_tmp_out, face)

                # WITHOUT GENDER:
                emo, time_str = get_emotion(face, get_time=True)
                print('THE EMOTION: {0}\n{1}'.format(emo, time_str))

                # if APPLY_GENDER: # WITH GENDER
                if self.emo_gen_state:
                    gen, time_str = get_gender(face, get_time=True)
                    print('THE GENDER: {0}\n{1}'.format(gen, time_str))

                # ------------------------------------
                if DB_TESTING_17_05_15:

                    # enablePrint()

                    print('\n{0}{0}\n<<< INSIDE TRACK... >>>\n'.format(PRINT_SEP_UNO))
                    print('\nself.trackers num, current idx: {}, {}\nself.groups: {}\n'.format(len(self.trackers), i, self.groups))

                    # DEPRECATED:
                        # tracker_current = self.trackers[i]
                        # if isinstance(tracker_current, cv2.TrackerBoosting):

                        #     tracker_id = tracker_current.__hash__() # is there a better technique?
                        #     print('TRACKER ID: {}'.format(tracker_id))

                        #     self.emo_count_dict[tracker_id][1][emo] = self.emo_count_dict[tracker_id][1].get(emo, 0) + 1
                        #     print(self.emo_count_dict)


                    gr_id_current = self.groups[i]
                    
                    self.emo_count_dict[gr_id_current][emo] = self.emo_count_dict[gr_id_current].get(emo, 0) + 1
                    print(self.emo_count_dict)

                    blockPrint()
                # ------------------------------------

                print('{0}{0}'.format(PRINT_SEP_UNO))


                if PRINT_EMOTIONS_ONLY:
                    blockPrint()
                if DRAW_EMOTIONS:
                # if self.emo_draw_state:

                    # ------------------------------
                    # DRAWING EMOTIONS:
                    # ------------------------------

                    if DRAW_EMOTIONS:

                        y_temp = y1 - int((y2-y1) * 0.3)
                        if y_temp < 0:
                            y_temp = 0

                        x_temp = x1 + int((x2-x1))
                        if x_temp >= displayFrame.shape[1]:
                            x_temp = displayFrame.shape[1]

                        cv2.rectangle(displayFrame, (x1, y_temp), (x_temp, y1), (255, 255, 255), -1)


                        pad_x, pad_y = int((x_temp-x1)*0.15), int((y1-y_temp)*0.3)
                        bottom_left = x1 + pad_x, y1 - pad_y

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.9
                        fontColor = (255,0,0)

                        text_thickness = 2

                        cv2.putText(displayFrame, emo, bottom_left, font, fontScale, fontColor, text_thickness)

                # -------------------------------------------

            if ok and self.num_tracked > i:
                # print "Starting update with PhotoSaver. Search is:", self.search
                if update_events:
                    saver = PhotoSaver(self.trackers, self.type, self.method, True, True, i, ((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]), frame),
                                       self.events, self.search, self.swaps, emo_dict=self.emo_count_dict) # 17_05_18

                    #saver.signals.photo_id.connect(self.add_id)
                    saver.setAutoDelete(True)
                    if self.gpu and self.encode_method == 'resnet':
                        saver.process_events(conn=None)
                    else:
                        self.save_pool.start(saver)

                    # ------------------------------------------
                    # if DB_WRITE_EMO_18_05_18:
                    #     saver.signals.emo_inserted.connect(self.refresh_emo_dict)

                    # if DB_WRITE_EMO_18_05_18:

                    #     self.emo_count_dict = saver.emo_dict
                    #     print('\nself.emo_count_dict: {}\n'.format(self.emo_count_dict))
                    # ------------------------------------------


                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                faces.append(bbox)

                # =====================================================
                if DEEPFAKES_MODEL_LOAD_OPT == 'queue':

                    x1 = p1[0] - 20 if p1[0] >= 20 else 0
                    y1 = p1[1] - 20 if p1[1] >= 20 else 0
                    x2 = p2[0] + 20 if p2[0] < frame.shape[1] - 20 else frame.shape[1]
                    y2 = p2[1] + 20 if p2[1] < frame.shape[0] - 20 else frame.shape[0]


                    print('\nFRAME (track): {}'.format(frame.shape))
                    print('FACES: {}\n'.format((x1,y1,x2,y2)))
            
                    face = frame[y1:y2, x1:x2]
                    deepf_queues[0].put(cv2.flip(face, 1))
                    print('PUT A FRAME (with flip)!')
                
                    while 1:
                        try:
                            image = deepf_queues[1].get()
                            image = cv2.flip(image, 1)
                            print('GOT AN IMAGE ... get_faces!')
                            
                            path = './data/deepf_face_{}.png'.format(self.frame_count)
                            cv2.imwrite(path, image)

                            if face.shape == image.shape:
                                displayFrame[y1:y2, x1:x2] = image
                                path = './data/deepf_frame_{}.png'.format(self.frame_count)
                                cv2.imwrite(path, displayFrame)
                            else:
                                print(face.shape, image.shape)

                            break
                        except:
                            pass
                # =====================================================

                if self.swaps[i]:
                    try:
                        tmp1 = (
                            p1[0] - 20 if p1[0] >= 20 else 0,
                            p1[1] - 20 if p1[1] >= 20 else 0
                        )
                        tmp2 = (
                            p2[0] + 20 if p2[0] < frame.shape[1] - 20 else frame.shape[1],
                            p2[1] + 20 if p2[1] < frame.shape[0] - 20 else frame.shape[0]
                        )
                        # displayFrame[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])] = \
                        #     self.swappers[i].face_swap(displayFrame[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])], None)
                        displayFrame[tmp1[1]:tmp2[1], tmp1[0]:tmp2[0]] = \
                            self.swappers[i].face_swap(displayFrame[tmp1[1]:tmp2[1], tmp1[0]:tmp2[0]], None, True)
                        self.swap_count += 1
                    except NoFaces:
                        pass
                elif self.parse_face:
                    x1 = int(bbox[0])
                    x2 = x1 + int(bbox[2])
                    y1 = int(bbox[1])
                    y2 = y1 + int(bbox[3])
                    unit = int(max(bbox[2], bbox[3]))
                    y_1 = max(0, y1 - unit / 2)
                    x_1 = max(0, x1 - unit / 2)
                    x_2 = min(frame.shape[1] - 1, x2 + unit / 2)
                    y_2 = min(frame.shape[0] - 1, y2 + unit / 2)
                    displayFrame[y_1:y_2, x_1:x_2] = get_parsing(displayFrame[y_1:y_2, x_1:x_2], bbox)
                if self.draw_rects:
                    cv2.rectangle(displayFrame, p1, p2, (0, 255, 0), displayFrame.shape[1] / 100, 1)

            elif ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                if self.draw_rects:
                    cv2.rectangle(displayFrame, p1, p2, (0, 0, 255), displayFrame.shape[1] / 100, 1)
            else:
                try:
                    if i < self.num_tracked:
                        self.trackers[i] = None
                        self.groups[i] = None
                        self.groups = [id for id in self.groups if id is not None]
                    else:
                        self.object_trackers[i - self.num_tracked] = None
                    if bbox is not None:
                        self.false_rects.append(bbox)
                except:
                    self.groups = [id for id in self.groups if id is not None]
                    if bbox is not None:
                        self.false_rects.append(bbox)

        self.trackers = [tr for tr in self.trackers if tr is not None]
        self.object_trackers = [tr for tr in self.object_trackers if tr is not None]
        if display:
            self.emit(SIGNAL("signal(PyQt_PyObject)"), displayFrame)
            # self.emit(SIGNAL("signal(PyQt_PyObject)"), analyzed_frame)
        return faces, displayFrame

    def update_tracker(self, num_tracker, frame, mask):
        ok, bbox = self.update(num_tracker, frame)
        #If face is inside frame, not on the edge, return True if area of the face is big enough
        if ok:
            f1 = bbox[0] > 0 and bbox[1] > 0 and bbox[0]+bbox[2]/2 < frame.shape[1] and bbox[1]+bbox[3]/2 < frame.shape[0]
            # f2 = not self.check_rect((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
            # if f1 and f2:
            #     self.new_faces = list(set(self.new_faces) - set([bbox]))
            self.min_area = int(min(self.min_area, bbox[2] * bbox[3]))
            if f1: # or f2: # if face is entering frame, not deleting tracker
                return bbox[2]*bbox[3] > self.min_area * 2, bbox
            # if not f1 and not f2:
            #     self.old_faces.append(bbox)
            #     return bbox[2]*bbox[3] > self.min_area * 2, bbox
            face_mask = np.zeros((mask.shape[0], mask.shape[1]))
            face_mask[int(bbox[0]):(int(bbox[0])+int(bbox[2])), int(bbox[1]):(int(bbox[1])+int(bbox[3]))] = 1
            mask = cv2.dilate(mask, np.ones((2, 2)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((int(mask.shape[1] / 10), int(mask.shape[0] / 10))))
            res = np.multiply(face_mask, mask)
            #Face is said to be on frame if it's area is big enough
            s = np.sum(np.sum(res, 1), 0)
            return s >= self.min_area, bbox
        else:
            return False, None

    def update(self, i, frame):
        self.num_tracked = len(self.trackers)
        if i < self.num_tracked:
            tracker = self.trackers[i]
        else:
            tracker = self.object_trackers[i - self.num_tracked]
        if tracker is not None:
            if self.track_op in ["MIL", "Boosting"]:
                ok, bbox = tracker.update(frame)
            elif self.track_op == "dlib":
                tracker.update(frame)
                rect = tracker.get_position()
                bbox = (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())
                ok = True
            else:
                print "Tracker type is not provided"
                return False, None
            return ok, tuple([int(b) for b in bbox])
        else:
            return False, None

    def find_face(self, rect, frame):
        x1, y1, x2, y2 = rect
        p1, q1 = x1, y1
        p2, q2 = x2 - x1, y2 - y1
        b = np.sqrt(frame.shape[0] * frame.shape[1]) / 5
        for i, bbox in enumerate(self.last_bboxes):
            if i == len(self.trackers):
                break
            if bbox is not None:
                s = abs(q1 - bbox[1])
                s += abs(p1 - bbox[0])
                s += abs(p2 - bbox[2])
                s += abs(q2 - bbox[3])
                if s < b:  # approximate
                    return True, i
                if p1 < bbox[0] and q1 < bbox[1] and p1+p2 > bbox[0]+bbox[2] and q1+q2 > bbox[1]+bbox[3] \
                    or p1 > bbox[0] and q1 > bbox[1] and p1+p2 < bbox[0]+bbox[2] and q1+q2 < bbox[1]+bbox[3]:
                    return True, i
        return False, -1

    def find_object(self, rect, frame):
        x1, y1, x2, y2 = rect
        p1, q1 = x1, y1
        p2, q2 = x2 - x1, y2 - y1
        b = np.sqrt(frame.shape[0] * frame.shape[1]) / 5
        print("b in find_object", b)
        self.num_tracked = len(self.trackers)
        for i, bbox in enumerate(self.last_bboxes):
            if i < self.num_tracked:
                continue
            if bbox is not None:
                s = abs(q1 - bbox[1])
                s += abs(p1 - bbox[0])
                s += abs(p1+p2 - bbox[0]-bbox[2])
                s += abs(q1+q2 - bbox[1]-bbox[3])
                if s < b:  # approximate
                    return True, i
                if p1 < bbox[0] and q1 < bbox[1] and p1+p2 > bbox[0]+bbox[2] and q1+q2 > bbox[1]+bbox[3] \
                    or p1 > bbox[0] and q1 > bbox[1] and p1+p2 < bbox[0]+bbox[2] and q1+q2 < bbox[1]+bbox[3]:
                    return True, i
        return False, -1

    def check_rect(self, rect, new=True):
        x1, y1, x2, y2 = rect
        p1, q1 = x1, y1
        p2, q2 = x2 - x1, y2 - y1
        if new:
            for i, bbox in enumerate(self.new_faces):
                s = abs(q1 - bbox[1])
                s += abs(p1 - bbox[0])
                s += abs(p2 - bbox[2] + bbox[0])
                s += abs(q2 - bbox[3] + bbox[1])
                if s < 100:
                    self.new_faces[i] = (bbox[0], bbox[1], bbox[2], bbox[3])
                    return False
        else:
            for i, bbox in enumerate(self.old_faces):
                s = abs(q1 - bbox[1])
                s += abs(p1 - bbox[0])
                s += abs(p2 - bbox[2])
                s += abs(q2 - bbox[3])
                if s < 100:
                    self.old_faces[i] = (bbox[0], bbox[1], bbox[2], bbox[3])
                    return False
        return True

    # @profile
    def count_faces2(self, mask, faces=[], objects=None):
        min_area = max(self.min_area, mask.shape[1] * mask.shape[0] / 200)
        # Move to self.ker
        ker = np.ones((5, 5))
        mask = cv2.dilate(mask, ker)
        # Move to self.a
        a = np.linspace(0.5, 1, num=2000)
        size = (int(a[mask.shape[0]] * 50 if mask.shape[0] < 2000 else 50), int(a[mask.shape[1]] * 50 if mask.shape[1] < 2000 else 50))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(size))
        if len(faces) > 0:
            for x, y, w, h in faces:
                mask[y:y+h, x:x+w] = 0
        if objects is not None:
            for res in objects.items():
                for x, y, w, h in res[1]:
                    mask[y:y + h, x:x + w] = 0
        (_, contours, hier) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cent_rad = []
        # distances = []
        new_contours = []
        if hier is not None:
            hier_ = hier[0]
            for i in range(0, len(contours)):
                if hier_[i][3] == -1:
                    new_contours.append(contours[i])
        from math import pi, sqrt
        for cnt in new_contours:
            M = cv2.moments(cnt)
            m = M['m00']
            # if m < min_area:
            #     msk = np.zeros(mask.shape)
            #     cv2.drawContours()
            cx = M['m10'] / m
            cy = M['m01'] / m
            r = sqrt(m / pi)
            cent_rad.append((cx, cy, r))
        labels = np.arange(0, len(new_contours), 1).astype(np.uint16)
        for i in range(len(new_contours)):
            for j in range(i+1, len(new_contours)):
                cnt_r_1 = cent_rad[i]
                cnt_r_2 = cent_rad[j]
                d = sqrt((cnt_r_1[0] - cnt_r_2[0])*(cnt_r_1[0] - cnt_r_2[0]) + (cnt_r_1[1] - cnt_r_2[1])*(cnt_r_1[1] - cnt_r_2[1]))
                D = d - cnt_r_1[2] - cnt_r_2[2]
                if D < mask.shape[0] / 5:
                    labels = [
                        (labels[i] if label == labels[j] else label)
                        for label in labels
                    ]
                    # labels[j] = labels[i]     - can still separate some close regions
        rects = []
        labels = np.array(labels)
        min_rect_area = mask.shape[1] * mask.shape[0] / 100
        if labels.shape[0] > 1:
            for label in np.unique(labels):
                ind = [i for i, lab in enumerate(labels) if lab == label]
                rect = cv2.boundingRect(np.concatenate([new_contours[i] for i in ind]))
                if rect[2] * rect[3] > min_rect_area:
                    rects.append(rect)
        elif labels.shape[0] == 1:
            rects.append(cv2.boundingRect(new_contours[0]))
        first_rects = self.process_rects(rects)
        final_rects = self.process_rects(first_rects)
        num = 0
        # if hier is not None:
        #     hier = hier[0]
        #     for i in range(0, len(contours)):
        #         if hier[i][3] == -1 and cv2.contourArea(contours[i]) > min_area:
        #             num += 1
        #self.emit(SIGNAL("signal(PyQt_PyObject)"), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        return num, final_rects, rects

    def process_rects(self, rects):
        ln = len(rects)
        final_rects = []
        if ln > 1:
            outers = []
            inners = []
            for i in range(ln):
                inners_ = []
                x1, y1, w1, h1 = rects[i]
                max_x = x1 + w1
                max_y = y1 + h1
                min_x = x1
                min_y = y1
                for j in range(ln):
                    if i == j:
                        continue
                    x2, y2, w2, h2 = rects[j]
                    f1 = x1 < x2 < x1 + w1
                    f3 = x1 < x2 + w2 < x1 + w1
                    f2 = y1 < y2 < y1 + h1
                    f4 = y1 < y2 + h2 < y1 + h1
                    if j not in inners and j not in outers:
                        if not (f1 and f3 or f2 and f4):
                            if f1 and f2:
                                inners_.append(j)
                                max_x = max(max_x, x2 + w2)
                                max_y = max(max_y, y2 + h2)
                            elif f3 and f4:
                                inners_.append(j)
                                min_x = min(min_x, x2)
                                min_y = min(min_y, y2)
                            elif f1 and f4:
                                inners_.append(j)
                                max_x = max(max_x, x2 + w2)
                                min_y = min(min_y, y2)
                            elif f2 and f3:
                                inners_.append(j)
                                min_x = min(min_x, x2)
                                max_y = max(max_y, y2 + h2)
                        else:
                            if f1 and f3:
                                if f2:
                                    inners_.append(j)
                                    max_y = max(max_y, y2 + h2)
                                elif f4:
                                    inners_.append(j)
                                    min_y = min(min_y, y2)
                            else:
                                if f1:
                                    inners_.append(j)
                                    max_x = max(max_x, x2 + w2)
                                elif f3:
                                    inners_.append(j)
                                    min_x = min(min_x, x2)
                if len(inners_) > 0:
                    outers.append(i)
                    final_rects.append((min_x, min_y, max_x - min_x, max_y - min_y))
                    [inners.append(j) for j in inners_]
            for i in range(ln):
                if i not in outers and i not in inners:
                    final_rects.append(rects[i])
        else:
            final_rects = rects
        return final_rects

    def unite_close_rects(self, rects, pixel_thresh):
        ln = len(rects)
        if ln > 1:
            labels = np.arange(0, ln, 1)
            for i in range(ln):
                x, y, w, h = rects[i]
                for j in range(ln):
                    if i == j:
                        continue
                    x1, y1, w1, h1 = rects[j]
                    united = False
                    if 0.75 < w * 1.0 / w1 < 1.33:
                        f1 = abs(y1 - (y+h)) < pixel_thresh or abs(y - (y1+h1)) < pixel_thresh
                        f2 = abs(x - x1) < min(w, w1) / 4
                        f3 = abs(x+w - (x1+w1)) < min(w, w1) / 4
                        if f1 and f2 and f3:
                            labels = [
                                (labels[i] if label == labels[j] else label)
                                for label in labels
                            ]
                            united = True
                    if not united and 0.75 < h * 1.0 / h1 < 1.33:
                        f1 = abs(x1 - (x+w)) < pixel_thresh or abs(x - (x1+w1)) < pixel_thresh
                        f2 = abs(y - y1) < min(h, h1) / 4
                        f3 = abs(y - y1 + h - h1) < min(h, h1) / 4
                        if f1 and f2 and f3:
                            labels = [
                                (labels[i] if label == labels[j] else label)
                                for label in labels
                            ]
            new_rects = []
            labels = np.array(labels)
            if labels.shape[0] > 1:
                for label in np.unique(labels):
                    ind = [i for i, lab in enumerate(labels) if lab == label]
                    if len(ind) == 1:
                        new_rects.append(rects[ind[0]])
                    else:
                        min_x, min_y, w, h = rects[ind[0]]
                        max_x = min_x + w
                        max_y = min_y + h
                        for i in ind[1:]:
                            x, y, w, h = rects[i]
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x+w)
                            max_y = max(max_y, y+h)
                        new_rects.append((min_x, min_y, max_x - min_x, max_y - min_y))
            elif labels.shape[0] == 1:
                min_x, min_y, w, h = rects[0]
                max_x = min_x + w
                max_y = min_y + h
                for x, y, w, h in rects[1:]:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
                new_rects.append((min_x, min_y, max_x - min_x, max_y - min_y))
            return new_rects
        else:
            return rects

    def watershed(self, frame, mask):
        tmp_frame = frame.copy()
        ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv2.watershed(tmp_frame, markers)
        tmp_frame[markers == -1] = [255, 0, 0]
        return tmp_frame

    def count_faces(self, mask, frame):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((int(mask.shape[1]/10), int(mask.shape[0]/10))))
        new_mask = mask
        #mask = cv2.erode(mask, np.ones((int(mask.shape[0]/200), int(mask.shape[0]/200))), iterations=2)

        #mask = cv2.dilate(mask, np.ones((int(mask.shape[0]/20), int(mask.shape[0]/20))), iterations=2)
        bboxes = []
        for i in range(0, self.num_tracked):
            ok, bbox = self.update(i, frame)
            if ok:
                bboxes.append(bbox)
                newx = int(max(bbox[0] - 2*bbox[2], 0))
                newy = bbox[1] + bbox[3]
                neww = min(bbox[2]*5, mask.shape[1] - newx)
                newh = min(bbox[3] * 8, mask.shape[0] - newy)
                nbox = (newx, newy, neww, newh)
                #Negating zone under face
                new_mask[int(nbox[1]):(int(nbox[1]) + int(nbox[3])), int(nbox[0]):(int(nbox[0]) + int(nbox[2]))] = 0

        for bbox in bboxes:
            #Unconditional highlighting of moving tracked faces
            new_mask[int(bbox[1]):(int(bbox[1]) + int(bbox[3])), int(bbox[0]):(int(bbox[0]) + int(bbox[2]))] = \
                mask[int(bbox[1]):(int(bbox[1]) + int(bbox[3])), int(bbox[0]):(int(bbox[0]) + int(bbox[2]))]

        mask = new_mask

        (_, contours, hier) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num = 0
        if hier is not None:
            hier = hier[0]
            for i in range(0, len(contours)):
                if hier[i][3] == -1 and cv2.contourArea(contours[i]) > self.min_area:
                   num += 1
        #self.emit(SIGNAL("signal(PyQt_PyObject)"), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        return num

    def detect(self, frame, bboxes):
        # frame should be grayscale 2D image

        detect_face = []
        if self.type in ["dlib", "cnn"]:
            for x, y, w, h in bboxes:
                if self.gpu:
                    analyzed_frame = frame[y:y+h, x:x+w, :]
                    dlib_cuda_lock.acquire()
                else:
                    analyzed_frame = frame[y:y + h, x:x + w]
                dets = self.classifiersList[0](analyzed_frame, 1)

                if self.gpu:
                    dlib_cuda_lock.release()
                for i, d in enumerate(dets):
                    if self.type in ["cnn"]:
                        d = d.rect
                    detect_face.append((d.left()+x, d.top()+y, d.right()+x, d.bottom()+y))
        elif self.type == "cascade":
            if self.gpu:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, w, h in bboxes:
                detect_faces = []
                analyzed_frame = frame[y:y+h, x:x+w]
                side_cascade = self.classifiersList[0]
                front_cascade = self.classifiersList[1]
                gray1 = cv2.flip(analyzed_frame, 1)

                faces = side_cascade.detectMultiScale(gray1, 1.3, 5)
                faces1 = side_cascade.detectMultiScale(analyzed_frame, 1.3, 5)
                fronts = front_cascade.detectMultiScale(analyzed_frame, 1.3, 5)

                detect_faces += get_face(faces1)
                detect_faces += get_face(fronts)
                height, width = gray1.shape
                detect_faces += get_face(faces, width, True)

                list_faces1 = get_face(faces1)
                list_frontes = get_face(fronts)
                height, width = gray1.shape
                list_fases = get_face(faces, width, True)
                detect_faces = list_fases + list_faces1

                if len(detect_faces) > 0:
                    for (x1, y1, x2, y2) in list_frontes:
                        bFrontFace = False
                        for (p1, q1, p2, q2) in list_fases:
                            if x2 < p1 or x1 > p2:
                                detect_faces.append((x1, y1, x2, y2))
                                bFrontFace = True
                                break

                        if bFrontFace == False:
                            for (p1, q1, p2, q2) in list_faces1:
                                if x2 < p1 or x1 > p2:
                                    detect_faces.append((x1, y1, x2, y2))
                                    break
                else:
                    detect_faces = list_frontes
                for x1, y1, w1, h1 in detect_faces:
                    detect_face.append((x1 + x, y1 + y, w1 + x, h1 + y))
        return detect_face

    # @profile
    def get_faces(self, frame_origin, classifiers=None, track=False, start=False, old_track='Boosting', display=True,
                  people_bboxes=None, objects_bboxes=None, faces=None, gray=None, new_frame=False):


        # if DB_TESTING_17_05_15 and self.frame_count != 2:
        if DB_WRITE_EMO_18_05_18 and not start:
            # enablePrint() 
            self.emo_count_dict = OrderedDict((key, dict()) for key in self.emo_count_dict) 
            print('*********** EMO_COUNT_DICT WAS REFRESHED! ***********')
            blockPrint()

        frame = frame_origin.copy()
        displayFrame = frame.copy()

        h, w = frame.shape[:2]
        
        # ----------------------------------------
        # 11_05_18 - block for testing purposes
        # ----------------------------------------
        TESSERACT_WITHOUT_YOLO = 0
        if TESSERACT_WITHOUT_YOLO:
            print('<<< GOT A CAR >>>') # 08_05_18
            car = frame

            plate_bboxes = []

            # plate detection:
            plate_bbox = self.get_plate(car)
            print("Plate detected: {0}".format(plate_bbox))
           
            detected_fname = "{}_{}".format(self.plate_algo, self.frame_counter)

            if 1: # FLAG FOR TESTS
                
                # plate text recognition:
                txt = get_tesseract_results(car, plate_bbox, show_frames=False, expand_area=None, fname_out=detected_fname)

                if txt:
                    tmp_path = os.path.join("data/tess_out", 'car_' + detected_fname + '.png')
                    cv2.imwrite(tmp_path, car)

                for box in plate_bbox:
                    box = tuple(int(item) for item in box)
                    box_abs_coords = tuple(item + 0 if not idx % 2 else item + 0 for idx, item in enumerate(box))                                   
                    plate_bboxes.append(box_abs_coords)

                color, width = (255, 255, 150), 2
                print('*******************************')
                print('PLATE BBOXES: {}'.format(plate_bboxes))
                print('*******************************')
                for box_abs_coords in plate_bboxes:
                    x1,y1,x2,y2,x3,y3,x4,y4 = box_abs_coords

                    cv2.line(displayFrame, (x1,y1), (x2,y2), color, width)
                    cv2.line(displayFrame, (x2,y2), (x3,y3), color, width)
                    cv2.line(displayFrame, (x3,y3), (x4,y4), color, width)
                    cv2.line(displayFrame, (x1,y1), (x4,y4), color, width)
        # ----------------------------------------

        trackers_to_draw = []
        obj_trackers_to_draw = []

        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
        gray3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if faces is None:
            if classifiers is None:
                self.emit(SIGNAL("PySigNoClassifier"))
                return [], None, None, trackers_to_draw, obj_trackers_to_draw

            # prof = Profiler()
            # prof.start()

            detect_face = []

            if self.type in ["dlib", "cnn"]:
                detector = classifiers[0]
                if self.gpu:
                    dlib_cuda_lock.acquire()
                    dets = detector(gray3d, 1)
                    dlib_cuda_lock.release()
                else:
                    dets = detector(gray, 1)
                for i, d in enumerate(dets):
                    if self.type in ["cnn"]:
                        d = d.rect
                    detect_face.append((d.left(), d.top(), d.right(), d.bottom()))

            elif self.type == "cascade":
                side_cascade = classifiers[0]
                front_cascade = classifiers[1]
                gray1 = cv2.flip(gray, 1)

                faces = side_cascade.detectMultiScale(gray1, 1.3, 5)
                faces1 = side_cascade.detectMultiScale(gray, 1.3, 5)
                fronts = front_cascade.detectMultiScale(gray, 1.3, 5)

                detect_face += get_face(faces1)
                detect_face += get_face(fronts)
                height, width, channels = displayFrame.shape
                detect_face += get_face(faces, width, True)

                list_faces1 = get_face(faces1)
                list_frontes = get_face(fronts)
                height, width, channels = displayFrame.shape
                list_fases = get_face(faces, width, True)
                detect_face = list_fases + list_faces1

                if len(detect_face) > 0:
                    for (x1, y1, x2, y2) in list_frontes:
                        bFrontFace = False
                        for (p1, q1, p2, q2) in list_fases:
                            if x2 < p1 or x1 > p2:
                                detect_face.append((x1, y1, x2, y2))
                                bFrontFace = True
                                break

                        if bFrontFace == False:
                            for (p1, q1, p2, q2) in list_faces1:
                                if x2 < p1 or x1 > p2:
                                    detect_face.append((x1, y1, x2, y2))
                                    break
                else:
                    detect_face = list_frontes
        else:
            detect_face = faces

        need_update = False
        if new_frame:
            self.trackers = []

        if len(detect_face) > 0:
            included_trackers = [] # INCLUDE TRACKS
            new_bboxes = []
            new_trackers = [None for face in detect_face]
            new_events = [None for face in detect_face]
            new_ids = [None for face in detect_face]
            self.new_swaps = []
            if self.events is None:
                self.events = [None for face in detect_face]
            if self.trackers is None or len(self.trackers) == 0 or old_track != self.track_op:
                self.trackers = [None for _ in detect_face]

                # ------------------------------------
                if 0:
                    print('{0}\nTRACKERS NUM: {1}\n{0}\n'.format(PRINT_SEP_UNO, len(self.trackers)))
                # ------------------------------------
                    
            self.ids = []
            need_update = False
            if len(self.swaps) < len(detect_face):
                self.swaps = self.swaps + [False for _ in range(len(self.swaps), len(detect_face))]
                self.groups = self.groups + [None for _ in range(len(self.groups), len(self.swaps))]
            self.swappers = self.swappers + [None for _ in range(len(self.swappers), len(self.swaps))]
            
            for i, id in enumerate(self.groups):
                if i < len(self.swaps):
                    self.swaps[i] = id in self.swap_groups

                    if self.swaps[i]:
                        num_swap = self.swap_groups.index(id)

                        if self.swappers[i] is None and self.swaps[i]:
                            if self.swap_data[num_swap] == '' or not self.swap_data[num_swap].endswith('.jpg') and not self.swap_data[num_swap].endswith('.png'):
                                swap_file = self.swapPath
                            else:
                                swap_file = self.swap_data[num_swap]

                            self.swappers[i] = FaceSwapper(self.detector, swap_file)
                            self.swap_init_count += 1
                        elif self.swaps[i]:
                            if self.swappers[i].swap_path != self.swap_data[num_swap]:
                                self.swappers[i].update_swap_img(self.swap_data[num_swap])
                                self.swap_init_count += 1

            nonified_faces = []
            self.disable_tracking = False
            self.new_faces = []
            on_frame_list = []

            gray3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            print(start, new_frame)


            # =====================================================
            if DEEPFAKES_MODEL_LOAD_OPT == 'queue':
                
                print('\nFRAME: {}'.format(frame.shape))
                print('FACES: {}\n'.format(detect_face))
            
                # print(frame.shape, frame.dtype) # (360, 640, 3), dtype('uint8')

                # -------------------------------------    
                # deepf_queues[0].put(cv2.flip(frame, 1)) # for the case when we apply deepfakes on a full frame
                
                if detect_face:
                    x1, y1, x2, y2 = detect_face[0]
                face = frame[y1:y2, x1:x2]
                deepf_queues[0].put(cv2.flip(face, 1))
                # -------------------------------------

                print('PUT A FRAME (with flip)!')
                
                while 1:
                    try:
                        image = deepf_queues[1].get()
                        print('GOT AN IMAGE ... get_faces!')
                        # print(type(image))
                        # print(image.shape)
                        
                        image = cv2.flip(image, 1)

                        path = './data/deepf_face_{}.png'.format(self.frame_count)
                        cv2.imwrite(path, image)

                        # displayFrame = image # for the case when we apply deepfakes on a full frame

                        if face.shape == image.shape:
                            displayFrame[y1:y2, x1:x2] = image
                            path = './data/deepf_frame_{}.png'.format(self.frame_count)
                            cv2.imwrite(path, displayFrame)
                        else:
                            print(face.shape, image.shape)

                        break
                    except:
                        pass


            # if DEEPFAKES_MODEL_LOAD_OPT == 'inside':
            #     model = run_func_inside_dir(lib_dir, faceit.get_model)
            #     converter = faceit.get_converter(model)

            # if DEEPFAKES_MODEL_LOAD_OPT == 'self':
            #     image = _convert_frame(frame, converter=self.deepf_converter, change_order=False)
            # elif DEEPFAKES_MODEL_LOAD_OPT == 4:
            #     image = deepf_instance.convert_frame(frame)
            # else:
            #     image = _convert_frame(frame, converter=converter, change_order=False)

            # =====================================================

            print('DETECTED FACES: {}'.format(detect_face)) # 14_05_18
            for i, (x1, y1, x2, y2) in enumerate(detect_face):

                # if APPLY_EMO:
                if self.emo_state: 

                    # ------------------------
                    # GET EMOTIONS BLOCK

                    # 14_05_18- UNABLE TO GET OUTPUT FROM get_emotion HERE:
                    # ValueError: Tensor Tensor("predictions/Softmax:0", shape=(?, 7), dtype=float32) is not an element of this graph.
                    # ------------------------
                    
                    if PRINT_EMOTIONS_ONLY:
                        enablePrint()

                    print('\n{0}{0}\n<<< GOT FACE! ... INSIDE GET FACES >>>\nTHE FACE NUM: {1}\n'.format(PRINT_SEP_DUO, i))

                    # print(x1, y1, x2, y2) # (153L, 24L, 282L, 153L)
                    x1, y1, x2, y2 = [int(item) for item in (x1, y1, x2, y2)]
                    print('THE BOX: ({}, {}, {}, {})\n'.format(x1, x2, y1, y2))

                    face = frame[y1:y2, x1:x2].astype('uint8')

                    # print(type(face))
                    # print(face.dtype)
                    # print(face.shape)

                    # WITHOUT GENDER:
                    emo, time_str = get_emotion(face, get_time=True)
                    print('THE EMOTION: {0}\n{1}'.format(emo, time_str))

                    # if APPLY_GENDER: # WITH GENDER
                    if self.emo_gen_state:
                        gen, time_str = get_gender(face, get_time=True)
                        print('THE GENDER: {0}\n{1}'.format(gen, time_str))

                    print('{0}{0}'.format(PRINT_SEP_DUO))

                    if PRINT_EMOTIONS_ONLY:
                        blockPrint()


                    # sys.exit(0) # for tests

                    # ------------------------------
                    # DRAWING EMOTIONS:
                    # ------------------------------
                    # assume drawing only with tracker

                    # ------------------------------
                # blockPrint() # 17_05_18

                if self.gpu:
                    getter = GroupGetter(i, (x1, y1, x2, y2), frame, self.method, (self.faiss_recog, match_template),
                                         len(detect_face) == 1 and self.cap is None, self.type, self.align, gray3d)
                else:
                    getter = GroupGetter(i, (x1, y1, x2, y2), frame, self.method, (self.faiss_recog, match_template),
                                         len(detect_face) == 1 and self.cap is None, self.type, self.align, gray)

                getter.signal.groups.connect(self.add_group)

                # ----------------------------
                if DB_TESTING_17_05_15:
                    parent_conn, child_conn = Pipe()
                    getter.get(child_conn)
                    g_id = parent_conn.recv()
                    self.add_group((g_id, i))

                else:
                    if self.gpu and self.encode_method == "resnet":
                        parent_conn, child_conn = Pipe()
                        getter.get(child_conn)
                        g_id = parent_conn.recv()
                        self.add_group((g_id, i))
                    else:
                        self.group_pool.start(getter)
                # ----------------------------

                if start:
                    on_frame, person = False, -1
                else:
                    on_frame, person = self.find_face((x1, y1, x2, y2), frame)
                if not on_frame:
                    need_update = True

                on_frame_list.append(on_frame)

                if self.swaps[i] and on_frame:
                    tmp1 = (
                        x1 - 20 if x1 >= 20 else 0,
                        y1 - 20 if y1 >= 20 else 0
                    )
                    tmp2 = (
                        x2 + 20 if x2 < frame.shape[1] - 20 else frame.shape[1],
                        y2 + 20 if y2 < frame.shape[0] - 20 else frame.shape[0]
                    )
                    # displayFrame[y1:y2, x1:x2] = self.swappers[i].face_swap(displayFrame[y1:y2, x1:x2], None)
                    displayFrame[tmp1[1]:tmp2[1], tmp1[0]:tmp2[0]] = \
                        self.swappers[i].face_swap(displayFrame[tmp1[1]:tmp2[1], tmp1[0]:tmp2[0]], None, detect=True)
                    self.swap_count += 1
                    self.new_swaps.append(True)
                else:
                    self.new_swaps.append(False)

                if not self.swaps[i] and on_frame and self.parse_face:
                    unit = int(max(x2-x1, y2-y1))
                    y_1 = max(0, y1 - unit / 2)
                    x_1 = max(0, x1 - unit / 2)
                    x_2 = min(frame.shape[1] - 1, x2 + unit / 2)
                    y_2 = min(frame.shape[0] - 1, y2 + unit / 2)
                    displayFrame[y_1:y_2, x_1:x_2] = get_parsing(displayFrame[y_1:y_2, x_1:x_2], bbox)

                if self.draw_rects:
                    cv2.rectangle(displayFrame, (x1, y1), (x2, y2), (0, 255, 0), thickness=displayFrame.shape[1] / 100)
               
                if track:

                    if x2 - x1 < 0 or y2 - y1 < 0 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                        print("\n\tFACE ON BORDER\n")
                        nonified_faces.append(i)
                        on_frame_list[i] = True

                    else:
                        if not on_frame or self.trackers[person] is None:
                            if people_bboxes is not None:
                                if len(people_bboxes) == len(detect_face):
                                    x1, y1, x2, y2 = people_bboxes[i]
                                    print("Got person! ", (x1, y1, x2, y2))
                                else:
                                    for bbox in people_bboxes:
                                        if self.check_person(bbox, (x1, y1, x2-x1, y2-y1), None, track=False):
                                            x1, y1, x2, y2 = bbox
                                            x2 += x1
                                            y2 += y1
                                            print("Got person! ", (x1, y1, x2, y2))
                                            break
                            x1 = 0 if x1 < 0 else x1
                            y1 = 0 if y1 < 0 else y1
                            x2 = frame.shape[1] - 1 if x2 >= frame.shape[1] else x2
                            y2 = frame.shape[0] - 1 if y2 >= frame.shape[0] else y2
                            if self.track_op == "Boosting":
                                tracker = cv2.TrackerBoosting_create()
                            else:
                                tracker = dlib.correlation_tracker()

                        if on_frame:
                            print(person, len(self.trackers), self.trackers[person])
                            if person < len(self.trackers) and self.trackers[person] is not None:
                                new_trackers[i] = self.trackers[person]
                                included_trackers.append(person)
                            else:
                                print("Assigning tracker")

                                self.tracker_init_count += 1
                                new_bboxes.append((x1, y1, x2-x1, y2-y1))
                                if self.track_op == "Boosting":
                                    tracker.init(gray3d, (x1, y1, x2 - x1, y2 - y1))
                                else:
                                    tracker.start_track(gray3d, dlib.rectangle(x1, y1, x2, y2))
                                new_trackers[i] = tracker
                            try:
                                new_events[i] = self.events[person]
                            except:
                                new_events[i] = -1
                        else:
                            print("Assigning tracker")

                            self.tracker_init_count += 1
                            new_bboxes.append((x1, y1, x2 - x1, y2 - y1))
                            if self.track_op == "Boosting":
                                tracker.init(gray3d, (x1, y1, x2 - x1, y2 - y1))
                            else:
                                tracker.start_track(gray3d, dlib.rectangle(x1, y1, x2, y2))
                            new_trackers[i] = tracker
                            new_events[i] = -1

                        if DB_TESTING_17_05_15:
                            
                            enablePrint()

                            print('\n{0}{0}\n<<< INSIDE GET FACES ... >>>\n'.format(PRINT_SEP_DUO))
                            print('\nself.trackers num: {}\n'.format(len(self.trackers)))


                            # DEPRECATED:
                                # --------------------------------
                                # parent_conn, child_conn = Pipe()
                                # getter.get(child_conn)
                                # gr_id = parent_conn.recv()

                                # print('--------- GOT GROUP_ID: {} -------------'.format(gr_id))
                                # --------------------------------

                                # if isinstance(tracker, cv2.TrackerBoosting):
                                    
                                    # tracker_id = tracker.__hash__() # is there a better technique?
                                    # print('TRACKER ID: {}'.format(tracker_id))

                                    # self.emo_count_dict[tracker_id] = (gr_id, {emo: 1})       
                                    # print(self.emo_count_dict)

                            for idx in self.groups: # LITTLE CRUTCH
                                if not idx in self.emo_count_dict:
                                    self.emo_count_dict[idx] = dict()

                            gr_id = next(i for i in reversed(self.groups) if i is not None)
                            print('GROUP_IDX: {}'.format(gr_id))
                            self.emo_count_dict[gr_id] = {emo: 1}

                            print(self.emo_count_dict)

                            blockPrint()

                            # --------------------------------
                            if 0:

                                full_info = self.update_info()

                                print('\n{0}{0}'.format(PRINT_SEP_DUO))
                                for entry in full_info:
                                    groupid, _, last_seen, fst_name, lst_name, _, _, _, _ = entry
                                    print('\nGROUP ID: {0}\nNAME: "{1} {2}"\nLAST SEEN: {3}'.format(groupid, fst_name, lst_name, last_seen))
                                print('\n{0}{0}\n'.format(PRINT_SEP_DUO))
                            # --------------------------------

            if self.gpu:
                saver = PhotoSaver(self.trackers, self.type, self.method, True, on_frame_list,
                                   list(range(len(detect_face))), (detect_face, frame),
                                   self.events, self.search, self.swaps, len(detect_face) == 1 and self.cap is None,
                                   gray3d, emo_dict=self.emo_count_dict) # 17_05_18
            else:
                saver = PhotoSaver(self.trackers, self.type, self.method, True, on_frame_list,
                                   list(range(len(detect_face))), (detect_face, frame),
                                   self.events, self.search, self.swaps, len(detect_face) == 1 and self.cap is None,
                                   gray, emo_dict=self.emo_count_dict) # 17_05_18
            saver.signals.photo_id.connect(self.add_id)
            saver.signals.events.connect(self.update_events)

            # saver.signals.trackers.connect(self.update_swaps)
            saver.setAutoDelete(True)
            if self.gpu and self.encode_method == 'resnet':
                p, c = Pipe()
                try:
                    saver.process_events(c)
                    p_id, evs = p.recv()
                    self.update_events(evs)
                    for i, on_frame in enumerate(on_frame_list):
                        self.add_id(p_id[i])
                except IndexError:
                    pass
            else:
                self.save_pool.start(saver)

            # ----------------------------------------------
            # if DB_WRITE_EMO_18_05_18:
            #     saver.signals.emo_inserted.connect(self.refresh_emo_dict)
            
            # if DB_WRITE_EMO_18_05_18
            #     self.emo_count_dict = saver.emo_dict
            #     print('\nself.emo_count_dict: {}\n'.format(self.emo_count_dict))
            # ----------------------------------------------

            self.swaps = self.new_swaps
            indices = list(set(range(len(detect_face))) - set(nonified_faces))
            ln = len(self.trackers)
            self.trackers = [new_trackers[ind] for ind in indices] + [self.trackers[i] for i in (set(range(ln)) - set(included_trackers))]
            trackers_to_draw = list(set(range(ln)) - set(included_trackers))
            self.events = [new_events[ind] for ind in indices]
            self.last_bboxes[:self.num_tracked] = new_bboxes + [self.last_bboxes[i] for i in included_trackers]
            detect_face = [detect_face[ind] for ind in indices]
            self.num_tracked = len(self.trackers)


        # track = False # 14_05_18
        if track and objects_bboxes is not None:
            new_bboxes = self.last_bboxes[:self.num_tracked]
            new_trackers = []
            nonified_objs = []
            included_trackers = []

            plate_bboxes = [] # 11_05_18 - for drawing
            car_num = 0 # 10_05_18 ==> value for test output images
            print('OBJECTs BBOXES: {}'.format(objects_bboxes)) # 10_05_18
            
            for res in objects_bboxes.items():
                
                yolo_obj, yolo_boxes = res[0], res[1]
                is_car = yolo_obj == "car"

                if not yolo_obj == "person":

                    for box in yolo_boxes:

                        x1, y1, x2, y2 = box
                        out_of_frame_condition = any((not 0 <= x1 <= x2, not 0 <= y1 <= y2, x2 >= w, y2 >= h))

                        # -------------------------------------
                        print('\n------------------')
                        print('THE ORIGINAL BBOX:')
                        print((x1, y1, x2, y2), yolo_obj)
                        # -------------------------------------
                        
                        # print('\nFINDING OBJECTS ...') # 10_05_18
                        on_frame, object = self.find_object((x1, y1, x2, y2), frame)
                        # print('THE OBJECTS: {}, {}'.format(on_frame, object)) # 10_05_18
                        
                        if not is_car:

                            if out_of_frame_condition:

                                nonified_objs.append(i)
                                ok = False
                        
                            else:

                                if self.track_op == "Boosting":
                                    tracker = cv2.TrackerBoosting_create()
                                else:
                                    tracker = dlib.correlation_tracker()

                                ok = True

                        else: 

                            if out_of_frame_condition:

                                print('\nAPPLYING BOX ALIGNING FOR A CAR ...') # 10_05_18

                                x1_c = 0 if x1 < 0 else x1
                                y1_c = 0 if y1 < 0 else y1
                                x2_c = w - 1 if x2 >= w else x2
                                y2_c = h - 1 if y2 >= h else y2

                                # -------------------------------------
                                print('ALIGNED COORDINATES:')
                                print((x1_c, y1_c, x2_c, y2_c), yolo_obj)
                                # -------------------------------------

                            else:
                                x1_c, y1_c, x2_c, y2_c = x1, y1, x2, y2

                            if self.track_op == "Boosting":
                                tracker = cv2.TrackerBoosting_create()
                            else:
                                tracker = dlib.correlation_tracker()

                            ok = True
                            
                            # --------------------------------
                            APPLY_PLATE_DETECT = 1 # FLAG FOR TESTS
                            if APPLY_PLATE_DETECT:
                                
                                # filtering 'too small' car images:
                                MIN_RESOLUTION_Y, MIN_RESOLUTION_X = 40, 50 # MAGIC
                                if any((y2_c-y1_c < MIN_RESOLUTION_Y, x2_c-x1_c < MIN_RESOLUTION_X)):
                                    print('THE IMAGE IS TOO SMALL! SKIP IT.')

                                else:
                                    
                                    car = frame[y1_c:y2_c, x1_c:x2_c]
                                    
                                    sep = '------------------------------'
                                    print('<<< GOT A CAR >>>')
                                    print('{0}\nTHE CAR PLATE DETECT METHOD: {1}\n{0}'.format(sep, self.plate_algo))

                                    # ----------------------------
                                    # plate detection:
                                    # ----------------------------
                                    plate_bbox = self.get_plate(car)

                                    print('{0}\nPLATE BBOXES DETECTED: {1}\n{0}'.format(sep, plate_bbox))


                                    APPLY_TESSERACT = 1 # FLAG FOR TESTS
                                    if APPLY_TESSERACT: 
                                        
                                        detected_fname = "{}_{}_{}".format(self.plate_algo, self.frame_counter, car_num)

                                        # plate text recognition:
                                        txt = get_tesseract_results(car, plate_bbox, show_frames=False, expand_area=None, fname_out=detected_fname)

                                        if txt:
                                            car_num += 1 
                
                                            tmp_path = os.path.join("data/tess_out", 'car_' + detected_fname + '.png')
                                            cv2.imwrite(tmp_path, car)


                                    APPLY_DRAW_BOXES = 0 # FLAG FOR TESTS
                                    if APPLY_DRAW_BOXES:
                                        for box in plate_bbox:
                                            box = tuple(int(item) for item in box)
                                            box_abs_coords = tuple(item + x1 if not idx % 2 else item + y1 for idx, item in enumerate(box))
                                                                                
                                            plate_bboxes.append(box_abs_coords)
                        # -------------------------------------

                        # print('\nIS THE BBOX ON FRAME: {}\n'.format(ok)) # 10_05_18

                        if ok:
                            if on_frame:
                                if object > -1 and object < len(self.object_trackers):
                                    if self.object_trackers[object] is not None:
                                        if object not in included_trackers:
                                            new_trackers.append(self.object_trackers[object])
                                            included_trackers.append(object)
                                            new_bboxes.append((x1, y1, x2 - x1, y2 - y1))
                                    else:
                                        print("Assigning tracker")
                                        new_bboxes.append((x1, y1, x2-x1, y2-y1))
                                        if self.track_op == "Boosting":
                                            tracker.init(gray3d, (x1, y1, x2 - x1, y2 - y1))
                                        else:
                                            tracker.start_track(gray3d, dlib.rectangle(x1, y1, x2, y2))
                                        new_trackers.append(tracker)
                            else:
                                print("Assigning tracker")
                                new_bboxes.append((x1, y1, x2 - x1, y2 - y1))
                                if self.track_op == "Boosting":
                                    tracker.init(gray3d, (x1, y1, x2 - x1, y2 - y1))
                                else:
                                    tracker.start_track(gray3d, dlib.rectangle(x1, y1, x2, y2))
                                new_trackers.append(tracker)
            
            # --------------------------------------
            # tmp_path = "data/{}_{}_display.png".format(self.frame_counter, self.plate_algo)
            # cv2.imwrite(tmp_path, displayFrame)
            # --------------------------------------

            indices = set(range(len(new_trackers))) - set(nonified_objs)
            ln = len(self.object_trackers)
            self.object_trackers = [new_trackers[ind] for ind in indices] + [self.object_trackers[i] for i in (set(range(ln)) - set(included_trackers))]
            obj_trackers_to_draw = list(set(range(ln)) - set(included_trackers))
            ln = len(self.last_bboxes) - self.num_tracked
            self.last_bboxes = new_bboxes + [self.last_bboxes[self.num_tracked+i] for i in (set(range(ln)) - set(included_trackers))]

        # print(self.trackers, self.events, self.num_tracked, detect_face)
        if display:
        # if 1:

            # -------------------------------------------------------
            # 11_05_18
            # -------------------------------------------------------
            # if APPLY_DRAW_BOXES:
            if 0:
                color, width = (255, 255, 150), 2
                print('*******************************')
                print('PLATE BBOXES: {}'.format(plate_bboxes))
                print('*******************************')
                for box_abs_coords in plate_bboxes:
                    x1,y1,x2,y2,x3,y3,x4,y4 = box_abs_coords

                    cv2.line(displayFrame, (x1,y1), (x2,y2), color, width)
                    cv2.line(displayFrame, (x2,y2), (x3,y3), color, width)
                    cv2.line(displayFrame, (x3,y3), (x4,y4), color, width)
                    cv2.line(displayFrame, (x1,y1), (x4,y4), color, width)
            # -------------------------------------------------------

            self.emit(SIGNAL('signal(PyQt_PyObject)'), displayFrame)
            # self.emit(SIGNAL('signal(PyQt_PyObject)'), gray3d)

        # prof.stop()
        # print(prof.output_text(unicode=True, color=True))

        return detect_face, displayFrame, need_update, trackers_to_draw, obj_trackers_to_draw

    def get_plate(self, car):
        # car = self.car

        res = None

        if self.plate_algo == 'anpr':
            res = find_plate(car)
    
        elif self.plate_algo in ('alpr', 'anpd'): # get rid of such text duplicates

            res_with_meta = get_plate_boxes(car, method=self.plate_algo)
            res = [item[0] for item in res_with_meta]

        elif self.plate_algo in ('EAST', 'CTPN'): # get rid of such text duplicates
            
            path_back = os.getcwd() # DIRTY
            os.chdir('_text_detect_tmp')

            my_detector = CAR_PLATES_SESS[self.plate_algo]
            res = my_detector.get_results(is_frame=car, save_files=False)[0]

            os.chdir(path_back) # DIRTY

        else:
            raise ValueError('Wrong self.plate_algo: {}'.format(self.plate_algo))

            
        # ------------------------------------
        # 03_05_18
        # for tests --> remove this block later
        # ------------------------------------
        SAVE_PLATES = False
        if SAVE_PLATES:
            color, width = (0, 255, 0), 7

            for box in res:

                x1,y1,x2,y2,x3,y3,x4,y4 = (int(item) for item in box)
                 
                cv2.line(car, (x1,y1), (x2,y2), color, width)
                cv2.line(car, (x2,y2), (x3,y3), color, width)
                cv2.line(car, (x3,y3), (x4,y4), color, width)
                cv2.line(car, (x1,y1), (x4,y4), color, width)
                
            tmp_path = "data/000_{}.png".format(self.plate_algo)
            cv2.imwrite(tmp_path, car)
        # ------------------------------------

        return res

    # TODO: every plate detection should return list of bboxes or one bbox (tuple)

    # -----------------------------------------------
    # 11_05_18: DOESN'T WORK
    # -----------------------------------------------
    def get_sess(self): 
        # DOES'T WORK: requires protobuf 3.5 (non-trivial issue)
        # "This program requires version 3.5.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1." 
        # https://github.com/BVLC/caffe/issues/5711
        return

        if self.plate_algo in ('CTPN', 'EAST'):
        
            print('LOADIND {} SESSION ... PLEASE WAIT'.format(self.plate_algo))

            path_back = os.getcwd() # DIRTY
            os.chdir('_text_detect_tmp')
            
            self.car_plates_sess = GetPlate(self.plate_algo)

            os.chdir(path_back)

            print('THE SESSION WAS SUCCESFULLY LOADED!')
            print(type(self.parent.get_thread.car_plates_sess))

        else:

            self.car_plates_sess = None
            print('THE SESIION IS {}'.format(self.car_plates_sess))
    
    # -----------------------------------------------
    # 18_05_18: DOESN'T WORK
    # -----------------------------------------------
    def refresh_emo_dict(self, bool_arg):
        if bool_arg:
            self.emo_count_dict = OrderedDict((key, dict()) for key in self.emo_count_dict)
        if DB_WRITE_EMO_18_05_18:
            print('\n\n\n\nEMO DICT WAS REFRESHED!\n\n\n\n')
            import time
            time.sleep(2)
    # -----------------------------------------------

    def add_id(self, id):
        matcher = FindMatchInsertGroup(self.search, id.id, self.type, self.method, face=id.face, frame=id.frame,
                                       estimator=self.estimator, recog=(self.faiss_recog, match_template))
        matcher.setAutoDelete(True)
        # matcher.signal.groups.connect(self.add_group)
        # self.trackers[len(self.ids)].id = id
        self.ids.append(id.id)
        self.update_swaps(self.ids)
        if self.gpu and self.encode_method == 'resnet':
            p, c = Pipe()
            matcher.find_match_insert_group(conn=c)
            id_templ = p.recv()
            if id_templ is not None:
                search_thread = threading.Thread(target=matcher.searchData, args=(id_templ[0], id_templ[1],))
                search_thread.start()
        else:
            self.find_match_pool.start(matcher)

    def add_group(self, id_numface):
        
        if id_numface[1] >= len(self.groups):
            self.groups = self.groups + [None for _ in range(len(self.groups), id_numface[1] - 1)]
            self.groups.append(id_numface[0])
        else:
            self.groups[id_numface[1]] = id_numface[0]

        if DB_TESTING_17_05_15:
        # if 1:
            enablePrint()
            print(id_numface)
            print('ADDED GROUP!')
            print('self.groups: {}'.format(self.groups))
            blockPrint()

    def update_events(self, events):
        self.events = events

    def update_trackers(self, trackers):
        self.trackers = trackers

    def update_info(self):
        # returns list of tuples: (time, first name, last name, image path)
        return get_info_(self.method)

    # -------------------------------
    # 20_04_18:
    # -------------------------------
    def faiss_recog_jail(self, query_vector, is_frame=True, num_neighb=1):

        print('INSIDE FAISS RECOG JAIL ...')

        threshold = self.jailbase_threshold
        method = self.encode_method

        index_obj = INDEX_JAIL[method]
        # print(index_obj)

        faiss_vector = index_obj.get_faiss_vector(query_vector, is_frame=is_frame)
        D, I = index_obj.search(faiss_vector, num_neighb=num_neighb)

        dist, idx = D[0][0], I[0][0]
        print(dist, idx)

        exist_face = dist < threshold
        exist_face_idx = (-1, idx)[exist_face]

        return exist_face, exist_face_idx
    # -------------------------------

    # @profile
    def faiss_recog(self, temp_color, faceIdx, frame, y, x, h, w, this_face):

        threshold = 0.3 if self.encode_method == "resnet" else 0.7

        if self.search_type == 'BruteForce':
            D, I = index_search(temp_color, is_frame=True,
                                encode_method=self.encode_method,
                                search_type=self.search_type,
                                num_neighb=3, update=True)

        minimum = 1e3
        ind = -1
        try:
            for i, d in enumerate(D[0]):
                if I[0][i] == this_face:
                    continue
                if d < minimum:
                    ind = i
                    minimum = d

            bExistFace = minimum < threshold
            if bExistFace:
                existFaceIdx = I[0][ind]
            else:
                existFaceIdx = -1
        except:
            bExistFace = False
            existFaceIdx = -1

        # -------------------------------
        # 20_04_18:
        # -------------------------------
        if self.jailbase_flag:
            exist_face, exist_face_idx = self.faiss_recog_jail(query_vector=temp_color)
            
            print("{} ... {}".format(exist_face, exist_face_idx))

            messages = ("FUH ... THIS GUY IS LAW-ABIDING!",
                "OH MY ... THIS GUY IS A CRIMINAL! ... our db's id = {}.".format(exist_face_idx))
            print(messages[exist_face])

            # ... do some stuff with DB etc
        # -------------------------------

        return bExistFace, existFaceIdx

    def swapFace(self, orgImg, d, num_person):
        try:
            if self.swap_data[num_person] == '' or not self.swap_data[num_person].endswith('.jpg') and not self.swap_data[num_person].endswith('.png'):
                swap_file = self.swapPath
            else:
                swap_file = self.swap_data[num_person]
        except:
            swap_file = self.swapPath

        # print swap_file

        swapImg = cv2.imread(str(swap_file))

        height, width, channels = orgImg.shape

        X1 = d[0] # d.left()
        Y1 = d[1]  # d.top()
        X2 = d[2] + d[0]  # d.right()
        Y2 = d[3] + d[1] # d.bottom()
        if (X1 < 0): X1 = 4
        if (X2 >= width): X2 = width - 5
        if (Y1 < 0): Y1 = 4
        if (Y2 >= height): Y2 = height - 5
        try:
            if self.draw_rects:
                cv2.rectangle(orgImg, (d[0], d[1]), (d[2] + d[0], d[3] + d[1]), (0, 255, 0), orgImg.shape[1] / 100)
            org = orgImg[Y1:Y2, X1:X2]
            orgImg[Y1:Y2, X1:X2] = faceSwap(org, swapImg)
        except Exception as e:
            pass
            # print "\n\t\t{0}\n".format(e)

        #self.emit(SIGNAL('signal(PyQt_PyObject)'), orgImg)
        return orgImg

    def dispatch_match_search_task(self, update):
        if update:
            match_thread = None # FindMatchThread(parent=self)
            match_thread.signals.result.connect(self.update_swaps)
            match_thread.set_data(self.data)
            match_thread.setAutoDelete(True)
            self.match_pool.start(match_thread)

    def dispatch_recognition_task(self, frame, classifiers, start):
        recog_thread = RecogThread()
        recog_thread.set_data(self.type, frame, classifiers, start, self.num_tracked)
        recog_thread.setAutoDelete(True)
        self.display_data = start, self.do_swap, self.track_op, frame
        self.data = self.method, self.type, frame
        recog_thread.signals.faces.connect(self.dispatch_display_task)
        recog_thread.signals.error.connect(self.send_error_signal)
        recog_thread.signals.update.connect(self.dispatch_match_search_task)
        self.recog_pool.start(recog_thread)
        #self.faces_lock.lock()

    def dispatch_display_task(self, faces):
        self.started = False
        self.faces = faces
        #self.faces_lock.unlock()
        self.display_data = self.display_data[0], self.display_data[1], self.display_data[2], self.display_data[3], faces, True
        from copy import copy
        display_thread = DisplayThread(display_data=copy(self.display_data))
        display_thread.setAutoDelete(True)
        display_thread.signals.frame.connect(self.emit_display)
        display_thread.signals.trackers.connect(self.set_trackers)
        display_thread.signals.swaps.connect(self.set_swaps)
        self.display_pool.start(display_thread)

    def emit_display(self, display_frame):
        self.emit(SIGNAL("signal(PyQt_PyObject)"), display_frame)

    def send_error_signal(self, message=''):
        self.emit(SIGNAL("PySigNoClassifier"))

    def set_trackers(self, tracks):
        self.started = True
        self.trackers = tracks
        self.num_tracked = len(tracks)

    def set_swaps(self, swaps):
        self.swaps = swaps

    def set_faces(self, faces):
        self.faces = faces


class RecogSingals(QObject):

    faces = pyqtSignal(list)
    update = pyqtSignal(bool)
    error = pyqtSignal(str)


class RecogThread(QRunnable):
    def __init__(self):
        QRunnable.__init__(self)
        self.type = None
        self.classifiers = None
        self.frame = None
        self.start = None
        self.num_tracked = None
        self.signals = RecogSingals()

    def set_data(self, type, frame, classifiers, start, num_tracked):
        from copy import copy
        self.type = copy(type)
        self.frame = copy(frame)
        self.classifiers = copy(classifiers)
        self.start = copy(start)
        self.num_tracked = copy(num_tracked)

    def run(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=self.find_faces, args=(child_conn,))
        p.start()
        faces, update = parent_conn.recv()
        p.join()
        self.signals.faces.emit(faces)
        self.signals.update.emit(update)

    def find_faces(self, conn):
        if self.classifiers is None:
            self.signals.error.emit("No classifier")
            conn.send([], None)
            conn.close()
            return

        frame = self.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        displayFrame = frame.copy()
        detect_face = []

        if self.type in ["dlib", "cnn"]:
            detector = self.classifiers[0]
            dets = detector(gray, 1)
            for i, d in enumerate(dets):
                if self.type in ["cnn"]:
                    d = d.rect
                detect_face.append((d.left(), d.top(), d.right(), d.bottom()))

        elif self.type == "cascade":
            side_cascade = self.classifiers[0]
            front_cascade = self.classifiers[1]
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.flip(gray1, 1)

            faces = side_cascade.detectMultiScale(gray, 1.3, 5)
            faces1 = side_cascade.detectMultiScale(gray1, 1.3, 5)
            fronts = front_cascade.detectMultiScale(gray1, 1.3, 5)

            detect_face += get_face(faces1)
            detect_face += get_face(fronts)
            height, width, channels = displayFrame.shape
            detect_face += get_face(faces, width, True)

            list_faces1 = get_face(faces1)
            list_frontes = get_face(fronts)
            height, width, channels = displayFrame.shape
            list_fases = get_face(faces, width, True)
            detect_face = list_fases + list_faces1

            if len(detect_face) > 0:
                for (x1, y1, x2, y2) in list_frontes:
                    bFrontFace = False
                    for (p1, q1, p2, q2) in list_fases:
                        if x2 < p1 or x1 > p2:
                            detect_face.append((x1, y1, x2, y2))
                            bFrontFace = True
                            break

                    if bFrontFace == False:
                        for (p1, q1, p2, q2) in list_faces1:
                            if x2 < p1 or x1 > p2:
                                detect_face.append((x1, y1, x2, y2))
                                break
            else:
                detect_face = list_frontes

        need_update = self.start or len(detect_face) > self.num_tracked
        conn.send((detect_face, need_update))
        conn.close()


class DisplaySignals(QObject):

    frame = pyqtSignal(np.ndarray)
    trackers = pyqtSignal(list)
    swaps = pyqtSignal(list)


class DisplayThread(QRunnable):
    def __init__(self, display_data):
        QRunnable.__init__(self)
        self.data = None
        self.swap_data = None
        self.swap_path = None
        self.display_data = display_data
        self.signals = DisplaySignals()

    def set_data(self, data, swap_data, swap_path):
        self.data = data
        self.swap_data = swap_data
        self.swap_path = swap_path

    def run(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=self.display, args=(child_conn,))
        p.start()
        trackers, displayFrame, swaps = parent_conn.recv()
        p.join()
        self.signals.frame.emit(displayFrame)
        self.signals.trackers.emit(trackers)
        self.signals.swaps.emit(swaps)

    def display(self, conn):
        start, do_swap, track_op, frame, detect_face, track = self.display_data
        displayFrame = frame.copy()
        trackers = []
        new_swaps = []
        if start:
            self.swaps = [do_swap for i in range(len(detect_face))]
        for i, (x1, y1, x2, y2) in enumerate(detect_face):
            if self.swaps[i]:
                displayFrame = self.swapFace(displayFrame, (x1, y1, x2 - x1, y2 - y1), i)
                new_swaps.append(True)
            else:
                cv2.rectangle(displayFrame, (x1, y1), (x2, y2), (0, 255, 0), displayFrame.shape[1] / 100)
                new_swaps.append(False)
            if track:
                print track_op
                if track_op == "MIL":
                    tracker = cv2.TrackerMIL_create()
                    ok = tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                elif track_op == "Boosting":
                    tracker = cv2.TrackerBoosting_create()
                    ok = tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                else:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
                    ok = True
                if ok:
                    trackers.append(tracker)
        conn.send((trackers, displayFrame, new_swaps))
        conn.close()

    def swapFace(self, orgImg, d, num_person):
        if self.swap_data[num_person] == '' or not self.swap_data[num_person].endswith('.jpg') and not self.swap_data[num_person].endswith('.png'):
            swap_file = self.swap_path
        else:
            swap_file = self.swap_data[num_person]

        swapImg = cv2.imread(str(swap_file))

        height, width, channels = orgImg.shape

        X1 = d[0] # d.left()
        Y1 = d[1]  # d.top()
        X2 = d[2] + d[0]  # d.right()
        Y2 = d[3] + d[1] # d.bottom()
        if (X1 < 0): X1 = 4
        if (X2 >= width): X2 = width - 5
        if (Y1 < 0): Y1 = 4
        if (Y2 >= height): Y2 = height - 5
        try:
            org = orgImg[Y1:Y2, X1:X2]
            orgImg[Y1:Y2, X1:X2] = faceSwap(org, swapImg)
            cv2.rectangle(orgImg, (d[0], d[1]), (d[2]+d[0], d[3]+d[1]), (0, 255, 0), orgImg.shape[1]/100)
        except:
            pass

        return orgImg


class PhotoFrameId(object):
    def __init__(self, id, frame, bbox):
        self.id = id
        self.frame = frame
        self.face = bbox


class PhotoSaverSignals(QObject):
    photo_id = pyqtSignal(PhotoFrameId)
    events = pyqtSignal(list)
    trackers = pyqtSignal(list)

    # if DB_WRITE_EMO_18_05_18:
    #     emo_inserted = pyqtSignal(bool) 


class PhotoSaver(QRunnable):
    def __init__(self, trackers, type, method, align, on_frame, num_face, data, events, search, swaps, is_alone=False, gray=None, emo_dict=None): # 17_05_18
        QRunnable.__init__(self)
        self.events = events
        self.trackers = trackers
        self.type = type
        self.method = method
        self.do_align = align
        self.on_frame = on_frame
        self.num_face = num_face
        self.data = data
        self.gray = gray
        self.signals = PhotoSaverSignals()
        self.search = search
        self.swaps = swaps
        self.alone = is_alone

        # 17_05_18
        self.emo_dict = emo_dict

    def run(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=self.process_events, args=(child_conn,))
        p.start()
        photo_id, events = parent_conn.recv()
        p.join()
        if type(self.on_frame) == list:
            self.signals.events.emit(events)
            for i, on_frame in enumerate(self.on_frame):
                if not on_frame:
                    self.signals.photo_id.emit(photo_id[i])
        else:
            if not self.on_frame:
                # self.signals.trackers.emit(self.trackers)
                self.signals.events.emit(events)
                self.signals.photo_id.emit(photo_id)

    # @profile
    def process_events(self, conn):
        photo_id = -1
        event_id = -1
        if type(self.on_frame) == list:
            print("\t\t\tCORRECT BRANCH PHOTO SAVER")
            photo_id = [-1 for _ in range(len(self.on_frame))]
            for i, on_frame in enumerate(self.on_frame):
                # try:
                if not on_frame:
                    roi_color, temp_color = self.get_images(i)
                    print("\t\t\tSAVE PHOTO")
                    photo_id[i], event_id = save_photo(temp_color, roi_color)
                    try:
                        self.events[i] = event_id
                    except:
                        self.events.append(event_id)
                else:
                    photo_id[i] = update_event(self.events[i])
                    
                    if DB_WRITE_EMO_18_05_18:

                        enablePrint() 
                        print('\nINSIDE PROCESS EVENTS ...')

                        event_id = self.events[i]
                        print('EVENT ID: {}'.format(event_id))
                        # emo_string = insert_emo(event_id, self.emo_dict) 
                        update_emo(event_id, self.emo_dict)
                        
                        # self.signals.emo_inserted.emit(True)
                        print(self.emo_dict)
                        
                        blockPrint()


                # except Exception as e:
                #     print "Exception in PhotoSaver: {0}".format(e)
                    # if event_id > -1:
                    #     if i == len(self.events):
                    #         self.events.append(event_id)
            results = []
            for i, photo in enumerate(photo_id):
                results.append(PhotoFrameId(photo, self.data[1][i], self.data[0]))
            if conn is not None:
                conn.send((results, self.events))
        else:
            try:
                if not self.on_frame:
                    roi_color, temp_color = self.get_images()
                    photo_id, event_id = save_photo(temp_color, roi_color)
                    self.events[self.num_face] = event_id
                else:
                    photo_id = update_event(self.events[self.num_face])

                    if DB_WRITE_EMO_18_05_18:
                    
                        enablePrint() 
                        print('\n... INSIDE PROCESS EVENTS ...')
                        
                        # sql = "SELECT * FROM EVENTS"
                        # sql = "SELECT * FROM PHOTO_GROUP"
                        # res = db_connector.runQueryMysql(sql, True)
                        # print(res)

                        event_id = self.events[self.num_face]
                        print('EVENT ID: {}'.format(event_id))
                        # emo_string = insert_emo(event_id, self.emo_dict) 
                        update_emo(event_id, self.emo_dict)

                        # self.signals.emo_inserted.emit(True)
                        print(self.emo_dict)
                        # self.emo_dict = OrderedDict((key, dict()) for key in self.emo_dict)
                        # print('\n\n\nself.emo_dict: {}\n\n\n'.format(self.emo_dict))
                        # print('********** THE SIGNAL WAS EMITTED ***************')

                        blockPrint()


            except Exception as e:
                print "Exception in PhotoSaver: {0}".format(e)
                if event_id > -1:
                    if self.num_face == len(self.events):
                        self.events.append(event_id)
            if conn is not None:
                conn.send((PhotoFrameId(photo_id, self.data[1], self.data[0]), self.events))
        if conn is not None:
            conn.close()

    def get_images(self, num_face=None):
        face, frame = self.data
        if num_face is not None:
            p1, q1, p2, q2 = face[num_face]
        else:
            p1, q1, p2, q2 = face
        if self.do_align:
            return align(frame, p1, q1, p2, q2, self.alone, self.gray)
        else:
            return prepare_face(frame, p1, q1, p2, q2, self.type)


aligner = FaceAligner(pose_predictor_68_point, desiredFaceWidth=256)

# @profile
def align(frame, p1, q1, p2, q2, alone=False, gray=None):
    height, width, channels = frame.shape
    rect = (p1, q1, p2, q2)
    unit = (p2 - p1) / 2

    x1 = int(p1 - unit * 1.5)
    y1 = int(q1 - unit * 1.5)
    x2 = int(p2 + unit * 1.5)
    y2 = int(q2 + unit * 1.5)
    if (x1 < 0): x1 = 4
    if (x2 >= width): x2 = width - 5
    if (y1 < 0): y1 = 4
    if (y2 >= height): y2 = height - 5
    roi_color = frame[y1:y2, x1:x2]

    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if self.type == "cascade":
    #     rect = dlib.rectangle(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
    # else:
    rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
    aligned = None
    for i in range(10):
        dlib_cuda_lock.acquire()
        try:
            aligned = aligner.align(frame, gray, rect)
            dlib_cuda_lock.release()
            break
        except:
            dlib_cuda_lock.release()
        time.sleep(0.05)
    if aligned is None:
        tempX1 = int(p1)
        tempY1 = int(q1)
        tempX2 = int(p2)
        tempY2 = int(q2)
        if (tempX1 < 0): tempX1 = 4
        if (tempX2 >= width): tempX2 = width - 5
        if (tempY1 < 0): tempY1 = 4
        if (tempY2 >= height): tempY2 = height - 5
        aligned = frame[tempY1:tempY2, tempX1:tempX2]
    if alone:
        return frame, aligned
    return roi_color, aligned

def prepare_face(frame1, p1, q1, p2, q2, _type="dlib"):
    height, width, channels = frame1.shape
    unit = (p2 - p1) / 2
    if _type in ["dlib", "cnn"]:
        unit = p2 / 2

    x1 = int(p1 - unit * 1.5)
    y1 = int(q1 - unit * 1.5)
    x2 = int(p2 + unit * 1.5)
    y2 = int(q2 + unit * 1.5)
    if _type in ["dlib", "cnn"]:
        x2 += int(p1)
        y2 += int(q1)
    if (x1 < 0): x1 = 4
    if (x2 >= width): x2 = width - 5
    if (y1 < 0): y1 = 4
    if (y2 >= height): y2 = height - 5
    roi_color = frame1[y1:y2, x1:x2]

    tempX1 = int(p1)
    tempY1 = int(q1)
    tempX2 = int(p2)
    tempY2 = int(q2)
    if _type in ["dlib", "cnn"]:
        tempX2 += int(p1)
        tempY2 += int(q1)
    if (tempX1 < 0): tempX1 = 4
    if (tempX2 >= width): tempX2 = width - 5
    if (tempY1 < 0): tempY1 = 4
    if (tempY2 >= height): tempY2 = height - 5
    temp_color = frame1[tempY1:tempY2, tempX1:tempX2]

    return [roi_color, temp_color]


class InsertGroupSignal(QObject):
    groups = pyqtSignal(tuple)


class FindMatchInsertGroup(QRunnable):
    def __init__(self, search, photo_id, type, method, frame, face, estimator="KERAS", recog=(getPostThread.faiss_recog, match_template)):
        QRunnable.__init__(self)
        self.search = search
        self.photo_id = photo_id
        self.type = type
        self.method = method
        self.frame = frame
        self.face = face
        self.estimator = estimator
        self.recog = { 'faiss' : recog[0], 'template' : recog[1] }
        self.signal = InsertGroupSignal()

    def run(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=self.find_match_insert_group, args=(child_conn,))
        p.start()
        id_templ = parent_conn.recv()
        p.join()
        if id_templ is not None:
            self.searchData(id_templ[0], id_templ[1])

    # @profile
    def find_match_insert_group(self, conn=None):
        if self.photo_id == -1:
            print("Incorrect photo id, neither recognition nor search could be run")
            if conn is not None:
                conn.send(None)
            return
        full_path = os.getcwd()
        templates = os.path.join(full_path, 'faceTemplates')

        # Suppose it is better to read already aligned image then align again even without reading
        templ = cv2.imread(os.path.join(templates, '{}.png'.format(self.photo_id)))

        print("id: {}, shape: {}".format(self.photo_id, templ.shape))
        # print("reading image {0}".format(os.path.join(templates, '{}.png'.format(self.photo_id))))
        if self.method == "faiss":
            exist, face_id = self.recog['faiss'](templ, None, None, None, None, None, None, self.photo_id)
        else:
            files = os.listdir(templates)
            x, y, h, w = self.face
            exist, face_id = self.recog['template'](templ, len(files) - 1, self.frame, y, x, h - y, w - x, self.photo_id)
        # try:
        if exist:
            group_id = get_group(face_id, self.method)
            if conn is not None:
                conn.send(None)
            insert_photo_group(self.photo_id, group_id)
        else:
            group_id = insert_group(self.method)
            insert_photo_group(self.photo_id, group_id)
            insert_search(self.search, self.photo_id)
            if conn is not None:
                conn.send(('{}.png'.format(self.photo_id), templ))
            # self.searchData('{}.png'.format(self.photo_id), templ)
        # except Exception as e:
        #     print "\tsomething wrong in FindMatchInsertGroup:", e

    def searchData(self, filename, temp_img):
        full_path = os.getcwd()
        full_path = full_path.replace("\\", "/")
        filePath = full_path + "/faces/" + filename
        searchInfo(filePath, self.search, int(filename.split('.')[0]), False, self.estimator, frame=self.frame, face=self.face, temp_img=temp_img, q=queues)
        # _searchTread = threading.Thread(target=searchInfo, args=(
        #     filePath, self.search, int(filename.split('.')[0]), False,))
        # _searchTread.start()


class GroupGetter(QRunnable):

    # 17_05_18
    # ----------------
    group_id = len(db_connector.runQueryMysql('SELECT * FROM GROUPS', True))
    # ----------------

    def __init__(self, num_face, face, frame, method, recog, alone, type, align, gray):
        QRunnable.__init__(self)
        self.num_face = num_face
        self.face = face
        self.frame = frame
        self.gray = gray
        self.method = method
        self.recog = recog
        self.alone = alone
        self.type = type
        self.do_align = align
        self.signal = InsertGroupSignal()

    def run(self):
        par, chld = Pipe()
        p = Process(target=self.get, args=(chld,))
        p.start()
        id = par.recv()
        p.join()
        self.signal.groups.emit((id, self.num_face))

    def get(self, conn=None):
        p1, q1, p2, q2 = self.face
        if self.do_align:
            _, templ = align(self.frame, p1, q1, p2, q2, self.alone, self.gray)
        else:
            _, templ = prepare_face(self.frame, p1, q1, p2, q2, self.type)

        print("\tgot faces in GroupGetter")

        if self.method == "faiss":
            exist, face_id = self.recog[0](templ, None, None, None, None, None, None, -1)
        else:
            full_path = os.getcwd()
            templates = os.path.join(full_path, 'faceTemplates')
            files = os.listdir(templates)
            x, y, h, w = self.face
            exist, face_id = self.recog[1](templ, len(files) - 1, self.frame, y, x, h - y, w - x, -1)
        if exist:
            group_id = get_group(face_id, self.method)
        else:
            # group_id = len(db_connector.runQueryMysql('SELECT * FROM GROUPS', True))

            # 17_05_18
            # ---------------------------
            # enablePrint()
            # print('HERE ... GroupGetter.group_id: {}'.format(GroupGetter.group_id))
            group_id = GroupGetter.group_id
            GroupGetter.group_id += 1
            # print('GroupGetter.group_id: {}'.format(GroupGetter.group_id))
            # blockPrint()
            # ---------------------------

        conn.send(group_id)


        if DB_TESTING_17_05_15:

            print("FACE_ID, EXISTS: {}, {}".format(face_id, exist))
            print("GROUP: {}".format(group_id))


class YoloSignals(QObject):
    frame_displayed = pyqtSignal(np.ndarray)
    results = pyqtSignal(tuple)


class YoloProcessing(QRunnable):

    def __init__(self, yolo, frame, objects_to_parse):
        QRunnable.__init__(self)
        self.yolo = yolo
        self.frame = frame
        self.objects = objects_to_parse
        self.signals = YoloSignals()

    def run(self):
        par, chld = Pipe()
        p = Process(target=self.process_frame, args=(chld,))
        p.start()
        res, frame = par.recv()
        p.join()
        # self.signals.frame_displayed.emit(frame)
        self.signals.results.emit((frame, res))

    def process_frame(self, conn=None):
        results, displayFrame = [], self.frame
        print("Objects count in YOLO: {0}".format(len(self.objects)))
        if len(self.objects) > 0:
            results, displayFrame = self.yolo.process_frame(self.frame, self.objects)
        conn.send((results, displayFrame))
