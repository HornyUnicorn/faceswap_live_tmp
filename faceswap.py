""" The master faceswap.py script """

import os
import sys

# ---------------------------------------
# checking py-version:
# (based on the original module's code)
# ---------------------------------------
print('\nRunning the DeepFakes FaceSwap module ...')

py_version, py_version_minor = sys.version_info[0], sys.version_info[1] 

if py_version < 3:
    print("WARNING! You're running the program on Python2 which is not fully supported!")
    # raise Exception("This program requires at least python3.2")
if py_version == 3 and py_version_minor < 2:
    raise Exception("This program requires at least python3.2")

# ---------------------------
# configure GPUs used:
# ---------------------------
import GPUtil

deviceIDs = GPUtil.getAvailable(order='first', limit=8, maxLoad=0.5, maxMemory=0.5)
print('FaceSwap deviceIDs: {}'.format(deviceIDs))

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6' if (4 in deviceIDs and 5 in deviceIDs and 6 in deviceIDs) else '0'
print('FaceSwap CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

# REMARK:
    # doesn't work in agreement with train method: 
    # for example, on train step the program uses the GPU devices from id=0 
    # while CUDA_VISIBLE_DEVICES can be 1,2,3  

# ---------------------------
# supress tflow warnings:
# ---------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# ---------------------------

import lib.cli as cli

def bad_args(args):
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)

# ---------------------------------------
# FaceSwap class - first iteration:
# (based on original faceswap module)
# ---------------------------------------

class MyFaceSwap():

    GPUS = 2

    # ---------------------------------------
    # DATA DIRS:
    # ---------------------------------------

    MODEL_DIR = '_data/video_prep_ffmpeg/model/' 

    INPUT_DIR_EXTRACT = '_data/video_prep_ffmpeg/frames/'
    OUTPUT_DIR_EXTRACT = '_data/video_prep_ffmpeg/faces/'

    DIR_A_TRAIN = '_data/video_prep_ffmpeg/faces/emma_360_cut.mp4_faces/'
    DIR_B_TRAIN = '_data/video_prep_ffmpeg/faces/jade_360_cut.mp4_faces/'

    INPUT_DIR_CONVERT = INPUT_DIR_EXTRACT
    OUTPUT_DIR_CONVERT = '_data/video_prep_ffmpeg/frames_swapped/'

    # ---------------------------------------
    # INIT PARSER:
    # ---------------------------------------

    PARSER = cli.FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()

    cli.ExtractArgs(SUBPARSER, "extract", "Extract the faces from pictures")
    cli.TrainArgs(SUBPARSER, "train", "This command trains the model for the two faces A and B")
    cli.ConvertArgs(SUBPARSER, "convert", "Convert a source image to a new one with the face swapped")

    PARSER.set_defaults(func=bad_args)


    def __init__(self):
        pass
                
    def extract(self, input_dir=None, output_dir=None):

        if input_dir is None:
            input_dir = MyFaceSwap.INPUT_DIR_EXTRACT
        if output_dir is None:
            output_dir = MyFaceSwap.OUTPUT_DIR_EXTRACT

        ARGUMENTS = MyFaceSwap.PARSER.parse_args(["extract"])
        
        ARGUMENTS.input_dir = input_dir
        ARGUMENTS.output_dir = output_dir

        ARGUMENTS.func(ARGUMENTS)

    def train(self, input_A=None, input_B=None, model_dir=None, gpus=None, preview=True, stop_threshold=0, stop_iternum=float('inf')):
        
        if input_A is None:
            input_A = MyFaceSwap.DIR_A_TRAIN
        if input_B is None:
            input_B = MyFaceSwap.DIR_B_TRAIN
        if model_dir is None:
            model_dir = MyFaceSwap.MODEL_DIR
        if gpus is None:
            gpus = MyFaceSwap.GPUS

        ARGUMENTS = MyFaceSwap.PARSER.parse_args(["train"])
        
        ARGUMENTS.input_A = input_A
        ARGUMENTS.input_B = input_B
        ARGUMENTS.model_dir = model_dir
        ARGUMENTS.gpus = gpus
        ARGUMENTS.preview = preview

        ARGUMENTS.thresh = stop_threshold
        ARGUMENTS.iter_num = stop_iternum

        ARGUMENTS.func(ARGUMENTS)

        # removing backup-files to free space
        import os 
        for item in os.listdir(model_dir):
            if item.endswith(".bk"):
                os.remove(os.path.join(model_dir, item))
        print('backup-files were removed')

    def convert(self, input_dir=INPUT_DIR_CONVERT, output_dir=OUTPUT_DIR_CONVERT, model_dir=MODEL_DIR, gpus=GPUS):

        ARGUMENTS = MyFaceSwap.PARSER.parse_args(["convert"])
        
        ARGUMENTS.input_dir = input_dir
        ARGUMENTS.output_dir = output_dir
        ARGUMENTS.model_dir = model_dir
        ARGUMENTS.gpus = gpus

        ARGUMENTS.func(ARGUMENTS)

# =========================================================
# FACEIT_LIVE INTEGRATION:
# =========================================================

# sys.path.append('faceswap')

from pathlib import Path
import shutil
import argparse
import time

import cv2
import numpy
import tqdm

# --------------------------------
import youtube_dl
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop
from moviepy.editor import clips_array, TextClip, CompositeVideoClip
# --------------------------------

from lib.utils import FullHelpArgumentParser
from scripts.extract import ExtractTrainingData
from scripts.train import TrainingProcessor
from lib.faces_detect import detect_faces_LIVE
from plugins.PluginLoader import PluginLoader
from lib.FaceFilter import FaceFilter_LIVE


# ---------------------------------------
# FaceSwap class - second iteration:
# (based on modified faceswap module)

# links:
    # https://github.com/goberoi/faceit
    # https://github.com/alew3/faceit_live
# ---------------------------------------


# ===================================================
from plugins.Model_LIVE import Model_LIVE
from plugins.Convert_Masked_LIVE import Convert as Convert_LIVE

import tensorflow as tf
from PyQt4.QtCore import QThread
from multiprocessing import Queue

class SwapGetter(QThread):

    def __init__(self, queues=None, model_path='models/test_2_faces'):
        QThread.__init__(self)

        self.queues_num = 2

        if type(queues) != list:
            raise AssertionError("'queues' parameter of SwapGetter.__init__ should be a list of {} multiprocessing.Queue objects".format(self.queues_num))
        else:
            if len(queues) < self.queues_num:
                raise AssertionError("'queues' parameter of SwapGetter.__init__ should be a list of {} multiprocessing.Queue objects".format(self.queues_num - 1))

        self.q = queues

        self.model_path = model_path
        self.model = None
        self.converter = None

        self.stopped = False

    def init(self):
        
        print("\tModel initialization")
    
        self.model = self.get_model()
        self.converter = self.get_converter()
        
        print("\tInitialization completed")

    def run(self):

        import tensorflow as tf
        import keras.backend.tensorflow_backend as K
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "4"
        K.set_session(tf.Session(config=config))
        self.init()

        while not self.stopped:
            
            try:
                frame = self.q[0].get()
                print('got_frame ... SwapGetter')

                image = self.convert_frame(frame)
                self.q[1].put(image)

                print('put_image ... SwapGetter')

            except:
                pass

    # -----------------------------------------------------------------
    def get_model(self):

        model = Model_LIVE(Path(self.model_path))
        
        if not model.load(False):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)

        print('Checkpoint_1 ... Model loaded')

        return model
    
    def get_converter(self):
         
        converter = Convert_LIVE(self.model.converter(False),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        print('Checkpoint_2 ... Converter loaded')

        return converter
    # -----------------------------------------------------------------

    def convert_frame(self, frame):

        for face in detect_faces_LIVE(frame, "cnn"): 
            frame = self.converter.patch_image(frame, face)

        return frame


# ===================================================
# TESTING SwapGetter:
# ===================================================

if 0:

    PAIR_VIDEO_FLAG = 0

    if PAIR_VIDEO_FLAG:
        model_path = 'models/test_2_faces'
        PATH_TO_VIDEO = './data/videos/pair_360p_cut.mp4'

    else:
        model_path = 'models/emma_to_jade'
        PATH_TO_VIDEO = './data/videos/emma_360_cut.mp4'
        
    
    queues = [Queue(), Queue()]

    inst = SwapGetter(queues, model_path=model_path)
    inst.start()

    video_capture = cv2.VideoCapture(PATH_TO_VIDEO)
    
    while 1:

        ret, frame = video_capture.read()

        if not ret:
            print("RET IS NONE ... I'M QUIT")
            video_capture.release()
            break

        if PAIR_VIDEO_FLAG:
            frame[:, 0:frame.shape[1]/2] = 0 # ~ cropping left half of an image

        print('HANDLING NEW FRAME ...')

        queues[0].put(cv2.flip(frame, 1))
        print('PUT A FRAME (with flip)!')
         

        # import time
        # time.sleep(1)

        
        cv2.imshow('Original', frame)

        while 1:
            try:
                image = queues[1].get()
                print('GOT AN IMAGE!')

                cv2.imshow('Video', cv2.flip(image, 1))
                break
            except:
                pass


        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("KEYBOARD INTERRUPT ... I'M QUIT")
            video_capture.release()
            break

    cv2.destroyAllWindows()
    exit()
    
    sys.exit(0)
# ===================================================


class FaceIt:
    """
    creates a faceswap-instance,
    defined by the names of conversion and persons,
    e.g. 'sid_to_nancy', 'sid', 'nancy' 
    """

    GPUS = 2

    # -----------------------------------
    # PATHS TO DATA DIRS:
    # -----------------------------------
    
    VIDEO_PATH = 'data/videos'
    PERSON_PATH = 'data/persons'
    PROCESSED_PATH = 'data/processed'
    OUTPUT_PATH = 'data/output'
    MODEL_PATH = 'models'

    # -----------------------------------
    # MODELS CLASS ATTRIBUTE:
    # -----------------------------------

    MODELS = {}

    # models expansion
    @classmethod
    def add_model(cls, model):
        FaceIt.MODELS[model._name] = model
    
    def __init__(self, name, person_a, person_b):
        
        # -----------------------------------
        def _create_person_data(person):
            return {
                'name' : person,
                'videos' : [],
                'faces' : os.path.join(FaceIt.PERSON_PATH, person + '.jpg'),
                'photos' : []
            }
        # -----------------------------------
        
        self._name = name

        self._people = {
            person_a : _create_person_data(person_a),
            person_b : _create_person_data(person_b),
        }
        self._person_a = person_a
        self._person_b = person_b
        
        self._faceswap = FaceSwapInterface()

        if not os.path.exists(os.path.join(FaceIt.VIDEO_PATH)):
            os.makedirs(FaceIt.VIDEO_PATH)            

    def add_photos(self, person, photo_dir):
        self._people[person]['photos'].append(photo_dir)
            
    def add_video(self, person, name, url=None, fps=20):
        self._people[person]['videos'].append({
            'name' : name,
            'url' : url,
            'fps' : fps
        })

    def fetch(self):
        self._process_media(self._fetch_video)

    def extract_frames(self):
        self._process_media(self._extract_frames)

    def extract_faces(self):        
        self._process_media(self._extract_faces)
        self._process_media(self._extract_faces_from_photos, 'photos')        

    def all_videos(self):
        return self._people[self._person_a]['videos'] + self._people[self._person_b]['videos']

    def _process_media(self, func, media_type='videos'):
        for person in self._people:
            for video in self._people[person][media_type]:
                func(person, video)

    def _video_path(self, video):
        return os.path.join(FaceIt.VIDEO_PATH, video['name'])        

    def _video_frames_path(self, video):
        return os.path.join(FaceIt.PROCESSED_PATH, video['name'] + '_frames')        

    def _video_faces_path(self, video):
        return os.path.join(FaceIt.PROCESSED_PATH, video['name'] + '_faces')

    def _model_path(self, use_gan=False):
        path = FaceIt.MODEL_PATH
        if use_gan:
            path += "_gan"
        return os.path.join(path, self._name)

    def _model_data_path(self):
        return os.path.join(FaceIt.PROCESSED_PATH, "model_data_" + self._name)
    
    def _model_person_data_path(self, person):
        return os.path.join(self._model_data_path(), person)

    def _fetch_video(self, person, video):
        options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio',
            'outtmpl': os.path.join(FaceIt.VIDEO_PATH, video['name']),
            'merge_output_format' : 'mp4'
        }
        # only download videos that are from youtube, otherwise they should be in data/videos
        with youtube_dl.YoutubeDL(options) as ydl:
            if (str(video['url']).startswith('https://www.youtube.com/watch?v')):
                x = ydl.download([video['url']])

    def _extract_frames(self, person, video):
        video_frames_dir = self._video_frames_path(video)
        video_clip = VideoFileClip(self._video_path(video))
        
        start_time = time.time()
        print('[extract-frames] about to extract_frames for {}, fps {}, length {}s'.format(video_frames_dir, video_clip.fps, video_clip.duration))
        
        if os.path.exists(video_frames_dir):
            print('[extract-frames] frames already exist, skipping extraction: {}'.format(video_frames_dir))
            return
        
        os.makedirs(video_frames_dir)
        frame_num = 0
        for frame in tqdm.tqdm(video_clip.iter_frames(fps=video['fps']), total = video_clip.fps * video_clip.duration):
            video_frame_file = os.path.join(video_frames_dir, 'frame_{:03d}.jpg'.format(frame_num))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            cv2.imwrite(video_frame_file, frame)
            frame_num += 1

        print('[extract] finished extract_frames for {}, total frames {}, time taken {:.0f}s'.format(
            video_frames_dir, frame_num-1, time.time() - start_time))            

    def _extract_faces(self, person, video):
        video_faces_dir = self._video_faces_path(video)

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(video_faces_dir))
        
        if os.path.exists(video_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(video_faces_dir))
            return
        
        os.makedirs(video_faces_dir)
        self._faceswap.extract(self._video_frames_path(video), video_faces_dir, self._people[person]['faces'])

    def _extract_faces_from_photos(self, person, photo_dir):
        photo_faces_dir = self._video_faces_path({ 'name' : photo_dir })

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(photo_faces_dir))
        
        if os.path.exists(photo_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(photo_faces_dir))
            return
        
        os.makedirs(photo_faces_dir)
        self._faceswap.extract(self._video_path({ 'name' : photo_dir }), photo_faces_dir, self._people[person]['faces'])

    def preprocess(self):
        self.fetch()
        self.extract_frames()
        self.extract_faces()
    
    def _symlink_faces_for_model(self, person, video):
        if isinstance(video, str):
            video = {'name': video}
        for face_file in os.listdir(self._video_faces_path(video)):
            target_file = os.path.join(self._model_person_data_path(person), video['name'] + "_" + face_file)
            face_file_path = os.path.join(os.getcwd(), self._video_faces_path(video), face_file)
            os.symlink(face_file_path, target_file)

    def train(self, dir_A=None, dir_B=None, dir_model=None, gpus=None, use_gan=False, preview=True, stop_threshold=0, stop_iternum=float('inf'), clean_data=True):
        # Setup directory structure for model, and create one director for person_a faces, and
        # another for person_b faces containing symlinks to all faces.

        # -------------------------------------------
        # printouts for inspecting the 'contains no images' bug: 
        # -------------------------------------------
        # print(self._model_path(use_gan)) # models/emma_to_jade
        # print(self._model_data_path()) # data/processed/model_data_emma_to_jade
        # for person in self._people:
        #     print(self._model_person_data_path(person)) 
        #     # data/processed/model_data_emma_to_jade/jade
        #     # data/processed/model_data_emma_to_jade/emma
        # sys.exit(0)
        # -------------------------------------------

        if not os.path.exists(self._model_path(use_gan)):
            os.makedirs(self._model_path(use_gan))

        if clean_data:
            
            if os.path.exists(self._model_data_path()):
                shutil.rmtree(self._model_data_path())

            for person in self._people:
                os.makedirs(self._model_person_data_path(person))

            self._process_media(self._symlink_faces_for_model)


        if gpus is None:
            gpus = FaceIt.GPUS

        if dir_A is None:
            dir_A = self._model_person_data_path(self._person_a) 
        if dir_B is None:
            dir_B = self._model_person_data_path(self._person_b)
        if dir_model is None:
            dir_model = self._model_path(use_gan)

        self._faceswap.train(dir_A, dir_B, dir_model, gpus=gpus, gan=use_gan, preview=preview, stop_threshold=stop_threshold, stop_iternum=stop_iternum)

    # -----------------------------------------------------------------
    # There are discovered some problems with loading models 
    # in the module from outer call ... 
    # So, need to edit the loading model part

    def get_model(self, swap_model=False, use_gan=False):
        # print(os.getcwd()) 
        
        # model = PluginLoader._import("Model_LIVE", "Model_LIVE")(Path(self._model_path(use_gan)))

        model = Model_LIVE(Path(self._model_path(use_gan)))
        
        if not model.load(swap_model):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)

        print('Checkpoint_1 ... Model loaded')

        return model
    
    def get_converter(self, model):
        
        # converter = PluginLoader._import("Convert", "Convert_Masked_LIVE")
        converter = Convert_LIVE
        converter = converter(model.converter(False),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        print('Checkpoint_2 ... Converter loaded')

        return converter

    # -----------------------------------------------------------------
    
    def convert(self, video_file, swap_model=False, duration=None, start_time=None, use_gan=False, face_filter=False, photos=True, crop_x=None, width=None, side_by_side=False, live=False):

        # Magic incantation to not have tensorflow blow up with an out of memory error.
        import tensorflow as tf
        import keras.backend.tensorflow_backend as K
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list="0"
        K.set_session(tf.Session(config=config))

        # Load model
        model_name = "Original"
        converter_name = "Masked"
        if use_gan:
            model_name = "GAN"
            converter_name = "GAN"

        # -----------------------------------------------------------
        # FIXING THE BUG with Model loading:
            # model = PluginLoader.get_model(model_name)(Path(self._model_path(use_gan)))
            # TypeError: __init__() takes exactly 3 arguments (2 given)
        # -----------------------------------------------------------

            # tmp_1 = PluginLoader.get_model(model_name)
            # tmp_1 = PluginLoader._import("Model_LIVE", "Model_LIVE") # that works (crutch however)
            # tmp_2 = Path(self._model_path(use_gan)) # models/emma_to_jade
            # print('\n\n\n{}\n{}\n{}\n{}\n\n\n'.format(tmp_1, type(tmp_1), tmp_2, type(tmp_2))) 
            # sys.exit(0)
            
            # values in faceit_live module:
                # plugins.Model_Original.Model
                # <type 'classobj'>
                # models/emma_to_jade
                # <class 'pathlib.PosixPath'>

            # values here:
                # plugins.Model_Original.Model.Model
                # <type 'classobj'>
                # models/emma_to_jade
                # <class 'pathlib.PosixPath'>
        # -----------------------------------------------------------

        # model = PluginLoader.get_model(model_name)(Path(self._model_path(use_gan))) # ==> crash
        model = PluginLoader._import("Model_LIVE", "Model_LIVE")(Path(self._model_path(use_gan)))

        # print('\n\n\n{}\n\n\n'.format(self._model_path(use_gan))) # e.g. models/test_2_faces
        # sys.exit(0)

        if not model.load(swap_model):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)

        print('Checkpoint_1 ... Model loaded')

        # -----------------------------------------------------------
        # FIXING THE BUG with Converter loading:
        # -----------------------------------------------------------

            # tmp_1 = PluginLoader.get_converter(converter_name)
            # tmp_1 = PluginLoader._import("Convert", "Convert_Masked_LIVE")
            # print('\n\n\n{}\n{}\n\n\n'.format(tmp_1, type(tmp_1))) 
            # sys.exit(0)

            # faceit_live module:
                # plugins.Convert_Masked.Convert
                # <type 'classobj'>

            # here:
                # plugins.Convert_Masked.Convert
                # <type 'classobj'>
        # -----------------------------------------------------------

        # Load converter
        # converter = PluginLoader.get_converter(converter_name) # ==> crash
        converter = PluginLoader._import("Convert", "Convert_Masked_LIVE")
        converter = converter(model.converter(False),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        print('Checkpoint_2 ... Converter loaded')

        # Load face filter
        filter_person = self._person_a
        if swap_model:
            filter_person = self._person_b
        filter = FaceFilter_LIVE(self._people[filter_person]['faces'])

        # Define conversion method per frame
        def _convert_frame(frame, convert_colors=True):
            # if convert_colors:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            
            DEBUG_MODE = 0
            for face in detect_faces_LIVE(frame, "cnn"):
                
                if DEBUG_MODE:
                    print('Got face!')
                    # print(dir(face)) # image, x, y, w, h, landmarks
                    print('Face geometry: ({},{},{},{})'.format(face.x,face.y,face.w,face.h))
                    print('Face landmarks: {}'.format(face.landmarks))

                    cv2.imshow('Face', face.image)
                    continue

                if (not face_filter) or (face_filter and filter.check(face)):
                    
                    # if 1:
                    #     print(dir(face.landmarks))
                    #     face.landmarks = []

                    frame = converter.patch_image(frame, face)
                    if not live:
                        frame = frame.astype(numpy.float32)

            # if convert_colors:                    
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV

            return frame

        def _convert_helper(get_frame, t):
            return _convert_frame(get_frame(t))

        # ===================================================
        if live:
            
            print('Staring live mode ...')
            print('Press "Q" to Quit')
            
            PATH_TO_VIDEO = './data/videos/emma_360_cut.mp4'
            
            if TEST_2_FACES_FLAG:
                # PATH_TO_VIDEO = './_data/videos/pair_360p_original.mp4'
                PATH_TO_VIDEO = './data/videos/pair_360p_cut.mp4'

            video_capture = cv2.VideoCapture(PATH_TO_VIDEO)
            
            width = video_capture.get(3)  # float
            height = video_capture.get(4) # float
            print("video dimensions = {} x {}".format(width,height))
            
            while 1:

                ret, frame = video_capture.read()
                # print(frame.shape, frame.dtype) # (360, 640, 3), uint8
                
                # frame = cv2.resize(frame, (640, 480))

                print('HANDLING NEW FRAME ...')

                if CROP_HALF_OF_FRAME == 'left':
                    frame[:, 0:frame.shape[1]/2] = 0 # ~ cropping left half of an image
                # elif CROP_HALF_OF_FRAME == 'right':
                    # pass

                if not ret:
                    print("RET IS NONE ... I'M QUIT")
                    video_capture.release()
                    break


                # block without try/except -  to catch actual errors:
                frame = cv2.flip(frame, 1)
                image = _convert_frame(frame, convert_colors=False)
                print('GOT AN IMAGE!')
                frame = cv2.flip(frame, 1)
                image = cv2.flip(image, 1)

                try: # with flip: 
                    
                    # flip image, because webcam inverts it and we trained the model the other way! 
                    frame = cv2.flip(frame, 1)

                    image = _convert_frame(frame, convert_colors=False)
                    print('GOT AN IMAGE!')

                    # flip it back
                    frame = cv2.flip(frame, 1)
                    image = cv2.flip(image, 1)
                        
                except:

                    try: # without flip:
            
                        image = _convert_frame(frame, convert_colors=False)
                        print('GOT AN IMAGE!')

                    except:

                        print("HMM ... CONVERTATION FAILED ... I'M QUIT")
                        continue
                        # video_capture.release()
                        # break

                cv2.imshow('Video', image)
                cv2.imshow('Original', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("KEYBOARD INTERRUPT ... I'M QUIT")
                    video_capture.release()
                    break

            cv2.destroyAllWindows()
            exit()
        # ===================================================

        media_path = self._video_path({'name':video_file})
        if not photos:
            # Process video; start loading the video clip
            video = VideoFileClip(media_path)

            # If a duration is set, trim clip
            if duration:
                video = video.subclip(start_time, start_time + duration)
            
            # Resize clip before processing
            if width:
                video = video.resize(width = width)

            # Crop clip if desired
            if crop_x:
                video = video.fx(crop, x2 = video.w / 2)

            # Kick off convert frames for each frame
            new_video = video.fl(_convert_helper)

            # Stack clips side by side
            if side_by_side:
                def add_caption(caption, clip):
                    text = (TextClip(caption, font='Amiri-regular', color='white', fontsize=80).
                            margin(40).
                            set_duration(clip.duration).
                            on_color(color=(0,0,0), col_opacity=0.6))
                    return CompositeVideoClip([clip, text])
                video = add_caption("Original", video)
                new_video = add_caption("Swapped", new_video)                
                final_video = clips_array([[video], [new_video]])
            else:
                final_video = new_video

            # Resize clip after processing
            #final_video = final_video.resize(width = (480 * 2))

            # Write video
            if not os.path.exists(os.path.join(self.OUTPUT_PATH)):
                os.makedirs(self.OUTPUT_PATH)
            output_path = os.path.join(self.OUTPUT_PATH, video_file)
            final_video.write_videofile(output_path, rewrite_audio = True)
            
            # Clean up
            del video
            del new_video
            del final_video
        else:
            # Process a directory of photos
            for face_file in os.listdir(media_path):
                face_path = os.path.join(media_path, face_file)
                image = cv2.imread(face_path)
                image = _convert_frame(image, convert_colors = False)
                cv2.imwrite(os.path.join(self.OUTPUT_PATH, face_file), image)

class FaceSwapInterface:

    def __init__(self):
        self._parser = FullHelpArgumentParser()
        self._subparser = self._parser.add_subparsers()

    def extract(self, input_dir, output_dir, filter_path):
        extract = ExtractTrainingData(
            self._subparser, "extract", "Extract the faces from a pictures.")
        args_str = "extract --input-dir {} --output-dir {} --processes 1 --detector cnn --filter {}"
        args_str = args_str.format(input_dir, output_dir, filter_path)
        self._run_script(args_str)

    def train(self, input_a_dir, input_b_dir, model_dir, gpus, gan=False, preview=True, stop_threshold=0, stop_iternum=float('inf')):

        # --------------------------
        # ORIGINAL IMPLEMENTATION:
        # --------------------------

            # model_type = "Original"
            # if gan:
            #     model_type = "GAN"

            # train = TrainingProcessor(
            #     self._subparser, "train", "This command trains the model for the two faces A and B.")

            # args_str = "train --input-A {} --input-B {} --model-dir {} --trainer {} --batch-size {} --gpus {} --write-image"
            # if preview:
            #     args_str += " -p"
            # args_str = args_str.format(input_a_dir, input_b_dir, model_dir, model_type, 512, gpus)
            
            # self._run_script(args_str)

        myswap = MyFaceSwap()
        myswap.train(input_A=input_a_dir, input_B=input_b_dir, model_dir=model_dir, gpus=gpus, preview=preview, stop_threshold=stop_threshold, stop_iternum=stop_iternum)


    def _run_script(self, args_str):
        args = self._parser.parse_args(args_str.split(' '))
        # print('\n\nARGS: {}\n\n'.format(args))
        args.func(args)


if __name__ == "__main__":

    # ------------------------------------
    # flags to run test cases:
    # ------------------------------------

    TEST_NUM_TO_RUN = None

    CROP_HALF_OF_FRAME = 'left' # 'right' etc
    CROP_HALF_OF_FRAME = None

    TEST_2_FACES_FLAG = None
    if TEST_2_FACES_FLAG:
        TEST_NUM_TO_RUN = None
    else:
        CROP_HALF_OF_FRAME = None

    # ------------------------------------
    # reliable test case:
    # ------------------------------------
    model_name = 'emma_to_jade'
    faceit = FaceIt(model_name, 'emma', 'jade')

    # FaceIt.add_model(faceit)
    # faceit = FaceIt.MODELS[model_name]

    # ----------------------------------------------
    # getting instance attributes with values:
    # ----------------------------------------------
    
    ADD_DATA_FLAG = 1
    if ADD_DATA_FLAG:
        faceit.add_video('jade', 'jade_360_cut.mp4')
        faceit.add_video('emma', 'emma_360_cut.mp4')

    # ----------------------------------------------
    # getting instance attributes with values:
    # ----------------------------------------------
    
    DEBUG_NO_IMAGES = 0
    if DEBUG_NO_IMAGES:
    
        attrs = faceit.__dict__
        # print(attrs)
        for k in sorted(attrs.keys()):
            print('{}: {}'.format(k, attrs.get(k)))

    # ----------------------------------------------
    # Test Case: ADD_DATA_FLAG == 0 ==>

    # scripts.train.Train.get_images printouts:

        # ADD_DATA_FLAG == 0:
            # _faceswap: <__main__.FaceSwapInterface instance at 0x7f548315c320>
            # _name: emma_to_jade
            # _people: {'jade': {'photos': [], 'name': 'jade', 'videos': [], 'faces': 'data/persons/jade.jpg'}, 'emma': {'photos': [], 'name': 'emma', 'videos': [], 'faces': 'data/persons/emma.jpg'}}
            # _person_a: emma
            # _person_b: jade

            # Getting images ...
            # ImageDir: data/processed/model_data_emma_to_jade/emma
            # Error: data/processed/model_data_emma_to_jade/emma contains no images

        # ADD_DATA_FLAG == 1:
            # _faceswap: <__main__.FaceSwapInterface instance at 0x7f4df59f9320>
            # _name: emma_to_jade
            # _people: {'jade': {'photos': [], 'name': 'jade', 'videos': [{'url': None, 'name': 'jade_360_cut.mp4', 'fps': 20}], 'faces': 'data/persons/jade.jpg'}, 'emma': {'photos': [], 'name': 'emma', 'videos': [{'url': None, 'name': 'emma_360_cut.mp4', 'fps': 20}], 'faces': 'data/persons/emma.jpg'}}
            # _person_a: emma
            # _person_b: jade

            # Getting images ...
            # ImageDir: data/processed/model_data_emma_to_jade/emma
            # ImageDir: data/processed/model_data_emma_to_jade/jade
            # Model A Directory: data/processed/model_data_emma_to_jade/emma
            # Model B Directory: data/processed/model_data_emma_to_jade/jade
    # ----------------------------------------------

    # faceit.train(preview=True, stop_iternum=500, clean_data=False) # ==> Error: data/processed/model_data_emma_to_jade/emma contains no images
    
    # ----------------------------------------------
    # setting dirs manually:
    # ----------------------------------------------
        
    if TEST_NUM_TO_RUN == 1:
        dir_A = 'data/processed/emma_360_cut.mp4_faces'
        dir_B = 'data/processed/jade_360_cut.mp4_faces'
        
        faceit.train(dir_A=dir_A, dir_B=dir_B, preview=False) # works fine

    # ----------------------------------------------
    # using clean_data flag:
    # ----------------------------------------------

    if TEST_NUM_TO_RUN == 2:
        faceit.train(preview=False, clean_data=False) # works fine

    # ===============================================
    # test case with 2 faces:
    # ===============================================

    if TEST_2_FACES_FLAG:
        model_name = 'test_2_faces'
        faceit = FaceIt(model_name, 'p1', 'jade')

        # preprocessing:
        # dir_photo_1 = '/home/ubuntu/_code/ConProfileCombined/lib_deepfakes/_data/faces/2_64x/'
        # dir_photo_2 = '/home/ubuntu/_code/ConProfileCombined/lib_deepfakes/data/processed/jade_360_cut.mp4_faces/'

        # faceit.add_photos('p1', dir_photo_1)
        # faceit.add_photos('jade', dir_photo_2)

        faceit.add_video('p1', 'pair_360p_cut.mp4')
        faceit.add_video('jade', 'jade_360_cut.mp4')

        # faceit.preprocess()

        # faceit.train(dir_A=dir_photo_1, dir_B=dir_photo_2, preview=True, stop_iternum=500, clean_data=False) 
        # faceit.train(preview=True, stop_threshold=0.01, stop_iternum=100000)

        # faceit.convert(None, live=True)
        faceit.convert(None, live=True, face_filter=True)


    if TEST_NUM_TO_RUN == 0:
        faceit.convert(None, live=True)
