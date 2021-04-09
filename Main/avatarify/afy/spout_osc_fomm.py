import os, sys
from sys import platform as _platform
import glob
import yaml
import time
import requests

import numpy as np
import cv2
import imutils

from afy.videocaptureasync import VideoCaptureAsync
from afy.arguments import opt
from afy.utils import info, Once, Tee, crop, pad_img, resize, TicToc
import afy.camera_selector as cam_selector

from ndi import finder
from ndi import receiver
from ndi import lib

# Spout load library
from spout import Spout
#OSC
'''
setup instruction
for Install pythonosc
pip install python-osc

'''
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

#For Async processing
from pythonosc.osc_server import AsyncIOOSCUDPServer
import asyncio


log = Tee('./var/log/cam_fomm.log')

#NDI
reciever = None
recieveSource = None

#OSC
osc_dispatcher = None
osc_myserver = None
osc_client  = None


if _platform == 'darwin':
    if not opt.is_client:
        info('\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()


def is_new_frame_better(source, driving, predictor):
    global avatar_kp
    global display_string

    if avatar_kp is None:
        display_string = "No face detected in avatar."
        return False

    if predictor.get_start_frame() is None:
        display_string = "No frame to compare to."
        return True

    driving_smaller = resize(driving, (128, 128))[..., :3]
    new_kp = predictor.get_frame_kp(driving)

    if new_kp is not None:
        new_norm = (np.abs(avatar_kp - new_kp) ** 2).sum()
        old_norm = (np.abs(avatar_kp - predictor.get_start_frame_kp()) ** 2).sum()

        out_string = "{0} : {1}".format(int(new_norm * 100), int(old_norm * 100))
        display_string = out_string
        log(out_string)

        return new_norm < old_norm
    else:
        display_string = "No face found!"
        return False


def load_stylegan_avatar():
    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(url, headers={'User-Agent': "My User Agent 1.0"}).content

    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize(image, (IMG_SIZE, IMG_SIZE))

    return image


def load_images(IMG_SIZE = 256):
    avatars = []
    filenames = []
    images_list = sorted(glob.glob(f'{opt.avatars}/*'))
    for i, f in enumerate(images_list):
        if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'):
            img = cv2.imread(f)
            if img.ndim == 2:
                img = np.tile(img[..., None], [1, 1, 3])
            img = img[..., :3][..., ::-1]
            img = resize(img, (IMG_SIZE, IMG_SIZE))
            avatars.append(img)
            filenames.append(f)
    return avatars, filenames


def change_avatar(predictor, new_avatar):
    global avatar, avatar_kp, kp_source
    avatar_kp = predictor.get_frame_kp(new_avatar)
    kp_source = None
    avatar = new_avatar
    predictor.set_source_image(avatar)


def draw_rect(img, rw=0.6, rh=0.8, color=(255, 0, 0), thickness=2):
    h, w = img.shape[:2]
    l = w * (1 - rw) // 2
    r = w - l
    u = h * (1 - rh) // 2
    d = h - u
    img = cv2.rectangle(img, (int(l), int(u)), (int(r), int(d)), color, thickness)


def print_help():
    info('\n\n=== Control keys ===')
    info('1-9: Change avatar')
    for i, fname in enumerate(avatar_names):
        key = i + 1
        name = fname.split('/')[-1]
        info(f'{key}: {name}')
    info('W: Zoom camera in')
    info('S: Zoom camera out')
    info('A: Previous avatar in folder')
    info('D: Next avatar in folder')
    info('Q: Get random avatar')
    info('X: Calibrate face pose')
    info('I: Show FPS')
    info('ESC: Quit')
    info('\nFull key list: https://github.com/alievk/avatarify#controls')
    info('\n\n')


def draw_fps(frame, fps, timing, x0=10, y0=20, ystep=30, fontsz=0.5, color=(255, 255, 255)):
    frame = frame.copy()
    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y0 + ystep * 0), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Model time (ms): {timing['predict']:.1f}", (x0, y0 + ystep * 1), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Preproc time (ms): {timing['preproc']:.1f}", (x0, y0 + ystep * 2), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Postproc time (ms): {timing['postproc']:.1f}", (x0, y0 + ystep * 3), 0, fontsz * IMG_SIZE / 256, color, 1)
    return frame


def draw_calib_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "FIT FACE IN RECTANGLE", (40, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "W - ZOOM IN", (60, 40), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "S - ZOOM OUT", (60, 60), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk)
    return frame

#### - - - - - - - - -
##global
predictor = None
cur_ava = 0
avatar = None
avatar_names = None
is_calibrated = False
fps_hist = []
fps = 0

IMG_SIZE = 256



def select_camera(config):
    cam_config = config['cam_config']
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, 'r') as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config['cam_id']
    else:
        cam_frames = cam_selector.query_cameras(config['query_n_cams'])

        if cam_frames:
            if len(cam_frames) == 1:
                cam_id = list(cam_frames)[0]
            else:
                cam_id = cam_selector.select_camera(cam_frames, window="CLICK ON YOUR CAMERA")
            log(f"Selected camera {cam_id}")

            with open(cam_config, 'w') as f:
                yaml.dump({'cam_id': cam_id}, f)
        else:
            log("No cameras are available")

    return cam_id

#NDI functions
def initNdi():
    global reciever
    global recieveSource
    find = finder.create_ndi_finder()
    NDIsources = find.get_sources()

    # If there is one or more sources then list the names of all source.
    # If only 1 source is detected, then automatically connect to that source.
    # If more than 1 source detected, then list all sources detected and allow user to choose source.
    if(len(NDIsources) > 0):
        print(str(len(NDIsources)) + " NDI Sources Detected")
        for x in range(len(NDIsources)):
            print(str(x) + ". "+NDIsources[x].name + " @ "+str(NDIsources[x].address))

        #wait key for NDI source
        #ndiindx = int(input('Select NDI source [number] :'))

        if(len(NDIsources) == 1):
            #If only one source, connect to that source
            # recieveSource = NDIsources[ min(ndiindx, len(NDIsources) ) ]
            recieveSource = NDIsources[ 0 ]
            print("Automatically Connecting To Source...")
        else:
            awaitUserInput = True;
            while(awaitUserInput):
                print("")
                try:
                    key = int(input("Please choose a NDI Source Number to connect to:"))
                    if(key < len(NDIsources) and key >= 0):
                        awaitUserInput = False
                        recieveSource = NDIsources[key]
                    else:
                        print("Input Not A Number OR Number not in NDI Range. Please pick a number between 0 and "+ str(len(NDIsources)-1))
                except:
                    print("Input Not A Number OR Number not in NDI Range. Please pick a number between 0 and "+ str(len(NDIsources)-1))

            #If more than one source, ask user which NDI source they want to use
    else:
        print("No NDI Sources Detected - Please Try Again")

    print("Width Resized To 500px. Not Actual Source Size")
    reciever = receiver.create_receiver(recieveSource)

#loop for Avatarify
def loopAvatarify():
    global reciever #NDI
    global predictor
    global reciever
    global IMG_SIZE
    global is_calibrated

    global cur_ava
    global avatar
    global avatar_names
    global fps_hist
    global fps

    show_fps = True

    #????
    find_keyframe = False
    tt = TicToc()

    timing = {
        'preproc': 0,
        'predict': 0,
        'postproc': 0
    }

    spout.check()
    # receive data
    frame = spout.receive()
    #Here NDI process
    # caution, without update the NDI update images. We can not see the update
    # frame = reciever.read()

    stream_img_size = frame.shape[1], frame.shape[0]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)# Color convert
    green_overlay = False

    tt.tic()

    #Resize
    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0
    frame = frame[..., ::-1]
    frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x, offset_y=frame_offset_y)
    frame = resize(frame, (IMG_SIZE, IMG_SIZE))[..., :3]

    if find_keyframe:
        if is_new_frame_better(avatar, frame, predictor):
            log("Taking new frame!")
            green_overlay = True
            predictor.reset_frames()

    timing['preproc'] = tt.toc()

    # passthrough = False
    # if passthrough:
    #     out = frame
    # el
    if is_calibrated:
        tt.tic()
        out = predictor.predict(frame)
        if out is None:
            log('predict returned None')
        timing['predict'] = tt.toc()
    else:
        out = None
        log("not is_calibrated")

    tt.tic()

    key = cv2.waitKey(1)

    if cv2.getWindowProperty('cam', cv2.WND_PROP_VISIBLE) < 1.0:
        return

    # elif is_calibrated and cv2.getWindowProperty('avatarify_window1', cv2.WND_PROP_VISIBLE) < 1.0:
    #     return

    if key == 27: # ESC
        return
    elif key == ord('d'):
        cur_ava += 1
        if cur_ava >= len(avatars):
            cur_ava = 0
        passthrough = False
        change_avatar(predictor, avatars[cur_ava])
    elif key == ord('a'):
        cur_ava -= 1
        if cur_ava < 0:
            cur_ava = len(avatars) - 1
        passthrough = False
        change_avatar(predictor, avatars[cur_ava])
    elif key == ord('x'):
        predictor.reset_frames()

        if not is_calibrated:
            cv2.namedWindow('avatarify_window1', cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow('avatarify_window1', cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow('avatarify_window1', 600, 250)

        is_calibrated = True

    elif key == ord('i'):
        show_fps = not show_fps

    preview_frame = frame.copy()


    if green_overlay:
        green_alpha = 0.8
        overlay = preview_frame.copy()
        overlay[:] = (0, 255, 0)
        preview_frame = cv2.addWeighted( preview_frame, green_alpha, overlay, 1.0 - green_alpha, 0.0)

    timing['postproc'] = tt.toc()

    if find_keyframe:
        preview_frame = cv2.putText(preview_frame, display_string, (10, 220), 0, 0.5 * IMG_SIZE / 256, (255, 255, 255), 1)

    if show_fps:
        preview_frame = draw_fps(preview_frame, fps, timing)

    if not is_calibrated:
        preview_frame = draw_calib_text(preview_frame)

    if not opt.hide_rect:
        draw_rect(preview_frame)

    cv2.imshow('cam', preview_frame[..., ::-1])


    if out is not None:
        if not opt.no_pad:
            out = pad_img(out, stream_img_size)

        cv2.imshow('avatarify_window1', out[..., ::-1])
        spout.send(out)

    fps_hist.append(tt.toc(total=True))
    if len(fps_hist) == 10:
        fps = 10 / (sum(fps_hist) / 1000)
        fps_hist = []

#this
async def mainLoop():
    try:
        while True:
            loopAvatarify()
            await asyncio.sleep(0)

    except KeyboardInterrupt:
        log("main: user interrupt")

    log("main: exit")

async def init_main():
    global osc_client

    log("setting up osc_client")
    #setting up OSC
    ip = "127.0.0.1"
    port = 5005
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/message", osc_messege_handler)
    server = AsyncIOOSCUDPServer((ip, port), osc_dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5006)
    log("sini up osc_client")

    await mainLoop()  # Enter main loop of program

    transport.close()  # Clean up serve endpoint

def initAvatarify():
    global predictor
    global cur_ava
    global avatars
    global avatar_names
    global is_calibrated
    global fps
    global fps_hist

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    global display_string
    display_string = ""

    log('Loading Predictor')
    predictor_args = {
        'config_path': opt.config,
        'checkpoint_path': opt.checkpoint,
        'relative': opt.relative,
        'adapt_movement_scale': opt.adapt_scale,
        'enc_downscale': opt.enc_downscale
    }

    #loading local predictor
    from afy import predictor_local
    log('predictor_local')
    predictor = predictor_local.PredictorLocal(
        **predictor_args
    )

    #camera
    cam_id = select_camera(config)

    if cam_id is None:
        exit(1)

    #Avators (externel target image)
    avatars, avatar_names = load_images()

    enable_vcam = not opt.no_stream
    enable_vcam = False

    cur_ava = 0
    avatar = None
    #here important, this function will process keypoint
    change_avatar(predictor, avatars[cur_ava])
    passthrough = False

    cv2.namedWindow('cam', cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('cam', 250, 250)

    fps_hist = []
    fps = 0

    log('Finish initAvatarify')


#OSC message handle
def osc_messege_handler(unused_addr, *p):
    global predictor
    global is_calibrated
    global cur_ava
    global avatars

    try:
        print(p[0])
        if(p[0] == 'calibrate'):
            print('calibrate face pose')
            predictor.reset_frames()
            is_calibrated = True


        if(p[0] == 'nextFace'):
            print('switch to nextFace')
            if cur_ava < len(avatars) - 1:
                cur_ava += 1
            change_avatar(predictor, avatars[cur_ava])

        if(p[0] == 'prevFace'):
            print('switch to nextFace')
            if cur_ava > 0:
                cur_ava -= 1
            change_avatar(predictor, avatars[cur_ava])

        if(p[0] == 'switchFace'):
            print('switchFace')
            _index = int(p[1])

            if _index < len(avatars) and _index >= 0:
                cur_ava = _index
                change_avatar(predictor, avatars[cur_ava])

    # elif key == ord('a'):
    #     cur_ava -= 1
    #     if cur_ava < 0:
    #         cur_ava = len(avatars) - 1
    #     passthrough = False
    #     change_avatar(predictor, avatars[cur_ava])


        print(unused_addr)

        # client.send_message("/message_echo", p)

    except ValueError: pass

if __name__ == "__main__":

    print('---------start setting up Spout')

    # create spout object
    spout = Spout(silent = True)
    # create receiver
    spout.createReceiver('input')
    # create sender
    spout.createSender('output')
    print('---------finish setting up Spout')

    print('---------start setting up Avatarify')
    initAvatarify()
    print('---------finish setting up Avatarify')

    asyncio.run(init_main())

    cv2.destroyAllWindows()
    predictor.stop()

    log("main: exit")
    exit()
