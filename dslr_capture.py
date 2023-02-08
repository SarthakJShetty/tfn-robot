from threading import Thread
import gphoto2 as gp
import os
import cv2

class VideoRecorder:
    def __init__(self, save_dir=os.getcwd()):
        self.camera = gp.Camera()
        self.camera.init()
        self.recording = False
        self.watch_thread = None
        self.save_dir = save_dir

    def start_movie(self):
        if self.recording:
            print("already recording, ignoring...")
            return
        self.recording = True
        cfg = self.camera.get_config()
        movie_cfg = cfg.get_child_by_name("movie")
        movie_cfg.set_value(1)
        self.camera.set_config(cfg)
        self.watch_thread = Thread(target=self._thread)
        self.watch_thread.start()

    def stop_movie(self):
        self.recording = False
        self.watch_thread.join()
        self.watch_thread = None

        cfg = self.camera.get_config()    
        movie_cfg = cfg.get_child_by_name("movie")
        movie_cfg.set_value(0)
        self.camera.set_config(cfg)

        print("STOPPING MOVIE")

        timeout=2000
        while True:
            event_type, event_data = self.camera.wait_for_event(timeout)
            if event_type == gp.GP_EVENT_CAPTURE_COMPLETE:
                print("GP_EVENT_CAPTURE_COMPLETE")
            if event_type == gp.GP_EVENT_FILE_ADDED:
                cam_file = self.camera.file_get(
                    event_data.folder, event_data.name, gp.GP_FILE_TYPE_NORMAL
                )
                target_path = os.path.join(self.save_dir, event_data.name)
                print("Image is being saved to {}".format(target_path))
                cam_file.save(target_path)
                break
            if event_type == gp.GP_EVENT_FILE_CHANGED:
                print("GP_EVENT_FILE_CHANGED")
            if event_type == gp.GP_EVENT_FOLDER_ADDED:
                print("GP_EVENT_FOLDER_ADDED")
            if event_type == gp.GP_EVENT_TIMEOUT:
                print("GP_EVENT_TIMEOUT")
            if event_type == gp.GP_EVENT_UNKNOWN:
                # print("GP_EVENT_UNKNOWN")
                pass
            else:
                print(event_type)

    def exit(self):
        self.camera.exit()

    def _thread(self):
        timeout = 10000

        while self.recording:
            event_type, event_data = self.camera.wait_for_event(timeout)
            if event_type == gp.GP_EVENT_CAPTURE_COMPLETE:
                print("GP_EVENT_CAPTURE_COMPLETE")
            if event_type == gp.GP_EVENT_FILE_ADDED:
                print("GP_EVENT_FILE_ADDED")
            if event_type == gp.GP_EVENT_FILE_CHANGED:
                print("GP_EVENT_FILE_CHANGED")
            if event_type == gp.GP_EVENT_FOLDER_ADDED:
                print("GP_EVENT_FOLDER_ADDED")
            if event_type == gp.GP_EVENT_TIMEOUT:
                print("GP_EVENT_TIMEOUT")
            if event_type == gp.GP_EVENT_UNKNOWN:
                # print("GP_EVENT_UNKNOWN")
                pass
            else:
                print(event_type)

class DoneCam:
    def __init__(self, index = 0, save_dir = os.getcwd()):
        '''This basically points to the OpenCV index for the camera'''
        self.index = index
        self.cameraObj = cv2.VideoCapture(self.index)
        self.save_dir = save_dir
        self.frames_captured = 0
        self.imgSize = (int(self.cameraObj.get(3)), int(self.cameraObj.get(4)))
        self.videoSaver = cv2.VideoWriter(os.path.join(save_dir, 'doneCam.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, self.imgSize)
    def capture(self):
        _, image = self.cameraObj.read()
        self.videoSaver.write(image)
        self.frames_captured+=1
    def done(self):
        print('Captured {} frames'.format(self.frames_captured))
        self.cameraObj.release()
        self.videoSaver.release()

if __name__ == '__main__':
    camera = VideoRecorder()
    camera.start_movie()
    while True:
        try:
            print('capturing frame')
        except KeyboardInterrupt:
            break
    camera.stop_movie()
    camera.exit()