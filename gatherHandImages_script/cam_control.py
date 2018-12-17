import cv2

class CamControl:
    def __init__(self, res=(320,240)):
        self.res = res
        self.cam = cv2.VideoCapture(-1)
        self.cam.set(3, self.res[0]) # Width
        self.cam.set(4, self.res[1]) # Height
        assert self.cam.isOpened(), "Camera not opened correctly"

        # For saving video, comment out if there is an issue.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_out = cv2.VideoWriter('output.avi', fourcc, 20.0, (128, 128))


    def snap(self, res=None):
        """

        :param res: Tuple, shape of required image in height x width
        :return: ndarray: height x width
        """
        if res is None:
            res = self.res
        ret, frame = self.cam.read()

        if not ret:
            raise AttributeError("Ret returned by camera read")

        # Trim image from both sides of the longest dimension so that it is square
        blubber = int((max(frame.shape[0:-1]) - min(frame.shape[0:-1])) / 2)
        cropped_frame = frame[:,blubber:-blubber,:]

        # Resize frame to appropriate size
        frame = cv2.resize(cropped_frame, dsize=res, interpolation=cv2.INTER_CUBIC)

        # Make frame grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame


    def render(self, res=None):
        """
        Same as the snap() function, except that it also renders the image onto a window
        :param res: Tuple, shape of required image in height x width
        :return: ndarray: height x width
        """
        if res is None:
            res = self.res

        # Take an image
        img = self.snap(res)

        # Render iamge
        cv2.imshow("webcam", img)

        # Opencv necessary stuff (listens for exit key)
        if cv2.waitKey(1) == 27:
            return None

        return img

    def close(self):
        self.cam.release()
        self.video_out.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # Test
    cam = CamControl()

    cv2.imshow('image', cam.snap())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cam.close()
