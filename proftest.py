import cv2
import numpy as np

def main ():
    print ("hola")

    image = np.zeros((480,640,3),dtype="uint8")

    print(image.shape,image.dtype)
    width = image.shape[1]
    print("width:",width)

    image[:,:,0] = 255
    image[:100,30:60] = (0,255,255) #yelloww

    subimage = np.copy(image[:200,10:100,:])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    gray_c = np.expand_dims(gray, -1)
    print(gray_c.shape)
    gray_b_c =  np.expand_dims(gray_c,0)


    shades = np.zeros((480,256),dtype="uint8")
    grayrange= np.arange(256,dtype="uint8")
    image[:30,:30,2] = 255
    shades[:] = grayrange

    float_image = shades.astype("float64")

    shades = cv2.convertScaleAbs(float_image)

    cv2.imshow("My Image", image)
    cv2.imshow("subimage", subimage)  
    cv2.imshow("gray", gray)
    cv2.imshow("Float_image", float_image)  
    cv2.imshow("shades", shades) 
   ## cv2.waitKey(-1)

    capture = cv2.VideoCapture("images/noice.mp4")

    while key == -1:
        _, image = capture.read()

        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cv2.imshow("NOICE",noice.mp4)


    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()