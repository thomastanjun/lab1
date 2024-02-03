import gradio as gr
import cv2
import numpy as np

def sky(img):

    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    '''
    # Print the region's mean hsv value    
    left = (1400, 100)  
    right = (1500, 200)  
    
    h,l, _ =img.shape
    region_hsv = img_hsv[left[1]:right[1], left[0]:right[0]]

    print("imageshape", h, l)    
    print("Hue: ", np.mean(region_hsv[:, :, 0]))
    print("Saturation: ", np.mean(region_hsv[:, :, 1]))
    print("Value: ", np.mean(region_hsv[:, :, 2]))
    '''
    
    # Threshold on the color range of sky
    lower_thresh = np.array([90, 20, 160])
    upper_thresh = np.array([190, 230, 255])
    sky_region = cv2.inRange(img_hsv, lower_thresh, upper_thresh)

    # Morphological closing and opening to erase noise in the background and the sky region 
    kernel = np.ones((5, 5), np.uint8)
    background_cleaned = cv2.morphologyEx(sky_region, cv2.MORPH_CLOSE, kernel)
    sky_cleaned = cv2.morphologyEx(background_cleaned, cv2.MORPH_OPEN, kernel)

    # Use the sky region to the original image
    result = cv2.bitwise_and(img, img, mask=sky_cleaned)
    
    return result
    
demo = gr.Interface(
    fn=sky,
    inputs=["image"],
    outputs=["image"],
)

demo.launch()
