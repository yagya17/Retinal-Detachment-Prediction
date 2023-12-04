# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 30 17:38:23 2023

# @author: yagya
# """

# import streamlit as st
# import cv2
# import pickle 
# import numpy as np
# from PIL import Image
# from keras.models import load_model

# loaded_model = load_model('trainedmodel.h5')
# def predict(eye_img):
#     prediction=loaded_model.predict(eye_img)
#     if(prediction[0][1]>=0.5):
#         return "Retinal Detachment present"
#     else:
#        return "Retinal Detachment not present"
   
    
# def main():
    
#     st.title('Retinal Detachment Prediction')
#     input_img = st.file_uploader("Please upload fundus image of the eye", type=["jpg", "jpeg"])
#     if(input_img is None):
#         st.warning("Please upload fundus image of the eye")
#     else:
#         diagnosis = ''
        
#         if st.button("Check"):
#             pil_image=Image.open(input_img)
#             img_array = np.array(pil_image)
#             resized_img=cv2.resize(img_array, (224, 224))
#             preprocessed_img = np.expand_dims(resized_img, axis=0)
#             diagnosis = predict(preprocessed_img)
        
#         st.success(diagnosis)
    

# if __name__=='__main__':
#     main()
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:38:23 2023

@author: yagya
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='trainedmodel.tflite')
interpreter.allocate_tensors()

def predict(eye_img):
    # Preprocess the image if needed
    # ...

    # Perform inference
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], eye_img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    if prediction[0][1] >= 0.5:
        return "Retinal Detachment present"
    else:
        return "Retinal Detachment not present"

def main():
    st.title('Retinal Detachment Prediction')
    input_img = st.file_uploader("Please upload fundus image of the eye", type=["jpg", "jpeg"])
    
    if input_img is None:
        st.warning("Please upload fundus image of the eye")
    else:
        diagnosis = ''
        
        if st.button("Check"):
            pil_image = Image.open(input_img)
            img_array = np.array(pil_image)
            resized_img = cv2.resize(img_array, (224, 224))
            preprocessed_img = np.expand_dims(resized_img, axis=0)
            diagnosis = predict(preprocessed_img)
        
        st.success(diagnosis)

if __name__ == '__main__':
    main()
