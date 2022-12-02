import gradio as gr
import tensorflow as tf
import cv2
import numpy as np


# Loading the Machine Learning model
model = tf.keras.models.load_model("models/modelv1.h5")


# Classifying the Image

def classify_image(input_image):
    opencvImage = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]

    if p == 0:
        p = 'Glioma Tumor'
    elif p == 1:
        p = "No Tumor"
    elif p == 2:
        p = 'Meningioma Tumor'
    else:
        p = 'Pituitary Tumor'
    if p != 1:
        # p = "Not able to detect anything from given image."
        print(f'The Model predicts that it is a {p}')

    
    return p

# Submiting the report throught SMS




# Building the Interface
with gr.Blocks() as demo:

    # Image inputs and outputs widget
    with gr.Row():
        image = gr.inputs.Image(shape=(224, 224))
        label = gr.outputs.Label(num_top_classes=4)

    # Submit button for classiying the image
    with gr.Row():
        submit_btn = gr.Button(value="Submit")
        submit_btn.click(classify_image, inputs=[image], outputs=[
                         label], show_progress=True)


# Main Function
if __name__ == "__main__":
    demo.launch()
