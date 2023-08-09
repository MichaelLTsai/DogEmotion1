import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import torchvision.transforms as T
from PIL import Image
from groupfdogvideo2text import video2text, chatgpt
import tempfile


RESNET_FILE = "model_all.pt"
# print("Password/Enpoint IP for localtunnel is:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

# import model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
yolov8model = YOLO("yolov8n.pt")
model = torch.load(RESNET_FILE, map_location=device)

# construct transformer
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# normalization function
transform_norm = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean, std)])

# Construct Stremlit GUI
st.write('# AI Pet Emotional Translator ')

# set header
st.header('Please upload a video or picture of your dog! ')

# Variables
# emotions = ('I am pissed off😡', 'You make my day😸', 'I am feeling down 😥', 'snore zzzz😌')
emotions = ('Angry', 'Happy', 'Sad', 'Relaxed')
class_names = yolov8model.names

# upload file UI
file = st.file_uploader(label='Your picture: ', type=['jpeg', 'jpg', 'png', 'mp4'])

# display image
if file is not None:

    ##### mp4 file ##########
    if ".mp4" in file.name:
        predictions_dict = {}
        # Video Caption
        caption = video2text(file)
        # st.write("## {}".format(caption))

        # Video YOLO + Resnet
        # Method 1
        # tfile = tempfile.NamedTemporaryFile(delete=False)
        # tfile.write(file.read())
        # cap = cv2.VideoCapture(tfile.name)

        # Method 2
        vid = file.name
        with open(vid, 'wb') as f:
            f.write(file.getbuffer())
        cap = cv2.VideoCapture(vid)

        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # st.write("## fps1:{}".format(fps))
        count = 0
        if not cap.isOpened():
            cap = cv2.VideoCapture("DogAction5.mp4")
            st.write("## -----Changing File Location-------")
        # YOLO + Resnet
        while cap.isOpened():
            st.write("## Analyzing file....")
            ret, frame = cap.read()
            # run detection every one second
            if count % 8 == 0:
                # Make detections
                results = yolov8model(frame, verbose=False)
                img_np = results[0].orig_img
                boxes = results[0].boxes
                # st.write("## checkpoint1- enter prediction entry every 14 frames")
                # st.write("## count:{}".format(count))
                for box in boxes:
                    class_index = box.cls
                    prediction = class_names[int(class_index)]
                    conf_score = float(box.conf)
                    if prediction == "dog" and conf_score >= 0.5:
                        # st.write("## checkpoint2- enter prediction = dog")
                        # Get dog object bonding box xyxy
                        dog_box = box.xyxy.cpu().numpy().astype(int)
                        x1, x2, y1, y2 = dog_box[0][0], dog_box[0][2], dog_box[0][1], dog_box[0][3]
                        crop_np = img_np[y1:y2, x1:x2]
                        crop_np = crop_np.copy()
                        x = transform_norm(crop_np).float()
                        x = x.to(device)
                        x = x.unsqueeze(0)
                        with torch.no_grad():
                            model.eval()
                            scores = model(x)
                            predictions = scores.data.cpu().numpy().argmax()
                            if predictions not in predictions_dict:
                                predictions_dict[predictions] = 1
                            else:
                                predictions_dict[predictions] += 1
            count += 1
            cap.release()
        if predictions_dict:
            final_prediction = emotions[max(predictions_dict)]
            ## Chatgpt
            answer = chatgpt(caption, final_prediction)
            st.write("## Here is what you dog want to tell you: \n{}".format(answer))
        else:
            st.write("## Sorry, We cannot detect dog in your video. Please try another file and re-upload")



    ##### image file ##########
    else:
        image = Image.open(file).convert('RGB')
        ######### Run YOLO & Resnet to get prediction from uploaded image ###############
        results = yolov8model(image)
        boxes = results[0].boxes
        img_np = results[0].orig_img
        for box in boxes:
            class_index = box.cls
            prediction = class_names[int(class_index)]
            conf_score = float(box.conf)
            if prediction == "dog" and conf_score >= 0.5:
                # Get dog object bonding box xyxy
                dog_box = box.xyxy.cpu().numpy().astype(int)
                x1, x2, y1, y2 = dog_box[0][0], dog_box[0][2], dog_box[0][1], dog_box[0][3]
                crop_np = img_np[y1:y2, x1:x2]
                crop_np = crop_np.copy()
                x = transform_norm(crop_np).float()
                x = x.to(device)
                x = x.unsqueeze(0)
                with torch.no_grad():
                    model.eval()
                    scores = model(x)
                    predictions = scores.data.cpu().numpy().argmax()

        ######### Display uploaded image & prediction streamlit UI ###############
        st.image(image, use_column_width=True)
        if predictions is None:
            st.write("## Sorry, We cannot detect dog in your video. Please try another file and re-upload")
        else:
            # classify image
            final_prediction = emotions[predictions]

            # write classification
            caption = "You and your owner are in the same space."
            # Chatgpt
            answer = chatgpt(caption, final_prediction)
            st.write("## Here is what you dog want to tell you: \n# *{}*".format(answer))
else:
    st.write("## Please upload your file")

