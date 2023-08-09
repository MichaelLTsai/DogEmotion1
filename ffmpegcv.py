import ffmpegcv
FILE = "DogAction5.mp4"
cap = ffmpegcv.VideoCapture(FILE)



print("if we capture the mp4?", cap.isOpen())

