import cv2
from flask import Flask, Response

app = Flask(__name__)

# Change '0' if you have multiple webcams connected
camera_index = 0
camera = cv2.VideoCapture(camera_index)

# Optional adjustments for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Yield the output frame in the multipart content type format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Listen on all local interfaces (0.0.0.0) so WSL can access it
    app.run(host='0.0.0.0', port=5000, threaded=True)
