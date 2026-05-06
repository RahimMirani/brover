# rover_web_control.py
#
# Low latency rover control + live camera webpage
# Run:
#   python3 rover_web_control.py
#
# Open on Mac:
#   http://RASPBERRYPI_IP:5000

from flask import Flask, Response, request, render_template_string
from gpiozero import OutputDevice
import subprocess
import threading
import time

app = Flask(__name__)

# ---------------------------------
# GPIO / L298N Pins
# ---------------------------------
IN1 = OutputDevice(17)   # Left motor
IN2 = OutputDevice(27)

IN3 = OutputDevice(22)   # Right motor
IN4 = OutputDevice(23)

# gpio ultra sonic sensor ports:
# gpio 24 is the trigger pin
# gpio 25 echo
## incase this does not work, swap the ports 
# make a compensation for sound and humidity 


# ---------------------------------
# Motor controls
# ---------------------------------
def stop_all():
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()

def forward():
    IN1.on();  IN2.off()
    IN3.on();  IN4.off()

def backward():
    IN1.off(); IN2.on()
    IN3.off(); IN4.on()

def right():
    # spin right
    IN1.off(); IN2.on()
    IN3.on();  IN4.off()

def left():
    # spin left
    IN1.on();  IN2.off()
    IN3.off(); IN4.on()


# ---------------------------------
# Safety watchdog
# Stops rover if commands stop coming
# ---------------------------------
last_command_time = time.time()

def watchdog():
    global last_command_time
    while True:
        if time.time() - last_command_time > 0.18:
            stop_all()
        time.sleep(0.03)

threading.Thread(target=watchdog, daemon=True).start()


# ---------------------------------
# Camera stream using rpicam-vid MJPEG
# ---------------------------------
camera_cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "--codec", "mjpeg",
    "-o", "-"
]

camera_process = subprocess.Popen(
    camera_cmd,
    stdout=subprocess.PIPE,
    bufsize=0
)

def generate_stream():
    buffer = b""
    while True:
        chunk = camera_process.stdout.read(4096)
        if not chunk:
            continue

        buffer += chunk

        start = buffer.find(b'\xff\xd8')
        end   = buffer.find(b'\xff\xd9')

        if start != -1 and end != -1 and end > start:
            jpg = buffer[start:end+2]
            buffer = buffer[end+2:]

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpg +
                b'\r\n'
            )


# ---------------------------------
# Web UI
# ---------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Rover Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {
    background:#111;
    color:white;
    text-align:center;
    font-family:Arial;
}
img {
    width:95%;
    max-width:900px;
    border-radius:12px;
    margin-top:10px;
}
.grid {
    display:grid;
    grid-template-columns:100px 100px 100px;
    gap:10px;
    justify-content:center;
    margin-top:20px;
}
button {
    font-size:28px;
    padding:18px;
    border:none;
    border-radius:10px;
}
</style>
</head>
<body>

<h2>Rover Live Control</h2>

<img src="/video_feed">

<div class="grid">
<div></div>
<button onmousedown="send('forward')" onmouseup="send('stop')">↑</button>
<div></div>

<button onmousedown="send('left')" onmouseup="send('stop')">←</button>
<button onmousedown="send('backward')" onmouseup="send('stop')">↓</button>
<button onmousedown="send('right')" onmouseup="send('stop')">→</button>
</div>

<p>Use WASD or Arrow Keys</p>

<script>
let activeKey = null;
let sendLoop = null;

function send(cmd){
    fetch('/move', {
        method:'POST',
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:'cmd=' + cmd
    });
}

function startCommand(cmd){
    if(activeKey === cmd) return;

    stopCommand();

    activeKey = cmd;

    send(cmd); // instant first send

    sendLoop = setInterval(() => {
        send(cmd);
    }, 60); // repeat every 60ms
}

function stopCommand(){
    if(sendLoop){
        clearInterval(sendLoop);
        sendLoop = null;
    }

    if(activeKey){
        send('stop');
        activeKey = null;
    }
}

document.addEventListener('keydown', function(e){
    if(e.repeat) return;

    if(e.key === 'w' || e.key === 'ArrowUp') startCommand('forward');
    if(e.key === 's' || e.key === 'ArrowDown') startCommand('backward');
    if(e.key === 'a' || e.key === 'ArrowLeft') startCommand('left');
    if(e.key === 'd' || e.key === 'ArrowRight') startCommand('right');
});

document.addEventListener('keyup', function(e){
    stopCommand();
});

window.addEventListener('blur', function(){
    stopCommand();
});
</script>

</body>
</html>
"""

# ---------------------------------
# Routes
# ---------------------------------
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/move", methods=["POST"])
def move():
    global last_command_time
    last_command_time = time.time()

    cmd = request.form.get("cmd", "stop")

    if cmd == "forward":
        forward()
    elif cmd == "backward":
        backward()
    elif cmd == "left":
        left()
    elif cmd == "right":
        right()
    else:
        stop_all()

    return "ok"


# ---------------------------------
# Main
# ---------------------------------
if __name__ == "__main__":
    stop_all()
    app.run(host="0.0.0.0", port=5000, threaded=True)
