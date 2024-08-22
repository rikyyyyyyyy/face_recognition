const video = document.getElementById('video');
const canvas = document.getElementById('canvas');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

function processFrame() {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL('image/jpeg');
    sendFrameToServer(frame);
}

function sendFrameToServer(frame) {
    fetch('/process-frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ frame })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Prediction:', data);
    });
}

setInterval(processFrame, 1000 / 30);  // 30FPSでフレームを処理
