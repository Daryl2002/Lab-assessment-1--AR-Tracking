let refImage, refKeypoints, refDescriptors;
let orb, bf;
let video, canvas, ctx, cap, src, gray;
let isTracking = false;
let streaming = false;

window.addEventListener('DOMContentLoaded', () => {
    const statusText = document.getElementById('statusText');
    const imageUpload = document.getElementById('imageUpload');
    const generateBtn = document.getElementById('generateBtn');
    const saveBtn = document.getElementById('saveBtn');
    const startARBtn = document.getElementById('startARBtn');
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');
    video = document.getElementById('webcamVideo');
    const arContainer = document.getElementById('ar-container');
    const outputCanvas = document.getElementById('outputCanvas');

    // Wait for OpenCV.js
    const checkOpenCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkOpenCV);
            statusText.innerText = "Status: OpenCV.js is ready. Please upload an image.";
            orb = new cv.ORB(500); // 500 features max
            bf = new cv.BFMatcher(cv.NORM_HAMMING, true); // Crosscheck true
        }
    }, 500);

    // 1. Upload Image
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0, img.width, img.height);
                    canvas.style.display = 'block';
                    refImage = img;
                    statusText.innerText = "Status: Image uploaded. Ready to generate features.";
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // 2. Generate Features
    generateBtn.addEventListener('click', () => {
        if (!refImage || !cv) return;
        statusText.innerText = "Status: Extracting features...";
        
        try {
            let imgMat = cv.imread(canvas);
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMat, imgGray, cv.COLOR_RGBA2GRAY);

            refKeypoints = new cv.KeyPointVector();
            refDescriptors = new cv.Mat();
            
            // Extract features
            orb.detectAndCompute(imgGray, new cv.Mat(), refKeypoints, refDescriptors);
            
            // Draw keypoints manually
            let outImg = new cv.Mat();
            cv.cvtColor(imgGray, outImg, cv.COLOR_GRAY2RGBA);
            let color = new cv.Scalar(0, 255, 0, 255);
            for (let i = 0; i < refKeypoints.size(); i++) {
                let p = refKeypoints.get(i).pt;
                cv.circle(outImg, new cv.Point(p.x, p.y), 3, color, -1);
            }
            cv.imshow(canvas, outImg);
            
            imgMat.delete(); imgGray.delete(); outImg.delete();

            statusText.innerText = "Status: Features generated! You can save the marker and start the AR camera.";
            saveBtn.classList.remove('disabled'); saveBtn.disabled = false;
            startARBtn.classList.remove('disabled'); startARBtn.disabled = false;
            
        } catch (err) {
            console.error("Feature extraction error:", err);
            statusText.innerText = "Status: Error extracting features.";
        }
    });

    // 3. Save Image
    saveBtn.addEventListener('click', () => {
        if (saveBtn.disabled) return;
        const link = document.createElement('a');
        link.download = 'ar-marker.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    });

    // 4. Start AR Camera
    startARBtn.addEventListener('click', () => {
        if (startARBtn.disabled) return;
        
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
                arContainer.style.display = 'block';
                statusText.innerText = "Status: Tracking started. Point camera at marker.";
                
                video.addEventListener('canplay', () => {
                    if (!streaming) {
                        outputCanvas.width = video.videoWidth;
                        outputCanvas.height = video.videoHeight;
                        src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
                        gray = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);
                        cap = new cv.VideoCapture(video);
                        streaming = true;
                        requestAnimationFrame(processVideo);
                    }
                });
            })
            .catch(function(err) {
                alert("Camera error: " + err); 
            });
    });

    function processVideo() {
        if (!streaming) return;
        
        // Setup memory management arrays to prevent memory leak crashes
        let matsToDelete = [];
        let matchesFound = 0;
        
        try {
            cap.read(src);
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            let frameKeypoints = new cv.KeyPointVector();
            let frameDescriptors = new cv.Mat();
            matsToDelete.push(frameKeypoints, frameDescriptors);
            
            // Extract features from current video frame
            orb.detectAndCompute(gray, new cv.Mat(), frameKeypoints, frameDescriptors);
            
            if (frameDescriptors.rows > 0 && refDescriptors.rows > 0) {
                let matches = new cv.DMatchVector();
                matsToDelete.push(matches);
                bf.match(refDescriptors, frameDescriptors, matches);
                
                // Filter good matches - ORB Hamming distances usually < 50 are excellent
                let goodMatches = [];
                for (let i = 0; i < matches.size(); i++) {
                    let m = matches.get(i);
                    // 65 is a standard threshold for ORB matching to avoid false positives
                    if (m.distance < 65) {
                        goodMatches.push(m);
                    }
                }
                
                matchesFound = goodMatches.length;
                
                // Need at least 4 points to find homography, 10 is safer
                if (matchesFound >= 10) {
                    let refPts = [];
                    let framePts = [];
                    for (let i = 0; i < goodMatches.length; i++) {
                        refPts.push(refKeypoints.get(goodMatches[i].queryIdx).pt.x);
                        refPts.push(refKeypoints.get(goodMatches[i].queryIdx).pt.y);
                        framePts.push(frameKeypoints.get(goodMatches[i].trainIdx).pt.x);
                        framePts.push(frameKeypoints.get(goodMatches[i].trainIdx).pt.y);
                    }
                    
                    let refMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, refPts);
                    let frameMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, framePts);
                    matsToDelete.push(refMat, frameMat);
                    
                    let H = cv.findHomography(refMat, frameMat, cv.RANSAC, 5.0);
                    
                    if (!H.empty()) {
                        matsToDelete.push(H);
                        
                        // Project a 3D looking Cube using Homography perspective transform
                        let w = refImage.width;
                        let h = refImage.height;
                        
                        let objCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [0,0, w,0, w,h, 0,h]);
                        let sceneCorners = new cv.Mat();
                        matsToDelete.push(objCorners, sceneCorners);
                        
                        cv.perspectiveTransform(objCorners, sceneCorners, H);
                        
                        // Draw bottom face (base of cube)
                        let colorBase = new cv.Scalar(255, 0, 0, 255); // Red base
                        let p1 = new cv.Point(sceneCorners.data32F[0], sceneCorners.data32F[1]);
                        let p2 = new cv.Point(sceneCorners.data32F[2], sceneCorners.data32F[3]);
                        let p3 = new cv.Point(sceneCorners.data32F[4], sceneCorners.data32F[5]);
                        let p4 = new cv.Point(sceneCorners.data32F[6], sceneCorners.data32F[7]);
                        cv.line(src, p1, p2, colorBase, 4);
                        cv.line(src, p2, p3, colorBase, 4);
                        cv.line(src, p3, p4, colorBase, 4);
                        cv.line(src, p4, p1, colorBase, 4);
                        
                        // Pseudo-3D Height extrusion
                        let heightOffset = -150; // pixels 'up' in 2D space
                        let p5 = new cv.Point(p1.x, p1.y + heightOffset);
                        let p6 = new cv.Point(p2.x, p2.y + heightOffset);
                        let p7 = new cv.Point(p3.x, p3.y + heightOffset);
                        let p8 = new cv.Point(p4.x, p4.y + heightOffset);
                        
                        // Draw top face
                        let colorTop = new cv.Scalar(0, 0, 255, 255); // Blue top
                        cv.line(src, p5, p6, colorTop, 4);
                        cv.line(src, p6, p7, colorTop, 4);
                        cv.line(src, p7, p8, colorTop, 4);
                        cv.line(src, p8, p5, colorTop, 4);
                        
                        // Draw vertical edges
                        let colorEdge = new cv.Scalar(0, 255, 0, 255); // Green sides
                        cv.line(src, p1, p5, colorEdge, 4);
                        cv.line(src, p2, p6, colorEdge, 4);
                        cv.line(src, p3, p7, colorEdge, 4);
                        cv.line(src, p4, p8, colorEdge, 4);
                    }
                }
            }
            
            // Draw debug text on video feed to prove tracking algorithm is running
            cv.putText(src, "Matches found: " + matchesFound + " (Need >10)", new cv.Point(10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(255,255,0,255), 2);
            
            // Render the final frame
            cv.imshow('outputCanvas', src);
            
        } catch (err) {
            console.error("Video processing error:", err);
            // Don't crash the loop, just keep attempting next frame
        } finally {
            // Guarantee cleanup of OpenCV memory to prevent browser tab crash!
            matsToDelete.forEach(m => m.delete());
            requestAnimationFrame(processVideo);
        }
    }
});
