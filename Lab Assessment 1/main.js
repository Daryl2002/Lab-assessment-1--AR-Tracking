let refImage, refKeypoints, refDescriptors;
let orb, bf;
let canvas, ctx;
let video, cap, src, gray;
let streaming = false;

// Three.js variables
let scene, camera, renderer, cube;
let currentObject; // Changed to match any 3D object
let markerCount = 0; // Track marker uploads for Requirement 3
const arOverlay = document.getElementById('ar-overlay');

window.addEventListener('DOMContentLoaded', () => {
    const statusText = document.getElementById('statusText');
    const imageUpload = document.getElementById('imageUpload');
    const generateBtn = document.getElementById('generateBtn');
    const saveBtn = document.getElementById('saveBtn');
    const startARBtn = document.getElementById('startARBtn');

    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');
    video = document.getElementById('webcamVideo');
    const arView = document.getElementById('ar-view');
    const outputCanvas = document.getElementById('outputCanvas');

    // Initialize Three.js
    function initThreeJS() {
        if (scene) {
            // Clean up old scene if it exists (for switching markers)
            while(scene.children.length > 0){ 
                scene.remove(scene.children[0]); 
            }
        } else {
            scene = new THREE.Scene();
        }

        const container = arOverlay.parentElement;
        if (!renderer) {
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            arOverlay.innerHTML = '';
            arOverlay.appendChild(renderer.domElement);
        }

        // Cycle through 3 different objects to fulfill Requirement 3
        let geometry;
        let color;
        const index = markerCount % 3;

        if (index === 0) {
            geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
            color = 0x4f46e5; // Indigo
        } else if (index === 1) {
            geometry = new THREE.SphereGeometry(0.3, 32, 32);
            color = 0x10b981; // Emerald
        } else {
            geometry = new THREE.CylinderGeometry(0.25, 0.25, 0.5, 32);
            color = 0xf59e0b; // Amber
        }

        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.8,
            specular: 0x111111,
            shininess: 100
        });

        currentObject = new THREE.Mesh(geometry, material);
        scene.add(currentObject);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 10, 10);
        scene.add(directionalLight);

        currentObject.visible = false;
    }

    // Wait for OpenCV.js
    const checkOpenCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkOpenCV);
            statusText.innerText = "Status: Engine Ready. Upload an image.";
            orb = new cv.ORB(500);
            bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
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
                    // Increment marker count and update 3D object for Requirement 3
                    markerCount++;
                    if (streaming) {
                        initThreeJS();
                    }
                    
                    statusText.innerText = "Status: Image " + markerCount + " Loaded. Click Generate.";
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // 2. Generate Features
    generateBtn.addEventListener('click', () => {
        if (!refImage || !cv) return;
        statusText.innerText = "Status: Processing Features...";

        try {
            let imgMat = cv.imread(canvas);
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMat, imgGray, cv.COLOR_RGBA2GRAY);

            refKeypoints = new cv.KeyPointVector();
            refDescriptors = new cv.Mat();
            orb.detectAndCompute(imgGray, new cv.Mat(), refKeypoints, refDescriptors);

            // Draw features
            let outImg = new cv.Mat();
            cv.cvtColor(imgGray, outImg, cv.COLOR_GRAY2RGBA);
            let color = new cv.Scalar(16, 185, 129, 255); // Emerald Green
            for (let i = 0; i < refKeypoints.size(); i++) {
                let p = refKeypoints.get(i).pt;
                cv.circle(outImg, new cv.Point(p.x, p.y), 3, color, -1);
            }
            cv.imshow(canvas, outImg);

            imgMat.delete(); imgGray.delete(); outImg.delete();

            statusText.innerText = "Status: Features Ready! You can start AR.";
            saveBtn.disabled = false;
            startARBtn.disabled = false;

        } catch (err) {
            console.error(err);
            statusText.innerText = "Status: Feature Error.";
        }
    });

    // 3. Save Image
    saveBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = 'marker-features.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    });

    // 4. Start AR Camera
    startARBtn.addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
            .then(function (stream) {
                video.srcObject = stream;
                video.play();
                arView.classList.add('active'); // Use class for appearance
                statusText.innerText = "Status: Tracking Active. Find the marker.";

                video.addEventListener('canplay', () => {
                    if (!streaming) {
                        outputCanvas.width = video.videoWidth;
                        outputCanvas.height = video.videoHeight;
                        src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
                        gray = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);
                        cap = new cv.VideoCapture(video);
                        initThreeJS();
                        streaming = true;
                        requestAnimationFrame(processVideo);
                    }
                });
            })
            .catch(err => alert("Camera error: " + err));
    });

    function processVideo() {
        if (!streaming) return;

        let matsToDelete = [];
        currentObject.visible = false; // Reset each frame to avoid ghosting

        try {
            cap.read(src);
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

            let frameKeypoints = new cv.KeyPointVector();
            let frameDescriptors = new cv.Mat();
            matsToDelete.push(frameKeypoints, frameDescriptors);

            orb.detectAndCompute(gray, new cv.Mat(), frameKeypoints, frameDescriptors);

            if (frameDescriptors.rows > 0 && refDescriptors.rows > 0) {
                let matches = new cv.DMatchVector();
                matsToDelete.push(matches);
                bf.match(refDescriptors, frameDescriptors, matches);

                let goodMatches = [];
                for (let i = 0; i < matches.size(); i++) {
                    let m = matches.get(i);
                    if (m.distance < 65) goodMatches.push(m);
                }

                if (goodMatches.length >= 20) {
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
                    let mask = new cv.Mat();
                    matsToDelete.push(refMat, frameMat, mask);
                    
                    let H = cv.findHomography(refMat, frameMat, cv.RANSAC, 5.0, mask);
                    
                    let inliers = 0;
                    if (!H.empty()) {
                        for (let i = 0; i < mask.rows; i++) {
                            if (mask.data[i]) inliers++;
                        }
                    }
                    
                    if (!H.empty() && inliers >= 15) {
                        matsToDelete.push(H);

                        let w = refImage.width;
                        let h = refImage.height;
                        let objCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, w, 0, w, h, 0, h]);
                        let sceneCorners = new cv.Mat();
                        matsToDelete.push(objCorners, sceneCorners);

                        cv.perspectiveTransform(objCorners, sceneCorners, H);

                        // Centroid calculation
                        let cx = 0, cy = 0;
                        for(let i=0; i<8; i+=2) {
                            cx += sceneCorners.data32F[i];
                            cy += sceneCorners.data32F[i+1];
                        }
                        cx /= 4; cy /= 4;

                        // Polygon validation
                        let px0 = sceneCorners.data32F[0], py0 = sceneCorners.data32F[1];
                        let px1 = sceneCorners.data32F[2], py1 = sceneCorners.data32F[3];
                        let px2 = sceneCorners.data32F[4], py2 = sceneCorners.data32F[5];
                        let px3 = sceneCorners.data32F[6], py3 = sceneCorners.data32F[7];

                        let crossProduct = (x1, y1, x2, y2) => x1 * y2 - y1 * x2;

                        let cross0 = crossProduct(px1 - px0, py1 - py0, px2 - px1, py2 - py1);
                        let cross1 = crossProduct(px2 - px1, py2 - py1, px3 - px2, py3 - py2);
                        let cross2 = crossProduct(px3 - px2, py3 - py2, px0 - px3, py0 - py3);
                        let cross3 = crossProduct(px0 - px3, py0 - py3, px1 - px0, py1 - py0);

                        let isConvex = (cross0 > 0 && cross1 > 0 && cross2 > 0 && cross3 > 0) || 
                                       (cross0 < 0 && cross1 < 0 && cross2 < 0 && cross3 < 0);

                        let area = 0.5 * Math.abs(crossProduct(px2 - px0, py2 - py0, px3 - px1, py3 - py1));
                        let minArea = (video.videoWidth * video.videoHeight) * 0.005;

                        if (isConvex && area > minArea && cx >= -video.videoWidth && cx <= video.videoWidth * 2 && cy >= -video.videoHeight && cy <= video.videoHeight * 2) {
                            let vFov = camera.fov * Math.PI / 180;
                            let planeHeight = 2 * Math.tan(vFov / 2) * 2;
                            let planeWidth = planeHeight * camera.aspect;

                            currentObject.visible = true;
                            currentObject.position.x = ((cx / video.videoWidth) * 2 - 1) * (planeWidth / 2);
                            currentObject.position.y = -((cy / video.videoHeight) * 2 - 1) * (planeHeight / 2);
                            currentObject.position.z = -2;
                            currentObject.rotation.set(0, 0, 0);
                        }
                    }
                }
            }

            renderer.clear();
            renderer.render(scene, camera);
            cv.imshow('outputCanvas', src);

        } catch (err) {
            console.error(err);
        } finally {
            matsToDelete.forEach(m => m.delete());
            requestAnimationFrame(processVideo);
        }
    }
});
