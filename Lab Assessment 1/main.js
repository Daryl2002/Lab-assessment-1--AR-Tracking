let refImage, refKeypoints, refDescriptors;
let orb, bf;
let canvas, ctx;
let video, cap, src, gray;
let streaming = false;

// Three.js variables
let scene, camera, renderer, cube;
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
        scene = new THREE.Scene();
        const container = arOverlay.parentElement;
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        arOverlay.innerHTML = '';
        arOverlay.appendChild(renderer.domElement);

        // Create a stylish cube - smaller scale
        const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
        const material = new THREE.MeshPhongMaterial({ 
            color: 0x4f46e5, 
            transparent: true, 
            opacity: 0.8,
            specular: 0x111111,
            shininess: 100
        });
        cube = new THREE.Mesh(geometry, material);
        scene.add(cube);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 10, 10);
        scene.add(directionalLight);

        cube.visible = false;
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
                    // Don't hide ar-view, just keep it ready
                    refImage = img;
                    statusText.innerText = "Status: Image Loaded. Click Generate Features.";
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
            .then(function(stream) {
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
        cube.visible = false; // Reset each frame to avoid ghosting
        
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
                    matsToDelete.push(refMat, frameMat);
                    
                    let H = cv.findHomography(refMat, frameMat, cv.RANSAC, 5.0);
                    
                    if (!H.empty()) {
                        matsToDelete.push(H);
                        
                        let w = refImage.width;
                        let h = refImage.height;
                        let objCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [0,0, w,0, w,h, 0,h]);
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

                        cube.visible = true;
                        cube.position.x = (cx / video.videoWidth) * 2 - 1;
                        cube.position.y = -((cy / video.videoHeight) * 2 - 1);
                        cube.position.z = -2;
                        
                        cube.rotation.y += 0.05;
                        cube.rotation.x += 0.02;
                    }
                }
            }
            
            // Render to clear/update
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
