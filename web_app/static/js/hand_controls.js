document.addEventListener("DOMContentLoaded", () => {
  const videoElement = document.getElementById("webcam-video");
  const canvasElement = document.getElementById("output-canvas");
  const canvasCtx = canvasElement.getContext("2d");
  const toggleBtn = document.getElementById("toggle-hands-btn");
  const feedback = document.getElementById("gesture-feedback");

  let isTracking = false;
  let camera = null;
  let hands = null;

  // Gesture State
  let isPinched = false;
  let lastPinchX = null;
  let lastPinchY = null;

  // Zoom State (Rotation Based)
  let lastHandAngle = null;

  // Thresholds
  const PINCH_THRESHOLD = 0.1; // Increased sensitivity (finger distance approx ~10% of screen width)
  const ROTATION_SENSITIVITY = 2.5;
  const ZOOM_ENSITIIVTY = 2.0; // Multiplier for angle delta

  // Helper: Calculate angle of vector between two points (in radians)
  function calculateAngle(p1, p2) {
    return Math.atan2(p2.y - p1.y, p2.x - p1.x);
  }

  toggleBtn.addEventListener("click", toggleHandControl);

  function onResults(results) {
    // Draw Landmarks
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height,
    );

    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 2,
        });
        drawLandmarks(canvasCtx, landmarks, {
          color: "#FF0000",
          lineWidth: 1,
          radius: 2,
        });
      }
    }
    canvasCtx.restore();

    if (!window.viewer) return;

    const landmarks = results.multiHandLandmarks;

    if (landmarks && landmarks.length > 0) {
      const lm = landmarks[0];
      const indexTip = lm[8];
      const thumbTip = lm[4];
      const wrist = lm[0];
      const middleMCP = lm[9]; // Middle finger base

      // --- 1. PINCH DETECTION (Rotate) ---
      // Distance between Index Tip and Thumb Tip
      const pinchDist = Math.hypot(
        indexTip.x - thumbTip.x,
        indexTip.y - thumbTip.y,
      );

      if (pinchDist < PINCH_THRESHOLD) {
        // PINCHED (Grabbing) -> Rotate Structure
        if (!isPinched) {
          isPinched = true;
          lastPinchX = indexTip.x;
          lastPinchY = indexTip.y;
          feedback.innerText = "✊ Pinch: Rotating";
          feedback.classList.add("active");
        } else {
          // Dragging
          const dx = (indexTip.x - lastPinchX) * ROTATION_SENSITIVITY;
          const dy = (indexTip.y - lastPinchY) * ROTATION_SENSITIVITY;

          // Invert X because webcam is mirrored. Invert Y for natural drag.
          // X-movement rotates around Y-axis. Y-movement rotates around X-axis.
          window.rotateViewer(dy, -dx);

          lastPinchX = indexTip.x;
          lastPinchY = indexTip.y;
        }

        // Reset Zoom state when pinching
        lastHandAngle = null;
      } else {
        // OPEN HAND (Not Pinched) -> Check for Zoom (Rotation)
        isPinched = false;

        // --- 2. HAND ROTATION (Zoom) ---
        // Calculate angle of hand: Vector from Wrist to Middle Finger Base
        // This represents the general orientation of the hand
        const currentAngle = calculateAngle(wrist, middleMCP);

        if (lastHandAngle !== null) {
          let deltaArg = currentAngle - lastHandAngle;

          // Handle wrap-around (e.g. -PI to +PI)
          if (deltaArg > Math.PI) deltaArg -= 2 * Math.PI;
          if (deltaArg < -Math.PI) deltaArg += 2 * Math.PI;

          // Threshold to avoid jitter
          if (Math.abs(deltaArg) > 0.02) {
            feedback.innerText = deltaArg > 0 ? "↻ Zoom In" : "↺ Zoom Out";
            feedback.classList.add("active");

            // Clockwise (positive delta in screen coords Y-down?)
            // Actually atan2(y,x): Y is down.
            // Let's test direction:
            // If moving Clockwise, angle increases?
            // 3Dmol zoom: >1 in, <1 out.

            // Heuristic: delta > 0 => Zoom In. delta < 0 => Zoom Out.
            const zoomFactor = 1 + deltaArg * ZOOM_ENSITIIVTY;
            window.zoomViewer(zoomFactor);
          } else {
            feedback.innerText = "🖐 Open Hand: Rotate to Zoom";
            feedback.classList.add("active");
          }
        } else {
          feedback.innerText = "🖐 Open Hand: Rotate to Zoom";
          feedback.classList.add("active");
        }
        lastHandAngle = currentAngle;
      }
    } else {
      feedback.innerText = "No Hand Detected";
      feedback.classList.remove("active");
      isPinched = false;
      lastHandAngle = null;
    }
  }

  async function toggleHandControl() {
    if (isTracking) {
      if (camera) await camera.stop();
      canvasElement.style.display = "none";
      toggleBtn.innerText = "Enable Hand Tracking";
      toggleBtn.classList.remove("btn-primary"); // Assuming active style
      toggleBtn.classList.add("btn-secondary");
      isTracking = false;
    } else {
      toggleBtn.innerText = "Starting Camera...";

      try {
        if (!hands) {
          hands = new Hands({
            locateFile: (file) => {
              return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            },
          });
          hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          hands.onResults(onResults);
        }

        camera = new Camera(videoElement, {
          onFrame: async () => {
            await hands.send({
              image: videoElement,
            });
          },
          width: 320,
          height: 240,
        });

        await camera.start();
        isTracking = true;

        // Show Canvas
        canvasElement.style.display = "block";
        canvasElement.width = 320;
        canvasElement.height = 240;

        // Style the canvas
        canvasElement.style.position = "fixed";
        canvasElement.style.bottom = "20px";
        canvasElement.style.left = "20px";
        canvasElement.style.borderRadius = "8px";
        canvasElement.style.border = "2px solid var(--accent-cyan)";
        canvasElement.style.width = "160px";
        canvasElement.style.height = "120px";

        toggleBtn.innerText = "Disable Hand Tracking";
        toggleBtn.classList.remove("btn-secondary");
        toggleBtn.classList.add("btn-primary"); // Switch to primary/active color
      } catch (err) {
        console.error(err);
        alert("Webcam error: " + err.message);
        toggleBtn.innerText = "Enable Hand Tracking";
        isTracking = false;
      }
    }
  }
});
