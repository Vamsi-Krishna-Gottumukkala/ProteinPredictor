document.addEventListener("DOMContentLoaded", () => {
  // Initialize 3Dmol viewer
  let viewer = null;
  const element = document.getElementById("mol-viewer");
  const config = { backgroundColor: "#11131f" };

  // Create viewer instance
  viewer = $3Dmol.createViewer(element, config);
  window.viewer = viewer; // Expose globally for hand controls

  // EXPOSE CONTROLS FOR HAND GESTURES
  window.data = viewer; // temporary debug access

  window.zoomViewer = function (factor) {
    if (viewer) {
      viewer.zoom(factor);
      // 3Dmol zoom: multiplier. >1 zooms in.
      // But usually zoom(factor) might be absolute or relative.
      // Docs: zoom(factor, animationDuration)
      // "Multiplies the current zoom by factor"
    }
  };

  window.rotateViewer = function (speedX, speedY) {
    if (viewer) {
      // user rotates by moving mouse/hand
      // rotate(angle, axis) or just spin logic?
      // "rotate(degrees, axis)"
      // Simplest way to emulate mouse drag is rotate the camera
      // BUT viewer.rotate() rotates the model.

      // Let's use viewer.rotate(angle, axis)
      // Axis logic: dragging X rotates around Y axis. Dragging Y rotates around X axis.
      if (speedX !== 0) viewer.rotate(speedX * 90, { x: 0, y: 1, z: 0 }); // speed is small diff
      if (speedY !== 0) viewer.rotate(speedY * 90, { x: 1, y: 0, z: 0 });
    }
  };

  // Resize handling
  window.addEventListener("resize", () => {
    if (viewer) viewer.resize();
  });

  // Handle Prediction
  const predictBtn = document.getElementById("predict-btn");
  const sequenceInput = document.getElementById("sequence-input");

  predictBtn.addEventListener("click", async () => {
    const rawText = sequenceInput.value;
    const sequence = parseFasta(rawText);

    if (!sequence) {
      alert("Please enter a valid sequence.");
      return;
    }

    // UI Loading State
    predictBtn.innerHTML = '<span class="btn-icon">⏳</span> Processing...';
    predictBtn.disabled = true;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: sequence }),
      });

      const data = await response.json();

      if (data.success) {
        // Render Structure
        renderStructure(data.pdb);
        // Update Metrics
        updateMetrics(data.metrics);
        // Add to history
        addToHistory(sequence.substring(0, 10) + "...");
      } else {
        alert("Error: " + data.error);
      }
    } catch (err) {
      console.error(err);
      alert("Network Error: " + err.message);
    } finally {
      predictBtn.innerHTML = '<span class="btn-icon">▶</span> Run Prediction';
      predictBtn.disabled = false;
    }
  });

  function parseFasta(text) {
    // Simple parser: remove header lines starting with >, join rest
    const lines = text.split("\n");
    let seq = "";
    for (let line of lines) {
      line = line.trim();
      if (line.startsWith(">")) continue;
      seq += line;
    }
    return seq.replace(/\s+/g, "").toUpperCase(); // Remove whitespace
  }

  function renderStructure(pdbData) {
    if (!viewer) return;

    console.log("Rendering PDB data length:", pdbData.length);
    viewer.clear();
    viewer.addModel(pdbData, "pdb");

    // Custom Color Function: Red (0) -> Yellow (50) -> Blue (100)
    const colorByConfidence = function (atom) {
      let score = atom.b;
      if (score === undefined || score === null) score = 0;

      let r, g, b;
      if (score <= 50) {
        // Red to Yellow
        const t = score / 50.0;
        r = 255;
        g = Math.floor(255 * t);
        b = 0;
      } else {
        // Yellow to Blue
        const t = (score - 50) / 50.0;
        const c = Math.floor(255 * (1 - t));
        r = c;
        g = c;
        b = Math.floor(255 * t);
      }
      // Return as standard CSS string which 3Dmol supports reliably
      return "rgb(" + r + "," + g + "," + b + ")";
    };

    // Style: Stick (visible) + Cartoon
    viewer.setStyle(
      {},
      {
        stick: { radius: 0.2, colorfunc: colorByConfidence },
        cartoon: { colorfunc: colorByConfidence },
      },
    );

    // Recenter camera
    viewer.zoomTo();

    // Debug: Log atom count
    const m = viewer.getModel();
    if (m) {
      const atomCount = m.selectedAtoms({}).length;
      console.log("Model atoms:", atomCount);
      if (atomCount === 0) {
        alert(
          "Error: Parsed 0 atoms from structure. The PDB format may be invalid.",
        );
      }
    } else {
      console.error("Model not added to viewer");
      alert("Error: Failed to add model to viewer.");
    }

    viewer.render();
    viewer.resize(); // Ensure fit
  }

  function updateMetrics(metrics) {
    console.log("Updating metrics:", metrics);

    // Confidence
    const conf = metrics.mean_confidence.toFixed(1);
    document.getElementById("metric-confidence").innerText = conf + "%";
    document.getElementById("conf-bar").style.width = conf + "%";

    // Atomic Clashes
    document.getElementById("metric-clashes").innerText = metrics.num_clashes;

    // Bond Deviation
    document.getElementById("metric-bond").innerText =
      metrics.avg_bond_deviation.toFixed(3) + " Å";

    // Hydrophobic Score
    document.getElementById("metric-hydro").innerText =
      metrics.hydrophobic_score.toFixed(2);

    // Length
    document.getElementById("metric-length").innerText =
      metrics.sequence_length + " aa";

    // Weight: Handle missing key safely if not present
    const weight = metrics.molecular_weight_kda
      ? metrics.molecular_weight_kda.toFixed(1)
      : "--";
    document.getElementById("metric-weight").innerText = weight + " kDa";

    // SS Stats
    const ss = metrics.ss_composition;
    document.getElementById("ss-helix").innerText = "H: " + ss.H;
    document.getElementById("ss-sheet").innerText = "S: " + ss.E;
    document.getElementById("ss-coil").innerText = "C: " + ss.C;
  }

  function addToHistory(name) {
    // Visual only for demo
    const list = document.querySelector(".history-list");
    const card = document.createElement("div");
    card.className = "history-card";
    card.innerHTML = `
            <div class="history-icon cyan"></div>
            <div class="history-info">
                <h4>${name}</h4>
                <span class="history-meta">Just now</span>
            </div>
        `;
    list.prepend(card);
  }

  // Initialize empty viewer grid
  viewer.resize();
});
