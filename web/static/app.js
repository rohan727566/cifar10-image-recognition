const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const startBtn = document.getElementById("startBtn");
const spinner = document.getElementById("spinner");
const results = document.getElementById("results");
const predictionsList = document.getElementById("predictionsList");

let selectedFile = null;

// OPEN FILE SELECT WHEN CLICKING DROPZONE
dropZone.addEventListener("click", () => imageInput.click());

// DRAG OVER
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#fff";
});

// FILE DROPPED
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#ffffff66";
  processFile(e.dataTransfer.files[0]);
});

// FILE SELECTED
imageInput.addEventListener("change", () => {
  processFile(imageInput.files[0]);
});

// PREVIEW FILE
function processFile(file) {
  if (!file) return;

  selectedFile = file;

  preview.src = URL.createObjectURL(file);
  preview.hidden = false;

  startBtn.hidden = false;
  results.hidden = true;
}

// START RECOGNITION
startBtn.addEventListener("click", () => {
  if (!selectedFile) return;

  startBtn.hidden = true;
  spinner.hidden = false;

  const formData = new FormData();
  formData.append("file", selectedFile);

  fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      spinner.hidden = true;
      results.hidden = false;
      predictionsList.innerHTML = "";

      // Use original backend fields
      const predictions = data.top3 || [];

      predictions.forEach((item) => {
        const percent = (item.probability * 100).toFixed(2);

        const div = document.createElement("div");
        div.className = "pred-item";
        div.innerHTML = `
          <div class="pred-label">${item.label} — ${percent}%</div>
          <div class="progress-bar">
            <div class="progress-fill" style="width:${percent}%"></div>
          </div>
        `;
        predictionsList.appendChild(div);
      });
    })
    .catch(() => {
      spinner.hidden = true;
      alert("Error: API not responding. Make sure FastAPI is running.");
    });
});
