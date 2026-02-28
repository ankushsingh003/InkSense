import './style.css'

// UI Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const processBtn = document.getElementById('process-btn');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsDisplay = document.getElementById('results-display');
const resultCanvas = document.getElementById('result-canvas');
const processingOverlay = document.getElementById('processing-overlay');
const statusText = document.getElementById('status-text');
const inkScore = document.getElementById('ink-score');

let selectedImage = null;

// Interaction Logic
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = 'var(--primary)';
});

dropZone.addEventListener('dragleave', () => {
  dropZone.style.borderColor = 'var(--glass-border)';
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;

  selectedImage = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    imagePreview.style.display = 'block';
    dropZone.style.display = 'none';
    processBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

processBtn.addEventListener('click', async () => {
  if (!selectedImage) return;

  // Start Simulation
  processingOverlay.style.display = 'flex';
  processBtn.disabled = true;

  const steps = [
    'Extracting Volumetric Slices...',
    'Targeting Carbon Signal...',
    '3D-ResNet Feature Extraction...',
    'Transformer Semantic Mapping...',
    'Morphological Denoising...'
  ];

  for (let i = 0; i < steps.length; i++) {
    statusText.textContent = steps[i];
    await new Promise(r => setTimeout(r, 600 + Math.random() * 400));
  }

  // Generate Simulated Prediction
  generatePrediction();

  processingOverlay.style.display = 'none';
  resultsPlaceholder.style.display = 'none';
  resultsDisplay.style.display = 'flex';
  processBtn.disabled = false;

  // Randomize a high score for "wow" effect
  inkScore.textContent = (0.85 + Math.random() * 0.1).toFixed(3);
});

function generatePrediction() {
  const ctx = resultCanvas.getContext('2d');
  const img = new Image();
  img.src = imagePreview.src;

  img.onload = () => {
    resultCanvas.width = img.width;
    resultCanvas.height = img.height;

    // Draw original at lower opacity
    ctx.globalAlpha = 0.3;
    ctx.drawImage(img, 0, 0);
    ctx.globalAlpha = 1.0;

    // Simulate ink detection with "heatmap" styling
    ctx.fillStyle = '#6366f1'; // Primary color for ink
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#6366f1';

    // Draw dummy "ink" characters using random paths
    for (let i = 0; i < 8; i++) {
      drawGreekCharacter(ctx, Math.random() * img.width, Math.random() * img.height);
    }
  };
}

function drawGreekCharacter(ctx, x, y) {
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#6366f1';

  // Create an abstract character-like shape
  for (let i = 0; i < 5; i++) {
    ctx.lineTo(x + Math.random() * 40 - 20, y + Math.random() * 40 - 20);
  }
  ctx.stroke();
}
