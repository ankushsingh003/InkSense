// Global variables
let currentFile = null;
let currentUrl = null;

/**
 * Switch between upload and URL tabs
 * @param {string} tab - 'upload' or 'url'
 */
function switchTab(tab) {
    // Update tabs
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    if (tab === 'upload') {
        document.querySelectorAll('.tab')[0].classList.add('active');
        document.getElementById('upload-tab').classList.add('active');
    } else {
        document.querySelectorAll('.tab')[1].classList.add('active');
        document.getElementById('url-tab').classList.add('active');
    }

    // Reset results and errors
    document.getElementById('results').classList.remove('show');
    document.getElementById('errorMessage').classList.remove('show');
    document.getElementById('preview').style.display = 'none';
}

/**
 * Initialize drag and drop functionality
 */
function initDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

/**
 * Handle file selection from input
 * @param {Event} event - File input change event
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

/**
 * Process selected file
 * @param {File} file - Selected image file
 */
function handleFile(file) {
    currentFile = file;
    currentUrl = null;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('preview');
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Hide results and errors
    document.getElementById('results').classList.remove('show');
    document.getElementById('errorMessage').classList.remove('show');
}

/**
 * Main prediction function
 */
async function predictPrice() {
    const urlInput = document.getElementById('urlInput');
    const predictBtn = document.getElementById('predictBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorMessage = document.getElementById('errorMessage');
    const priceDisplay = document.getElementById('priceDisplay');

    // Reset UI
    results.classList.remove('show');
    errorMessage.classList.remove('show');

    // Check which tab is active
    const activeTab = document.querySelector('.tab-content.active').id;

    // Validate input
    if (activeTab === 'upload-tab' && !currentFile) {
        showError('Please select an image file');
        return;
    }

    if (activeTab === 'url-tab') {
        currentUrl = urlInput.value.trim();
        if (!currentUrl) {
            showError('Please enter an image URL');
            return;
        }
    }

    // Show loading state
    predictBtn.disabled = true;
    loading.classList.add('show');

    try {
        let response;

        if (activeTab === 'upload-tab') {
            // Upload file
            const formData = new FormData();
            formData.append('file', currentFile);

            response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
        } else {
            // Send URL
            response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_url: currentUrl })
            });

            // Show preview for URL
            const preview = document.getElementById('preview');
            preview.src = currentUrl;
            preview.style.display = 'block';
        }

        const data = await response.json();

        if (data.success) {
            // Display price
            priceDisplay.textContent = data.formatted_price;
            results.classList.add('show');
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        // Reset loading state
        predictBtn.disabled = false;
        loading.classList.remove('show');
    }
}

/**
 * Display error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = '❌ ' + message;
    errorMessage.classList.add('show');
}

// Initialize drag and drop when page loads
document.addEventListener('DOMContentLoaded', () => {
    initDragAndDrop();
});
