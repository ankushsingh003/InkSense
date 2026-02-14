
let currentFile = null;

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    if (tab === 'upload') {
        document.querySelectorAll('.tab')[0].classList.add('active');
        document.getElementById('upload-tab').classList.add('active');
    } else {
        document.querySelectorAll('.tab')[1].classList.add('active');
        document.getElementById('url-tab').classList.add('active');
    }
    resetUI();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('preview');
            img.src = e.target.result;
            img.style.display = 'block';
        };
        reader.readAsDataURL(file);
        resetUI();
    }
}

function resetUI() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('errorMessage').innerText = '';
}

async function generateCaption() {
    const btn = document.getElementById('generateBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const captionText = document.getElementById('captionText');
    const errorDiv = document.getElementById('errorMessage');

    // Check inputs
    const isUpload = document.getElementById('upload-tab').classList.contains('active');
    if (isUpload && !currentFile) {
        alert("Please select an image first");
        return;
    }

    btn.disabled = true;
    loading.style.display = 'block';
    resetUI();

    try {
        let response;
        if (isUpload) {
            const formData = new FormData();
            formData.append('file', currentFile);
            response = await fetch('/caption', { method: 'POST', body: formData });
        } else {
            const url = document.getElementById('urlInput').value;
            if (!url) { alert("Please enter a URL"); btn.disabled = false; loading.style.display = 'none'; return; }
            response = await fetch('/caption', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_url: url })
            });
            // Show preview for URL
            document.getElementById('preview').src = url;
            document.getElementById('preview').style.display = 'block';
        }

        const data = await response.json();

        if (data.success) {
            captionText.innerText = data.caption;
            results.style.display = 'block';
        } else {
            errorDiv.innerText = "Error: " + data.error;
            errorDiv.style.display = 'block';
        }

    } catch (e) {
        errorDiv.innerText = "Network Error: " + e.message;
        errorDiv.style.display = 'block';
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}
