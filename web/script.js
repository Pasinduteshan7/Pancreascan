// ============================================================
// PancreaScan — Frontend Logic (Two-Stage Pipeline)
// ============================================================

const API_BASE = '';  // Same origin

// DOM Elements
const uploadZone      = document.getElementById('uploadZone');
const fileInput       = document.getElementById('fileInput');
const samplesGrid     = document.getElementById('samplesGrid');
const resultsSection  = document.getElementById('resultsSection');
const resultsLoading  = document.getElementById('resultsLoading');
const resultsContent  = document.getElementById('resultsContent');
const resultsSubtitle = document.getElementById('resultsSubtitle');
const detectionAlert  = document.getElementById('detectionAlert');
const imageGrid       = document.getElementById('imageGrid');
const statBackground  = document.getElementById('statBackground');
const statPancreas    = document.getElementById('statPancreas');
const statTumor       = document.getElementById('statTumor');
const statConfidence  = document.getElementById('statConfidence');
const newAnalysisBtn  = document.getElementById('newAnalysisBtn');

// Pipeline image elements
const ctImage          = document.getElementById('ctImage');
const stage1Image      = document.getElementById('stage1Image');
const cropImage        = document.getElementById('cropImage');
const overlayImage     = document.getElementById('overlayImage');
const confidenceImage  = document.getElementById('confidenceImage');
const groundTruthCard  = document.getElementById('groundTruthCard');
const groundTruthImage = document.getElementById('groundTruthImage');

// ============================================================
// UPLOAD HANDLING
// ============================================================

uploadZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

async function uploadFile(file) {
    if (!file.name.endsWith('.npy')) {
        showError('Please upload a .npy file');
        return;
    }

    showLoading('Running two-stage pipeline on uploaded slice...');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Prediction failed');
        displayResults(data, file.name);
    } catch (err) {
        showError(err.message);
    }
}

// ============================================================
// SAMPLE GALLERY
// ============================================================

async function loadSamples() {
    try {
        const response = await fetch(`${API_BASE}/api/samples`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);

        if (data.samples.length === 0) {
            samplesGrid.innerHTML = `
                <div class="samples__loading">
                    <p>No sample slices found. Make sure data/processed/ exists.</p>
                </div>`;
            return;
        }

        samplesGrid.innerHTML = '';
        data.samples.forEach((sample, idx) => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.style.animationDelay = `${idx * 0.05}s`;
            card.innerHTML = `
                <div class="sample-card__name">${sample.filename}</div>
                <div class="sample-card__tags">
                    ${sample.has_pancreas ? '<span class="sample-card__tag sample-card__tag--pancreas">Pancreas</span>' : ''}
                    ${sample.has_tumor    ? '<span class="sample-card__tag sample-card__tag--tumor">Tumor</span>'    : ''}
                </div>`;
            card.addEventListener('click', () => predictSample(sample.filename));
            samplesGrid.appendChild(card);
        });
    } catch (err) {
        samplesGrid.innerHTML = `
            <div class="samples__loading">
                <p style="color: var(--accent-red);">Failed to load samples: ${err.message}</p>
            </div>`;
    }
}

async function predictSample(filename) {
    showLoading(`Running pipeline on: ${filename}`);
    try {
        const response = await fetch(`${API_BASE}/api/predict-sample`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Prediction failed');
        displayResults(data, filename);
    } catch (err) {
        showError(err.message);
    }
}

// ============================================================
// RESULTS DISPLAY
// ============================================================

function showLoading(message) {
    resultsSection.style.display = '';
    resultsLoading.style.display = '';
    resultsContent.style.display = 'none';
    resultsLoading.querySelector('.results__loading-text').textContent = message;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    resultsSection.style.display = '';
    resultsLoading.style.display = 'none';
    resultsContent.style.display = '';

    detectionAlert.className = 'alert alert--danger';
    detectionAlert.innerHTML = `&#9888; <span>${message}</span>`;
    detectionAlert.style.display = '';

    document.getElementById('statsRow').style.display = 'none';
    imageGrid.style.display = 'none';
    document.querySelector('.legend').style.display = 'none';
    document.querySelector('.pipeline-flow').style.display = 'none';

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayResults(data, filename) {
    resultsLoading.style.display = 'none';
    resultsContent.style.display = '';
    document.getElementById('statsRow').style.display = '';
    imageGrid.style.display = '';
    document.querySelector('.legend').style.display = '';
    document.querySelector('.pipeline-flow').style.display = '';

    resultsSubtitle.textContent = `Two-stage pipeline analysis of: ${filename}`;

    const stats = data.stats;

    // Detection alert
    if (stats.tumor_detected) {
        detectionAlert.className = 'alert alert--danger';
        detectionAlert.innerHTML = `<span class="alert__icon">&#128308;</span>
            <span>Tumor tissue detected &mdash; Max confidence: <strong>${(stats.max_tumor_confidence * 100).toFixed(1)}%</strong></span>`;
    } else if (stats.pancreas_detected) {
        detectionAlert.className = 'alert alert--success';
        detectionAlert.innerHTML = `<span class="alert__icon">&#9989;</span>
            <span>Pancreas detected &mdash; No tumor identified in this slice</span>`;
    } else {
        detectionAlert.className = 'alert alert--info';
        detectionAlert.innerHTML = `<span class="alert__icon">&#8505;</span>
            <span>No pancreas detected &mdash; This slice may not contain pancreatic tissue</span>`;
    }
    detectionAlert.style.display = '';

    // Stat cards
    statBackground.textContent = `${stats.background_pct.toFixed(1)}%`;
    statPancreas.textContent   = `${stats.pancreas_pct.toFixed(1)}%`;
    statTumor.textContent      = `${stats.tumor_pct.toFixed(2)}%`;
    statConfidence.textContent = `${(stats.max_tumor_confidence * 100).toFixed(1)}%`;

    // Pipeline images
    ctImage.src = `data:image/png;base64,${data.ct_image}`;

    if (data.stage1_image) {
        stage1Image.src = `data:image/png;base64,${data.stage1_image}`;
        stage1Image.closest('.image-card').style.display = '';
    }

    if (data.crop_image) {
        cropImage.src = `data:image/png;base64,${data.crop_image}`;
        cropImage.closest('.image-card').style.display = '';
    } else {
        cropImage.closest('.image-card').style.display = 'none';
    }

    overlayImage.src = `data:image/png;base64,${data.overlay_image}`;

    if (data.confidence_image) {
        confidenceImage.src = `data:image/png;base64,${data.confidence_image}`;
        confidenceImage.closest('.image-card').style.display = '';
    } else {
        confidenceImage.closest('.image-card').style.display = 'none';
    }

    // Ground truth (samples only)
    if (data.has_ground_truth && data.ground_truth_image) {
        groundTruthCard.style.display = '';
        groundTruthImage.src = `data:image/png;base64,${data.ground_truth_image}`;
    } else {
        groundTruthCard.style.display = 'none';
    }

    retriggerAnimations();
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function retriggerAnimations() {
    const animated = resultsContent.querySelectorAll('.stat-card, .image-card');
    animated.forEach(el => {
        el.style.animation = 'none';
        el.offsetHeight;
        el.style.animation = '';
    });
}

// ============================================================
// NEW ANALYSIS
// ============================================================

newAnalysisBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    fileInput.value = '';
    document.getElementById('analyze').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

// ============================================================
// INIT
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadSamples();
});
