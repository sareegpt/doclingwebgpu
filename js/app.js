// js/app.js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

let modelPipeline = null;

// Initialize the model
async function initModel() {
    console.log('Loading model...');
    modelPipeline = await pipeline(
        'image-to-text',
        'ibm-granite/granite-docling-258M',
        { 
            device: 'webgpu',
            dtype: 'fp16' // Use 16-bit quantization for faster loading
        }
    );
    console.log('Model loaded!');
}

// Process uploaded image
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Initialize model if not already loaded
    if (!modelPipeline) {
        await initModel();
    }
    
    // Read image
    const reader = new FileReader();
    reader.onload = async (event) => {
        const image = new Image();
        image.src = event.target.result;
        
        // Process with model
        const result = await modelPipeline(image);
        document.getElementById('output').innerHTML = result;
    };
    reader.readAsDataURL(file);
});

// Preload model on page load (optional)
// initModel();

