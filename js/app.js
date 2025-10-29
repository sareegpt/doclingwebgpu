// Minimal Granite Docling (browser) using Transformers.js v3
// Imports: AutoProcessor + AutoModelForVision2Seq + RawImage + TextStreamer
// CDN ref: https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5
import {
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
  TextStreamer
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

// DOM
const fileInput = document.getElementById("fileInput");
const runBtn = document.getElementById("runBtn");
const promptEl = document.getElementById("prompt");
const outEl = document.getElementById("out");
const preview = document.getElementById("preview");
const previewWrap = document.getElementById("previewWrap");
const loader = document.getElementById("loader");
const progress = document.getElementById("progress");
const progressText = document.getElementById("progressText");
const deviceNote = document.getElementById("deviceNote");

// Hidden canvas to convert image -> RawImage
const hiddenCanvas = document.createElement("canvas");
const ctx = hiddenCanvas.getContext("2d");

// Choose device (WebGPU or WASM fallback)
const DEVICE = typeof navigator !== "undefined" && navigator.gpu ? "webgpu" : "wasm";
deviceNote.textContent = DEVICE === "webgpu"
  ? "Using WebGPU acceleration"
  : "WebGPU unavailable — falling back to WASM (slower)";

// ONNX model recommended by the official Space (converted for web runtimes)
const MODEL_ID = "onnx-community/granite-docling-258M-ONNX";

let processor = null;
let model = null;

// ———————————————————————————————
// Model init (with progress bar)
async function initModel() {
  loader.classList.remove("hidden");

  // track download of the 3 .onnx_data blobs
  const prog = {};

  processor = await AutoProcessor.from_pretrained(MODEL_ID);

  model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
    device: DEVICE,
    // Per the reference Space, mixed precision is safe; q4 for decoder can cause repetition.
    dtype: {
      embed_tokens: "fp16",       // ~116 MB
      vision_encoder: "fp32",     // ~374 MB
      decoder_model_merged: "fp32"// ~658 MB  (q4 ~105MB but can repeat)
    },
    progress_callback: (data) => {
      if (data.status === "progress" && data.file?.endsWith?.("onnx_data")) {
        prog[data.file] = data;
        if (Object.keys(prog).length !== 3) return;

        let loaded = 0, total = 0;
        for (const v of Object.values(prog)) { loaded += v.loaded; total += v.total; }
        const pct = Math.round((loaded / total) * 100);
        progress.value = pct;
        progressText.textContent = pct + "%";
      }
    },
  });

  loader.classList.add("hidden");
  runBtn.disabled = false;
}

// ———————————————————————————————
// Prepare inputs: (image + prompt) -> processor -> model.generate(stream)
async function run(imageBitmap) {
  outEl.textContent = ""; // clear
  let full = "";

  // 1) Draw to canvas -> RawImage
  hiddenCanvas.width = imageBitmap.width;
  hiddenCanvas.height = imageBitmap.height;
  ctx.drawImage(imageBitmap, 0, 0);
  const raw = RawImage.fromCanvas(hiddenCanvas);

  // 2) Messages -> chat template (Docling expects instruction text)
  const messages = [{
    role: "user",
    content: [{ type: "image" }, { type: "text", text: promptEl.value }]
  }];

  const chatText = processor.apply_chat_template(messages, { add_generation_prompt: true });

  // 3) Processor packs text + image(s)
  const inputs = await processor(chatText, [raw], {
    do_image_splitting: true, // like the Space; helps large pages
  });

  // 4) Streamed generation to the <pre>
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    // Keep special tokens (Docling tags are "tokens" we want to see)
    skip_special_tokens: false,
    callback_function: (piece) => {
      full += piece;
      outEl.textContent += piece;
    }
  });

  await model.generate({
    ...inputs,
    max_new_tokens: 4096,
    streamer
  });

  // Trim any end token string if present
  outEl.textContent = full.replace(/<\|end_of_text\|>$/, "");
}

// ———————————————————————————————
// UI handlers
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  // show preview
  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.classList.remove("hidden");

  // lazy-load model on first use
  if (!model) await initModel();

  const bitmap = await createImageBitmap(file);
  await run(bitmap);
});

runBtn.addEventListener("click", async () => {
  // allow running again with the same image (e.g., changed prompt)
  const file = fileInput.files?.[0];
  if (!file) return;
  if (!model) await initModel();

  const bitmap = await createImageBitmap(file);
  await run(bitmap);
});

// Optional: eager-load the model at page open
// initModel();
