// Qwen 3.5 Vision — Browser inference via Transformers.js + WebGPU
// Supports multimodal (image + text) and text-only generation
import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

// ─── DOM refs ───────────────────────────────────────
const landing     = document.getElementById("landing");
const appSection  = document.getElementById("app");
const loadBtn     = document.getElementById("loadBtn");
const loadBtnLabel= document.getElementById("loadBtnLabel");
const modelMenu   = document.getElementById("modelMenu");
const fileInput   = document.getElementById("fileInput");
const dropZone    = document.getElementById("dropZone");
const runBtn      = document.getElementById("runBtn");
const promptEl    = document.getElementById("prompt");
const outEl       = document.getElementById("out");
const statsEl     = document.getElementById("stats");
const preview     = document.getElementById("preview");
const previewWrap = document.getElementById("previewWrap");
const loader      = document.getElementById("loader");
const loaderTitle = document.getElementById("loaderTitle");
const loaderDesc  = document.getElementById("loaderDesc");
const progress    = document.getElementById("progress");
const progressText= document.getElementById("progressText");
const deviceNote  = document.getElementById("deviceNote");
const modelTag    = document.getElementById("modelTag");

// ─── Config ─────────────────────────────────────────
const DEVICE = navigator?.gpu ? "webgpu" : "wasm";

const MODELS = {
  "onnx-community/Qwen3.5-0.8B-ONNX": { label: "0.8B", size: "~500 MB" },
  "onnx-community/Qwen3.5-2B-ONNX":   { label: "2B",   size: "~1.3 GB" },
};

let selectedModel = "onnx-community/Qwen3.5-0.8B-ONNX";
let processor = null;
let model = null;
let isGenerating = false;

// Hidden canvas for image → RawImage conversion
const hiddenCanvas = document.createElement("canvas");
const ctx = hiddenCanvas.getContext("2d");

// ─── Model dropdown ─────────────────────────────────
loadBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  modelMenu.classList.toggle("hidden");
});

document.addEventListener("click", () => {
  modelMenu.classList.add("hidden");
});

modelMenu.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedModel = btn.dataset.model;
    loadBtnLabel.textContent = `Load Model (${btn.dataset.label})`;
    modelMenu.classList.add("hidden");
    initModel();
  });
});

// Also allow direct click on the main button text to load default
loadBtn.addEventListener("dblclick", () => {
  modelMenu.classList.add("hidden");
  initModel();
});

// ─── Model loading ──────────────────────────────────
async function initModel() {
  loader.classList.remove("hidden");
  loaderTitle.textContent = "Loading model…";
  loaderDesc.textContent = "This downloads to your browser cache (first time only).";
  deviceNote.textContent = DEVICE === "webgpu"
    ? "Using WebGPU acceleration"
    : "WebGPU unavailable — falling back to WASM (slower)";
  progress.value = 0;
  progressText.textContent = "0%";

  const fileProgress = {};

  try {
    // Try loading as a vision-language model first
    processor = await AutoProcessor.from_pretrained(selectedModel);

    model = await AutoModelForImageTextToText.from_pretrained(selectedModel, {
      device: DEVICE,
      dtype: "q4f16",
      progress_callback: (data) => {
        if (data.status === "progress" && data.file) {
          fileProgress[data.file] = data;
          let loaded = 0, total = 0;
          for (const v of Object.values(fileProgress)) {
            loaded += v.loaded || 0;
            total += v.total || 0;
          }
          if (total > 0) {
            const pct = Math.round((loaded / total) * 100);
            progress.value = pct;
            progressText.textContent = `${pct}%`;
          }
        }
      },
    });
  } catch (err) {
    // Fallback: try as a causal LM if vision model class isn't supported
    console.warn("AutoModelForImageTextToText failed, trying text-only generation:", err);

    const { AutoModelForCausalLM, AutoTokenizer } = await import(
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5"
    );

    processor = await AutoTokenizer.from_pretrained(selectedModel);

    model = await AutoModelForCausalLM.from_pretrained(selectedModel, {
      device: DEVICE,
      dtype: "q4f16",
      progress_callback: (data) => {
        if (data.status === "progress" && data.file) {
          fileProgress[data.file] = data;
          let loaded = 0, total = 0;
          for (const v of Object.values(fileProgress)) {
            loaded += v.loaded || 0;
            total += v.total || 0;
          }
          if (total > 0) {
            const pct = Math.round((loaded / total) * 100);
            progress.value = pct;
            progressText.textContent = `${pct}%`;
          }
        }
      },
    });
  }

  loader.classList.add("hidden");

  // Switch to app view
  const info = MODELS[selectedModel];
  modelTag.textContent = `Qwen 3.5 — ${info.label}`;
  landing.classList.add("hidden");
  appSection.classList.remove("hidden");
  runBtn.disabled = false;
}

// ─── Image handling ─────────────────────────────────
let currentFile = null;

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  loadImage(file);
});

// Drag & drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files?.[0];
  if (file && file.type.startsWith("image/")) loadImage(file);
});

function loadImage(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.classList.remove("hidden");
}

// ─── Inference ──────────────────────────────────────
async function run() {
  if (isGenerating || !model) return;
  isGenerating = true;
  runBtn.disabled = true;
  runBtn.textContent = "Running…";
  outEl.textContent = "";
  statsEl.classList.add("hidden");

  const t0 = performance.now();
  let tokenCount = 0;
  let full = "";

  try {
    // Build messages
    const userContent = [];

    if (currentFile) {
      // Vision: image + text
      const bitmap = await createImageBitmap(currentFile);
      hiddenCanvas.width = bitmap.width;
      hiddenCanvas.height = bitmap.height;
      ctx.drawImage(bitmap, 0, 0);
      const rawImage = RawImage.fromCanvas(hiddenCanvas);

      userContent.push({ type: "image", image: rawImage });
      userContent.push({ type: "text", text: promptEl.value });
    } else {
      userContent.push({ type: "text", text: promptEl.value });
    }

    const messages = [{ role: "user", content: userContent }];

    // Check if processor is a tokenizer (text-only fallback) or full processor
    const isFullProcessor = typeof processor.apply_chat_template === "function"
      && typeof processor !== "function";

    if (isFullProcessor && processor.apply_chat_template) {
      // Vision-language model path
      const chatText = processor.apply_chat_template(messages, {
        add_generation_prompt: true,
      });

      const inputs = currentFile
        ? await processor(chatText, [userContent[0].image])
        : await processor(chatText);

      const streamer = new TextStreamer(processor.tokenizer || processor, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (piece) => {
          tokenCount++;
          full += piece;
          outEl.textContent = full;
        },
      });

      await model.generate({
        ...inputs,
        max_new_tokens: 2048,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
        streamer,
      });
    } else {
      // Text-only causal LM path
      const chatMessages = messages.map(m => ({
        role: m.role,
        content: typeof m.content === "string"
          ? m.content
          : m.content.filter(c => c.type === "text").map(c => c.text).join(" "),
      }));

      const chatText = processor.apply_chat_template(chatMessages, {
        add_generation_prompt: true,
        return_dict: false,
      });

      const inputs = processor(chatText, { return_tensors: "pt" });

      const streamer = new TextStreamer(processor, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (piece) => {
          tokenCount++;
          full += piece;
          outEl.textContent = full;
        },
      });

      await model.generate({
        ...inputs,
        max_new_tokens: 2048,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
        streamer,
      });
    }
  } catch (err) {
    outEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }

  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  const tps = tokenCount > 0 ? (tokenCount / ((performance.now() - t0) / 1000)).toFixed(1) : "—";
  statsEl.textContent = `${tokenCount} tokens in ${elapsed}s (${tps} tok/s)`;
  statsEl.classList.remove("hidden");

  isGenerating = false;
  runBtn.disabled = false;
  runBtn.textContent = "Run";
}

runBtn.addEventListener("click", run);

// Enter key in prompt
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    run();
  }
});
