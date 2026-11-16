const endpointInput = document.querySelector("#endpoint");
const statusNode = document.querySelector("#status");
const promptInput = document.querySelector("#prompt");
const responseNode = document.querySelector("#response");
const errorNode = document.querySelector("#error");
const sendButton = document.querySelector("#send");
const copyButton = document.querySelector("#copy");

function apiBase() {
  return endpointInput.value.trim().replace(/\/+$/, "");
}
function showError(message) {
  errorNode.textContent = message;
  errorNode.hidden = !message;
}
async function checkServer() {
  statusNode.textContent = "Checking…";
  statusNode.className = "status";
  try {
    const response = await fetch(apiBase() + "/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Server returned HTTP " + response.status);
    const info = await response.json();
    statusNode.textContent = "Online · " + (info.model || "Model ready");
    statusNode.className = "status online";
    return true;
  } catch (error) {
    statusNode.textContent = "Offline";
    statusNode.className = "status offline";
    showError("Cannot reach the local model server. Check the address and make sure the server is running. " + error.message);
    return false;
  }
}
document.querySelector("#check").addEventListener("click", checkServer);
sendButton.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    showError("Write a message first.");
    promptInput.focus();
    return;
  }
  showError("");
  sendButton.disabled = true;
  sendButton.textContent = "Generating…";
  responseNode.textContent = "The local model is generating…";
  copyButton.disabled = true;
  try {
    const response = await fetch(apiBase() + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: Number(document.querySelector("#tokens").value),
        temperature: Number(document.querySelector("#temperature").value),
        top_k: Number(document.querySelector("#topk").value)
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || "HTTP " + response.status);
    responseNode.textContent = payload.text || "(The model returned an empty response.)";
    copyButton.disabled = !payload.text;
  } catch (error) {
    responseNode.textContent = "No response.";
    showError("Generation failed: " + error.message);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Generate response";
  }
});
copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(responseNode.textContent);
    copyButton.textContent = "Copied";
    setTimeout(() => { copyButton.textContent = "Copy"; }, 1200);
  } catch {
    showError("Clipboard access is unavailable in this browser context.");
  }
});
if ("serviceWorker" in navigator && (location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
}
checkServer();
