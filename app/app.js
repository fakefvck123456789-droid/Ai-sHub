const statusNode = document.querySelector("#status");
const promptInput = document.querySelector("#prompt");
const responseNode = document.querySelector("#response");
const errorNode = document.querySelector("#error");
const sendButton = document.querySelector("#send");
const copyButton = document.querySelector("#copy");

function showError(message) {
  errorNode.textContent = message;
  errorNode.hidden = !message;
}

function initialize() {
  if (!window.AiHub || typeof AiHub.generate !== "function") {
    statusNode.textContent = "Unavailable";
    statusNode.className = "status offline";
    showError("The on-device model runtime is unavailable. Reinstall the official Ai'Hub Android build.");
    sendButton.disabled = true;
    return;
  }
  statusNode.textContent = "Ready · " + AiHub.modelName();
  statusNode.className = "status online";
  document.querySelector("#model").replaceChildren(new Option(AiHub.modelName(), "core-prototype"));
}

sendButton.addEventListener("click", () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    showError("Write a message first.");
    promptInput.focus();
    return;
  }
  showError("");
  sendButton.disabled = true;
  sendButton.textContent = "Generating…";
  responseNode.textContent = "Generating on this device…";
  copyButton.disabled = true;
  try {
    const result = JSON.parse(AiHub.generate(prompt, Number(document.querySelector("#tokens").value)));
    if (result.error) throw new Error(result.error);
    responseNode.textContent = result.text || "(The model returned an empty response.)";
    copyButton.disabled = !result.text;
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
    showError("Clipboard access is unavailable in this app context.");
  }
});

initialize();
