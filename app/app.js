const statusNode = document.querySelector("#status");
const promptInput = document.querySelector("#prompt");
const conversationNode = document.querySelector("#conversation");
const errorNode = document.querySelector("#error");
const sendButton = document.querySelector("#send");
const newChatButton = document.querySelector("#new-chat");
const HISTORY_KEY = "aihub.conversation.v1";
let messages = loadMessages();

function loadMessages() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(saved) ? saved.filter(item =>
      item && (item.role === "user" || item.role === "assistant") &&
      typeof item.text === "string").slice(-80) : [];
  } catch { return []; }
}
function saveMessages() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-80))); }
  catch { showError("Local storage is full. Start a new chat to free space."); }
}
function showError(message) { errorNode.textContent = message; errorNode.hidden = !message; }
function renderConversation() {
  conversationNode.replaceChildren();
  if (!messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Your conversation stays on this device. Send a message to begin.";
    conversationNode.append(empty); return;
  }
  for (const message of messages) {
    const article = document.createElement("article");
    article.className = "message " + message.role;
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = message.role === "user" ? "YOU" : "AI'HUB CORE";
    const content = document.createElement("p");
    content.textContent = message.text;
    article.append(label, content); conversationNode.append(article);
  }
  conversationNode.scrollTop = conversationNode.scrollHeight;
}
function initialize() {
  if (!window.AiHub || typeof AiHub.generate !== "function") {
    statusNode.textContent = "Unavailable"; statusNode.className = "status offline";
    showError("The on-device runtime could not initialize. Please reinstall this build.");
    sendButton.disabled = true; return;
  }
  const name = AiHub.modelName();
  statusNode.textContent = "Ready · " + name;
  statusNode.className = "status online";
  document.querySelector("#model").replaceChildren(new Option(name, "core"));
  const warning = AiHub.startupStatus();
  if (warning) showError(warning);
  renderConversation();
}
sendButton.addEventListener("click", () => {
  const prompt = promptInput.value.trim();
  if (!prompt) { showError("Write a message first."); promptInput.focus(); return; }
  showError(""); sendButton.disabled = true; sendButton.textContent = "Generating…";
  messages.push({role: "user", text: prompt}); renderConversation();
  const context = messages.slice(-8).map(item =>
    (item.role === "user" ? "User: " : "Ai'Hub: ") + item.text).join("\n") + "\nAi'Hub:";
  try {
    const result = JSON.parse(AiHub.generate(context, Number(document.querySelector("#tokens").value)));
    if (result.error) throw new Error(result.error);
    messages.push({role: "assistant", text: result.text || "(The model returned an empty response.)"});
    messages = messages.slice(-80); saveMessages(); renderConversation(); promptInput.value = "";
  } catch (error) {
    messages.pop(); renderConversation(); showError("Generation failed: " + error.message);
  } finally { sendButton.disabled = false; sendButton.textContent = "Send message"; }
});
newChatButton.addEventListener("click", () => {
  messages = []; saveMessages(); renderConversation(); showError(""); promptInput.focus();
});
initialize();
