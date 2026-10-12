const API_BASE = "";

const conversationList = document.querySelector("#conversation-list");
const messagesElement = document.querySelector("#messages");
const messageForm = document.querySelector("#message-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const titleElement = document.querySelector("#conversation-title");
const kindElement = document.querySelector("#conversation-kind");
const headerStatus = document.querySelector("#header-status");
const connectionDot = document.querySelector("#connection-dot");
const connectionLabel = document.querySelector("#connection-label");
const newConversationButton = document.querySelector("#new-conversation");

let conversations = [];
let activeConversation = null;
let activeMessages = [];
let sending = false;

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch { /* Keep the HTTP status as the error message. */ }
    throw new Error(detail);
  }
  return response.json();
}

function setConnection(online, label) {
  connectionDot.className = `connection-dot ${online ? "online" : "offline"}`;
  connectionLabel.textContent = label;
}

function displayName(conversation) {
  return conversation.mode === "role-play"
    ? (conversation.character_name || "角色对话")
    : "助手对话";
}

function renderConversationList() {
  conversationList.replaceChildren();
  if (!conversations.length) {
    const empty = document.createElement("div");
    empty.className = "list-empty";
    empty.textContent = "还没有对话";
    conversationList.append(empty);
    return;
  }

  for (const conversation of conversations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-item";
    if (conversation.conversation_id === activeConversation?.conversation_id) {
      button.classList.add("active");
    }
    button.disabled = sending;
    button.title = displayName(conversation);

    const icon = document.createElement("span");
    icon.className = "conversation-icon";
    icon.textContent = conversation.mode === "role-play" ? "人" : "A";

    const copy = document.createElement("span");
    copy.className = "conversation-copy";
    const name = document.createElement("span");
    name.className = "conversation-name";
    name.textContent = displayName(conversation);
    const subtitle = document.createElement("span");
    subtitle.className = "conversation-subtitle";
    subtitle.textContent = conversation.mode === "role-play" ? "角色扮演" : "Assistant";
    copy.append(name, subtitle);
    button.append(icon, copy);
    button.addEventListener("click", () => openConversation(conversation.conversation_id));
    conversationList.append(button);
  }
}

function renderMessages() {
  messagesElement.replaceChildren();
  if (!activeConversation) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = '<div class="empty-icon" aria-hidden="true">✦</div><h2>从一段对话开始</h2><p>新建一个助手对话，或从左侧打开已有对话。</p>';
    messagesElement.append(empty);
    return;
  }

  if (!activeMessages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    const isRolePlay = activeConversation.mode === "role-play";
    empty.innerHTML = isRolePlay
      ? '<div class="empty-icon" aria-hidden="true">✦</div><h2>角色对话已准备好</h2><p>发送第一条消息，开始这段角色扮演。</p>'
      : '<div class="empty-icon" aria-hidden="true">✦</div><h2>你好</h2><p>告诉 Assistant 你想做什么，或让它帮你创建角色对话。</p>';
    messagesElement.append(empty);
    return;
  }

  for (const message of activeMessages) {
    appendMessage(message.role, message.content);
  }
  messagesElement.scrollTop = messagesElement.scrollHeight;
}

function appendMessage(role, content, className = "") {
  const row = document.createElement("article");
  row.className = `message ${role} ${className}`.trim();
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "我" : "A";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = content;
  row.append(avatar, bubble);
  messagesElement.append(row);
  messagesElement.scrollTop = messagesElement.scrollHeight;
  return row;
}

function updateHeader() {
  newConversationButton.disabled = sending;
  conversationList.querySelectorAll(".conversation-item").forEach((button) => {
    button.disabled = sending;
  });
  if (!activeConversation) {
    kindElement.textContent = "未选择对话";
    titleElement.textContent = "欢迎";
    messageInput.disabled = true;
    sendButton.disabled = true;
    return;
  }
  kindElement.textContent = activeConversation.mode === "role-play" ? "角色扮演" : "ASSISTANT";
  titleElement.textContent = displayName(activeConversation);
  messageInput.placeholder = activeConversation.mode === "role-play"
    ? `对 ${displayName(activeConversation)} 说点什么…`
    : "给 Assistant 发消息…";
  messageInput.disabled = sending;
  sendButton.disabled = sending;
}

async function refreshConversations() {
  const result = await api("/conversations");
  conversations = result.conversations || [];
  renderConversationList();
}

async function openConversation(conversationId) {
  headerStatus.textContent = "正在打开…";
  try {
    const conversation = await api(`/conversations/${encodeURIComponent(conversationId)}`);
    activeConversation = conversation;
    activeMessages = conversation.messages || [];
    localStorage.setItem("activeConversationId", conversationId);
    renderConversationList();
    updateHeader();
    renderMessages();
    headerStatus.textContent = "";
  } catch (error) {
    headerStatus.textContent = `打开失败：${error.message}`;
  }
}

async function createAssistantConversation() {
  newConversationButton.disabled = true;
  headerStatus.textContent = "正在创建…";
  try {
    const conversation = await api("/conversations", { method: "POST" });
    await refreshConversations();
    await openConversation(conversation.conversation_id);
  } catch (error) {
    headerStatus.textContent = `创建失败：${error.message}`;
  } finally {
    newConversationButton.disabled = false;
  }
}

async function sendMessage(content) {
  if (!activeConversation || sending || !content.trim()) return;
  const conversationId = activeConversation.conversation_id;
  activeMessages = [...activeMessages, { role: "user", content }];
  renderMessages();
  sending = true;
  updateHeader();
  headerStatus.textContent = "Assistant 正在处理…";
  const pending = appendMessage("assistant", "正在思考…", "pending");

  try {
    const result = await api(
      `/conversations/${encodeURIComponent(conversationId)}/messages`,
      { method: "POST", body: JSON.stringify({ content }) },
    );
    pending.remove();
    if (result.reply && activeConversation?.conversation_id === conversationId) {
      activeMessages = [...activeMessages, { role: "assistant", content: result.reply }];
      renderMessages();
    }
    await refreshConversations();
    headerStatus.textContent = "";

    const created = result.created_conversations || [];
    if (created.length) {
      const newest = created[created.length - 1];
      headerStatus.textContent = `已创建角色对话：${newest.character_name || "角色"}，正在打开…`;
      await refreshConversations();
      await openConversation(newest.conversation_id);
    }
  } catch (error) {
    pending.remove();
    if (activeConversation?.conversation_id === conversationId) {
      activeMessages = activeMessages.filter((message, index) =>
        !(index === activeMessages.length - 1 && message.role === "user" && message.content === content),
      );
      renderMessages();
      appendMessage("assistant", `请求失败：${error.message}`, "message-error");
    }
    headerStatus.textContent = "";
  } finally {
    sending = false;
    updateHeader();
    messageInput.focus();
  }
}

newConversationButton.addEventListener("click", createAssistantConversation);

messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = messageInput.value.trim();
  if (!content) return;
  messageInput.value = "";
  messageInput.style.height = "auto";
  await sendMessage(content);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    messageForm.requestSubmit();
  }
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
});

async function start() {
  try {
    await api("/health");
    setConnection(true, "后端已连接");
    await refreshConversations();
    const savedId = localStorage.getItem("activeConversationId");
    const existing = conversations.find((item) => item.conversation_id === savedId)
      || conversations[0];
    if (existing) {
      await openConversation(existing.conversation_id);
    } else {
      await createAssistantConversation();
    }
  } catch (error) {
    setConnection(false, "后端未连接");
    headerStatus.textContent = `无法连接后端：${error.message}`;
    renderMessages();
  }
}

start();
