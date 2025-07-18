const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter") sendMessage();
});

function sendMessage() {
  const message = userInput.value.trim();
  if (message === "") return;

  addMessage("user", message);
  userInput.value = "";

  // Fake bot reply (thay bằng gọi API thực tế ở đây)
  setTimeout(() => {
    const reply = `Bạn hỏi "${message}", tôi đang xử lý...`;
    addMessage("bot", reply);
  }, 600);
}

function addMessage(sender, text) {
  const messageElem = document.createElement("div");
  messageElem.className = `message ${sender === "user" ? "user-message" : "bot-message"}`;
  messageElem.textContent = text;
  chatBox.appendChild(messageElem);
  chatBox.scrollTop = chatBox.scrollHeight;
}
