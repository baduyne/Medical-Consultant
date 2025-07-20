function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const userText = input.value.trim();
    if (userText === "") return;
  
    // Tin nhắn người dùng
    const userMessage = document.createElement("div");
    userMessage.className = "message user";
    userMessage.innerHTML = `<div class="bubble user">${userText}</div>`;
    chatBox.appendChild(userMessage);
  
    // Giả lập phản hồi từ bot
    const botMessage = document.createElement("div");
    botMessage.className = "message bot";
    botMessage.innerHTML = `<div class="bubble bot">Bạn đã ghi: ${userText}</div>`;
    chatBox.appendChild(botMessage);
  
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  