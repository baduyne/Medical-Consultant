const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

// Thêm tin nhắn vào giao diện
function appendMessage(role, message) {
    const msg = document.createElement("div");
    msg.classList.add("message", role);
    msg.innerText = message;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Khoá input trong lúc chờ phản hồi
function setInputState(enabled) {
    userInput.disabled = !enabled;
    sendButton.disabled = !enabled;
}

let waiting = false;

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || waiting) return; // Không gửi nếu trống hoặc đang chờ

    appendMessage("user", message);
    userInput.value = "";
    setInputState(false); // Vô hiệu hóa input và nút gửi
    waiting = true; // Đặt cờ chờ

    try {
        // Gửi request và chờ response (sẽ chờ cho đến khi có phản hồi hoặc timeout)
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) throw new Error("Phản hồi lỗi");

        const data = await response.json();
        appendMessage("bot", data.response || "Không có nội dung trả về.");
    } catch (err) {
        console.error("Lỗi:", err);
        appendMessage("bot", "Lỗi kết nối hoặc backend không phản hồi.");
    } finally {
        setInputState(true); // Kích hoạt lại input và nút gửi
        waiting = false; // Reset cờ chờ
    }
}

sendButton.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault(); // Ngăn hành vi mặc định của Enter (ví dụ: xuống dòng)
        sendMessage();
    }
});