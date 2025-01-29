function sendMessage() {
    let inputField = document.getElementById("user-input");
    let query = inputField.value.trim();

    if (query === "") return;

    let chatBox = document.getElementById("chat-box");

    // Displaying User Message
    chatBox.innerHTML += `<p class="user-message"><b>You:</b> ${query}</p>`;
    inputField.value = "";

    // Sending to Flask API
    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<p class="bot-message"><b>Bot:</b> ${data.answer}</p>`;
        chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll
    });
}
