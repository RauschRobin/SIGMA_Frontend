document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let drawing = false;
    let penMode = true;

    const penTool = document.getElementById("pen-tool");
    const eraserTool = document.getElementById("eraser-tool");

    // Get the clear button element
    const clearButton = document.getElementById("clear-button");

    clearButton.addEventListener("click", function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });

    penTool.addEventListener("click", function () {
        penMode = true;
        penTool.classList.add("active");
        eraserTool.classList.remove("active");
        ctx.globalCompositeOperation = "source-over"; // Set to draw mode
        ctx.lineWidth = 1;
    });

    eraserTool.addEventListener("click", function () {
        penMode = false;
        eraserTool.classList.add("active");
        penTool.classList.remove("active");
        ctx.globalCompositeOperation = "destination-out"; // Set to erase mode (make transparent)
        ctx.lineWidth = 10;
    });

    // Add the "active" class to the pen tool initially
    penTool.classList.add("active");

    canvas.addEventListener("mousedown", function (e) {
        drawing = true;
        ctx.beginPath();
        ctx.moveTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
    });

    canvas.addEventListener("mousemove", function (e) {
        if (drawing) {
            ctx.lineTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
            ctx.stroke();
        }
    });

    canvas.addEventListener("mouseup", function () {
        drawing = false;
    });

    document.getElementById("save-button").addEventListener("click", function () {
        const imageData = canvas.toDataURL(); // Convert canvas to a data URL
        const imageInput = document.getElementById("image-data");
        imageInput.value = imageData;
    
        // Submit the form to send the image data to the server
        document.getElementById("image-form").submit();
    });    
});
