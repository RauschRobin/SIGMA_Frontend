from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, send_file, url_for
import base64
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    # Retrieve the image data from the form
    image_data = request.form.get("image_data")

    # Decode the base64-encoded image data
    image_bytes = base64.b64decode(image_data.split(",")[1])

    # Process the image using Pillow (PIL)
    with Image.open(BytesIO(image_bytes)) as img:
        img.save("static/unedited_image.png")

        # Perform image processing here (e.g., resizing, filtering, etc.)
        img = img.rotate(90)  # Example: Rotate the image by 90 degrees

        # Save the edited image to a file
        edited_image_path = "static/edited_image.png"
        img.save(edited_image_path)

    # Create URLs for the before and after images
    before_image_url = url_for("static", filename="unedited_image.png")
    after_image_url = url_for("static", filename="edited_image.png")

    return render_template("result.html", before_image=before_image_url, after_image=after_image_url, download_url=after_image_url)

if __name__ == "__main__":
    app.run(debug=True)
