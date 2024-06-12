from PIL import Image, ImageOps
from io import BytesIO
from flask import Flask, render_template, request, send_file, url_for
import base64
import os
from sigma.pix2pixSingle import Sigma
import tensorflow as tf


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        # Retrieve the image data from the form
        image_data = request.form.get("image_data")
        if not image_data:
            return "No image data provided", 400

        # Decode the base64-encoded image data
        try:
            image_bytes = base64.b64decode(image_data.split(",")[1])
        except (IndexError, ValueError):
            return "Invalid image data format", 400

        temp_img_path = os.path.join("static", "drawing.png")
        with Image.open(BytesIO(image_bytes)) as img:
            # make transparent color white 
            img = img.convert("RGBA")
            datas = img.getdata()
            new_data = []

            for item in datas:
                if item[3] == 0:
                    new_data.append((255, 255, 255, 255))
                else:
                    new_data.append(item)
            
            img.putdata(new_data)
            img = img.convert("RGB")

            # invert colors
            img = ImageOps.invert(img)
            img.save(temp_img_path)
            print("Image saved")


        # Perform image processing here
        sigma = Sigma()
        generator = sigma.generator()
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        print("save checkpoint")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
        checkpoint_dir = 'sigma/training_checkpoints-cars'
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        print("generate image")
        img_name = "drawing"
        output_path = sigma.generate_single_image(generator, temp_img_path, img_name)
        
        print("generated image")

        with Image.open(output_path) as out_img:
            # Save the edited image to a file
            out_img.save(output_path)
            print("generated image saved")

        # Create URLs for the before and after images
        before_image_url = url_for("static", filename="drawing.png")
        after_image_url = url_for("static", filename="result_drawing.jpg")

        return render_template("result.html", before_image=before_image_url, after_image=after_image_url, download_url=after_image_url)


    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
