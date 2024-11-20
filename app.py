from flask import Flask, request, jsonify
from PIL import Image
import os
import remwm  # Ensure remwm.py is in the same directory or in the PYTHONPATH
import torch

app = Flask(__name__)

# Function to remove watermark
def remove_watermark(input_image):
    # Load the Florence2 model and processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    florence_model = remwm.AutoModelForCausalLM.from_pretrained(
        'microsoft/Florence-2-large', trust_remote_code=True
    ).to(device)
    florence_model.eval()
    florence_processor = remwm.AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

    # Load LaMa model
    model_manager = remwm.ModelManager(name="lama", device=device)

    # Get watermark mask
    mask_image = remwm.get_watermark_mask(input_image, florence_model, florence_processor, device)

    # Process image with LaMa
    result_image = remwm.process_image_with_lama(np.array(input_image), np.array(mask_image), model_manager)

    # Convert result from BGR to RGB and save
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_image_pil = Image.fromarray(result_image_rgb)

    return result_image_pil

@app.route('/remove-watermark', methods=['POST'])
def remove_watermark_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Open the image
        input_image = Image.open(file).convert("RGB")
        # Remove watermark
        result_image = remove_watermark(input_image)

        # Save the result to a temporary location
        output_path = 'output_image.png'
        result_image.save(output_path)

        return jsonify({"message": "Watermark removed successfully.", "output_image_path": output_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
