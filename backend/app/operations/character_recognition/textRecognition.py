import os
from paddleocr import TextRecognition

def recognize_and_save(image_path, save_img_path, model_name):
    model = TextRecognition(model_name=model_name)

    # Ensure output directory exists
    os.makedirs(save_img_path, exist_ok=True)

    # Run prediction
    output = model.predict(input=image_path, batch_size=1)

    # Process and save results
    for res in output:
        res.print()
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        #res.save_to_img(save_path=os.path.join(save_img_path, f"{img_name}_output.png"))
        json_path = os.path.join(save_img_path, f"{img_name}.json")
        res.save_to_json(save_path=json_path)
        print(f"Saved JSON to {json_path}")
