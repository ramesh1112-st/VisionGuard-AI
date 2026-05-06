from flask import Flask, render_template, request
import subprocess
import os
import sys

app = Flask(__name__)

UPLOAD_FOLDER = "video"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template("index.html", result=None)


@app.route('/run', methods=['POST'])
def run():
    if 'file' not in request.files:
        return render_template("index.html", result="No file uploaded")

    file = request.files['file']

    if file.filename == "":
        return render_template("index.html", result="No file selected")

    # ✅ Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print("Saved file:", filepath)

    # ✅ Clear old results
    open("submission.txt", "w").close()

    try:
        # ✅ Run inference with correct Python (venv)
        subprocess.run([sys.executable, "inference.py", filepath], check=True)

    except subprocess.CalledProcessError as e:
        print("Inference error:", e)
        return render_template("index.html", result="Error during inference")

    # ✅ Read result
    result_text = ""
    filename = os.path.splitext(file.filename)[0]

    if os.path.exists("submission.txt"):
        with open("submission.txt", "r") as f:
            for line in f:
                if filename in line:
                    result_text = line.strip()
                    break

    print("FINAL RESULT:", result_text)

    # ✅ If still empty, show fallback
    if not result_text:
        result_text = "No result generated"

    return render_template("index.html", result=result_text)


if __name__ == "__main__":
    app.run(debug=True)