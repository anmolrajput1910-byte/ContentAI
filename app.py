from flask import Flask, render_template, request
from model import generate_script

app = Flask(__name__)

from datetime import datetime

@app.route("/", methods=["GET", "POST"])
def home():
    script = ""

    if request.method == "POST":
        topic = request.form.get("topic", "motivation")
        script = generate_script(topic)

        # 🔢 METRIC LOG
        print(f"[{datetime.now()}] Script generated | Topic: {topic}")

    return render_template("index.html", script=script)

if __name__ == "__main__":
    app.run(debug=True)
